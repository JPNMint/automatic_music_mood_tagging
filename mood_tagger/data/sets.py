import logging
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset as Dataset
from torch.utils.data.dataset import T_co

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FramesToSamples:
    def __init__(self, hop_size: int = 441, window_size: int = 2048):
        self.hop_size = hop_size
        self.window_size = window_size

    def __call__(self, num_frames: int, do_length: bool) -> int:
        if do_length:
            return self.frames_to_length(num_frames)
        else:
            return self.frames_to_samples(num_frames)

    def frames_to_samples(self, num_frames: int) -> int:
        return num_frames * self.hop_size

    def frames_to_length(self, num_frames: int) -> int:
        return (num_frames - 1) * self.hop_size + self.window_size


def frames_to_samples(num_frames: int, do_length: bool) -> int:
    ...


class SnippetDataset(Dataset):
    def __init__(
        self,
        feat_list: list,
        targ_list: list,
        seq_len: int,
        hop_size: int,
        single_mean_targets: bool = False,
        zero_pad: bool = True,
        seq_len_dist_fun: frames_to_samples = None,
        balance_classes: bool = False,
        squeeze: bool = False,
        class_num_targ: bool = False,
    ):
        self.balance_classes = balance_classes
        self.single_mean_targets = single_mean_targets
        self.squeeze = squeeze
        self.class_num_targ = class_num_targ

        self.features = feat_list
        self.targets = targ_list
        self.seq_len = seq_len

        self.targ_dict = isinstance(targ_list[0], dict)
        self.hop_size = hop_size
        self.zero_pad = zero_pad
        self.seq_len_dist_fun = seq_len_dist_fun
        if seq_len_dist_fun is not None:
            if self.seq_len < 1:
                self.in_seq_len = -1
            else:
                self.in_seq_len = seq_len_dist_fun(self.seq_len, True)
            self.in_hop_size = seq_len_dist_fun(self.hop_size, False)
        else:
            self.in_seq_len = self.seq_len
            self.in_hop_size = self.hop_size

        if seq_len == 0:
            self.seq_len = 1

        self.snip_cnt = []
        total_len = 0

        self.snip_idxs = []

        if self.in_seq_len < 0 or all([self.in_seq_len > len(feat) for feat in feat_list]) and not self.zero_pad:
            # seq_len -1 means we use full sequences. if seq_len is larger than all tracks same.
            total_len = len(feat_list)
            self.snip_cnt = [1 for _ in range(total_len)]
            self.snip_idxs = [(idx, 0) for idx in range(total_len)]
            if not self.in_seq_len < 0:  # it was not intentionally set to full sequences...
                logger.info("Provided seq_len is longer than all sequences, using full sequences...")
        else:
            if any([self.in_seq_len > len(feat) for feat in feat_list]) and not self.zero_pad:
                logger.warning(
                    f"Some ({sum([self.in_seq_len > len(feat) for feat in feat_list])}/{len(feat_list)}) "
                    f"sequences are shorten than seq_len! Minimum sequence length is: "
                    f"{min([len(feat) for feat in feat_list])}"
                )

            min_snip = 1 if self.zero_pad else 0
            for feat_idx, feat in enumerate(feat_list):
                seq_len = feat.shape[0]
                cur_len = int(np.ceil(max(min_snip, seq_len - self.in_seq_len) / float(self.in_hop_size)))
                self.snip_cnt.append(cur_len)
                total_len += cur_len
                self.snip_idxs.extend([(feat_idx, s_idx) for s_idx in range(cur_len)])

        self.length = int(total_len)
        assert len(self.snip_idxs) == self.length == np.sum(self.snip_cnt)

        if self.balance_classes:
            self.analyze_targets()

    def analyze_targets(self):
        if self.targ_dict:
            num_classes = self.targets[0]["classes"].shape[1]
        else:
            num_classes = self.targets[0].shape[1]
        snip_classes = []
        snip_cls_dict = [[] for _ in range(num_classes)]
        snip_cls_counts = [[] for _ in range(num_classes)]
        for index in range(self.length):
            idx, h_idx = self.snip_idxs[index]
            targ_pos = h_idx * self.hop_size

            if self.in_seq_len > 0:
                targ_seq_len = self.seq_len
            else:
                if self.targ_dict:
                    targ_seq_len = self.targets[idx]["classes"].shape[0]
                else:
                    targ_seq_len = self.targets[idx].shape[0]

            cur_target = self.targets[idx]

            if self.targ_dict:
                target = cur_target["classes"][targ_pos : (targ_pos + targ_seq_len), :]
            else:
                target = cur_target[targ_pos : (targ_pos + targ_seq_len), :]

            class_counts = np.sum(target, axis=0)
            snip_classes.append(class_counts)
            nonzeros_idx = np.nonzero(class_counts)
            if len(nonzeros_idx[0]) == 1:
                cls_idx = nonzeros_idx[0][0]
                snip_cls_dict[cls_idx].append((idx, h_idx))
                snip_cls_counts[cls_idx].append(class_counts[cls_idx])

        total_counts = np.sum(np.asarray(snip_classes), axis=0)
        max_total = np.max(total_counts)
        missing = max_total - total_counts
        # available = np.sum(np.asarray(snip_cls_counts), axis=0)
        available = np.asarray([np.sum(cnts) for cnts in snip_cls_counts])
        adds = np.divide(missing, available, out=np.zeros_like(missing), where=available != 0)

        for cls_idx in range(num_classes):
            full = int(adds[cls_idx])
            frac = adds[cls_idx] - full
            for _ in range(full):
                self.snip_idxs.extend(snip_cls_dict[cls_idx])

            self.snip_idxs.extend(snip_cls_dict[cls_idx][: int(available[cls_idx] * frac)])

        self.length = len(self.snip_idxs)

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> T_co:
        idx, h_idx = self.snip_idxs[index]

        feat_pos = h_idx * self.in_hop_size
        targ_pos = h_idx * self.hop_size

        if self.in_seq_len > 0:
            feat_seq_len = self.in_seq_len
            targ_seq_len = self.seq_len
        else:
            assert feat_pos == 0
            feat_seq_len = self.features[idx].shape[0]
            if self.targ_dict:
                targ_seq_len = self.targets[idx]["classes"].shape[0]
            else:
                targ_seq_len = self.targets[idx].shape[0]

        cur_feat = self.features[idx]
        cur_target = self.targets[idx]

        sample = cur_feat[feat_pos : (feat_pos + feat_seq_len), :]
        sample = self.do_pad(feat_seq_len, sample)

        if self.targ_dict:
            target = {
                t_name: self.do_pad(targ_seq_len, t_data[targ_pos : (targ_pos + targ_seq_len), :])
                for t_name, t_data in cur_target.items()
            }
        else:
            target = cur_target[targ_pos : (targ_pos + targ_seq_len), :]
            target = self.do_pad(targ_seq_len, target)

        if self.single_mean_targets:
            target = np.mean(target, axis=-2)
            # target = target[None, :] # single mean targets usually have no seq len -
            # if so we need to retain it in the model output
            # but not sure if the catalyst metrics can handle this.

        if self.class_num_targ:
            target = np.asarray(np.argmax(target, axis=-1))
            if len(target.shape) > 0:
                target = target.squeeze()

        if self.squeeze:
            sample = sample.squeeze()

        warnings.filterwarnings(action="ignore", category=UserWarning)
        return torch.from_numpy(sample), torch.from_numpy(target)

    def do_pad(self, seq_len, sample):
        if self.zero_pad and seq_len > sample.shape[0]:
            to_pad = seq_len - sample.shape[0]
            pad_shape = list(sample.shape)
            pad_shape[0] = to_pad
            pads = np.zeros(pad_shape, dtype=sample.dtype)
            sample = np.concatenate((sample, pads), axis=0)
        return sample


class InfiniteBatchSampler:
    """Generates infinite batches.

    Args:
        data_size (int): Size of dataset.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        shuffle (bool): If ``True``, the sampler will shuffle the indices before generating batches.

    """

    def __init__(self, data_size, batch_size, shuffle=True):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integral value, " "but got batch_size={}".format(batch_size)
            )
        self.data_size = data_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            self.indices = torch.randperm(self.data_size).tolist()
        else:
            self.indices = range(self.data_size)
        self.stopped = False
        self.cur_idx = -1

    def __iter__(self):
        batch = []
        while not self.stopped:
            batch.append(self.next())
            if len(batch) == self.batch_size:
                # print('generated batch idx is ' + str(self.cur_idx))
                yield batch
                batch = []

    def __len__(self):
        return (self.data_size + self.batch_size - 1) // self.batch_size

    def next(self):
        self.cur_idx = self.cur_idx + 1
        if self.cur_idx >= self.data_size:
            self.cur_idx = 0
            if self.shuffle:
                self.indices = torch.randperm(self.data_size).tolist()
            # print('data through!')

        return self.indices[self.cur_idx]

    def stop(self):
        self.stopped = True
