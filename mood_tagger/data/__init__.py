import pandas as pd
import numpy as np
import os
import librosa

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mood_tagger.data.sets import SnippetDataset, FramesToSamples, ListDataset
from torch.utils.data import DataLoader

SAMPLE_RATE = 44100

DATA_PATH = os.path.dirname(os.path.realpath(__file__))

# TODO move data out of here
AUDIO_PATH = os.path.join(DATA_PATH, 'audio')
CACHE_PATH = os.path.join(DATA_PATH, 'cache')
ANNOT_CSV_FILE = os.path.join(DATA_PATH, "GEMS-INN_2023-01-30_expert.csv")

os.makedirs(CACHE_PATH, exist_ok=True)

GEMS_9 = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
NUM_CLASSES = len(GEMS_9)
GENRE_MAP = {'H': 'Hip-Hop', 'K': 'Classical', 'P': 'Pop'}


class FeatureSetup:
    # TODO fix this mess :)
    def __init__(
        self,
        name: str,
        sample_rate: int,
        channels: int,
        fps: float,
        frame_size: int,
        num_bins: int,
        fmin: float,
        diff: bool,
        freq_scale: str,
        mag_scale: str,
        **kwargs,
    ):
        self.name = name

        self.sample_rate = sample_rate
        self.channels = channels
        self.fps = fps
        self.hop_size = int(sample_rate / fps)
        self.frame_size = frame_size
        self.num_bins = num_bins
        self.fmin = fmin
        self.diff = diff
        self.freq_scale = freq_scale
        self.mag_scale = mag_scale

        self.feat_size = np.sum(num_bins) * (2 if diff else 1)

        self.__dict__.update(kwargs)


def remove_special_chars(name_with_chars):
    output_name = name_with_chars
    output_name = output_name.replace('3. Klavierkonzert, 3. Satz', '3KK3Sat')
    output_name = output_name.replace('3. Klaviersonate, 4. Satz', '3KS4Sat')
    output_name = output_name.replace('Klaviersonate No. 7', 'KSNo7')
    output_name = output_name.replace('Preludes op. 11 no. 1', 'Pre1')
    output_name = output_name.replace('Preludes op. 11 no. 15', 'Pre15')
    output_name = output_name.replace('Die 4 Jahreszeiten', '4Jahr')
    output_name = output_name.replace('Sinfonie', 'Sinf')
    output_name = output_name.replace(' ', '')
    output_name = output_name.replace(':', '-')
    output_name = output_name.replace('\'', '')
    output_name = output_name.replace('.', '')
    output_name = output_name.replace('!', '')
    output_name = output_name.replace('*', 'X')
    output_name = output_name.replace('&', 'and')
    output_name = output_name.replace('$', 'S')
    output_name = output_name.replace('\xfc', 'ue')
    output_name = output_name.replace(',', '')
    output_name = output_name.replace('(', '')
    output_name = output_name.replace(')', '')
    output_name = output_name.replace('é', 'e')
    output_name = output_name.replace('É', 'E')
    output_name = output_name.replace('ö', 'oe')
    output_name = output_name.replace('á', 'a')
    output_name = output_name.replace('ä', 'ae')

    return output_name

class Entry:
    def __init__(self, audio_file_name, data_row):
        self.genre_key = audio_file_name[:1]
        self.artist_name = data_row.artist
        self.track_name = data_row.title
        self.audio_file_name = audio_file_name
        self.data_row = data_row
        self.audio_data = None

    def _get_cache_path(self):
        base_name, _ = os.path.splitext(self.audio_file_name)
        return os.path.join(CACHE_PATH, base_name.lower() + '.npy')

    def _get_full_audio_path(self):
        return os.path.join(AUDIO_PATH, self.audio_file_name)

    def get_audio(self):
        if self.audio_data is None:
            cache_file = self.re_cache()

            self.audio_data = np.load(cache_file, mmap_mode='r')
        return self.audio_data

    def re_cache(self):
        cache_file = self._get_cache_path()
        if not os.path.exists(cache_file):
            full_file_name = self._get_full_audio_path()
            audio_data, sr = librosa.load(full_file_name, sr=SAMPLE_RATE)
            audio_data = audio_data[:, None]  # add feature dim
            np.save(cache_file, audio_data)
        return cache_file

    def get_gems_9(self):
        return self.data_row[GEMS_9].to_numpy(dtype=np.float32)

    def get_name(self):
        return f"{self.artist_name} {self.track_name}"

    def get_genre(self):
        return self.genre_key


def match_files():
    emma_df = pd.read_csv(ANNOT_CSV_FILE, encoding="ISO-8859-1")
    emma_df.dropna(inplace=True)
    audio_files = os.listdir(AUDIO_PATH)
    audio_files = [cur_file for cur_file in audio_files if cur_file.endswith('.mp3')]
    not_found_count = 0
    matched = 0

    entries = []
    for col_idx, col_series in emma_df.iterrows():
        artist_name = col_series.artist
        track_name = col_series.title
        if artist_name.lower().startswith('the '):
            artist_name = artist_name[4:]

        artist_name_ns = remove_special_chars(artist_name)
        track_name_ns = remove_special_chars(track_name)

        found = False

        for cur_audio_file in audio_files:
            audio_file_short = cur_audio_file[2:-4]
            # print(audio_file_short)
            if len(audio_file_short.split('_')) == 2:
                audio_artist_name, audio_track_name = audio_file_short.split('_')
                if audio_artist_name.lower().startswith(artist_name_ns.lower()) and \
                        track_name_ns[:7].lower().startswith(audio_track_name.lower()):
                    # audio_track_name.lower() == track_name_ns[:7].lower():
                    # print(f'    found! {artist_name}  {track_name} == {audio_file_short}')
                    found = True
                    matched += 1
                    entries.append(Entry(cur_audio_file, col_series))
                    break
            else:
                print(f"problem with {audio_file_short} - more than 2 parts")
        if not found:
            print(f"not found! {artist_name} {track_name}")
            not_found_count += 1

    print(f"not found count :  {not_found_count}   - matched: {matched}")

    return entries


class RandomPartitioner:
    def __init__(self, feats, annots, targs, names, genres, seed, test_split=0.15, valid_split=0.15):
        self.feats = feats
        self.annots = annots
        self.targs = targs
        self.names = names
        self.genres = genres
        self.seed = seed
        num_samples = len(feats)
        assert num_samples == len(annots) == len(targs) == len(names) == len(genres)
        import random as rand
        rand.seed(seed)
        indices = list(range(num_samples))
        rand.shuffle(indices)
        test_stop = int(num_samples * test_split)
        valid_stop = test_stop + int(num_samples * valid_split)
        self.test_indices = indices[:test_stop]
        self.valid_indices = indices[test_stop:valid_stop]
        self.train_indices = indices[valid_stop:]

        test_list = []
        test_list.extend(self.test_indices)
        test_list.extend(self.valid_indices)
        test_list.extend(self.train_indices)
        assert test_list == indices

    def get_split(self, split_name):
        if split_name == 'train':
            indices = self.train_indices
        elif split_name == 'valid':
            indices = self.valid_indices
        elif split_name == 'test':
            indices = self.test_indices
        else:
            raise ValueError(f"Unknown split name {split_name}")

        return ([self.feats[idx] for idx in indices],
                [self.annots[idx] for idx in indices],
                [self.targs[idx] for idx in indices],
                [self.names[idx] for idx in indices],
                [self.genres[idx] for idx in indices])


def load_data(
        batch_size,
        feat_settings,
        gpu_num,
        hop_size,
        k_samples,
        num_pt_workers,
        seq_len,
        use_audio_in,
        val_size,
        sequential
):
    entries = match_files()

    if use_audio_in:
        feats = [entry.get_audio() for entry in entries]
    else:
        feats = [entry.get_audio() for entry in entries]  # TODO calculate spectrogram here.
        raise NotImplemented()

    annots = [entry.get_gems_9() for entry in entries]
    targs = annots
    names = [entry.get_name() for entry in entries]
    genres = [entry.get_genre() for entry in entries]

    partitioner = RandomPartitioner(feats, annots, targs, names, genres, 42)

    valid_subset = None
    if val_size is not None and 0.0 < val_size < 1.0:
        valid_subset = val_size
    pin_memory = gpu_num > -1
    train_feats, train_annots, train_targs, train_names, train_genres = partitioner.get_split('train')
    frames_to_samples = FramesToSamples(hop_size=feat_settings.hop_size, window_size=feat_settings.frame_size)
    train_set = SnippetDataset(train_feats, train_targs,
                               seq_len,
                               hop_size,
                               single_targets=True,
                               seq_len_dist_fun=frames_to_samples,
                               squeeze=True,
                               )
    total_num_train_samples = len(train_set)
    # train_batch_sampler = InfiniteBatchSampler(data_size=total_num_train_samples, batch_size=batch_size,
    #                                            shuffle=True)
    train_loader = DataLoader(
        train_set, num_workers=num_pt_workers, pin_memory=pin_memory, batch_size=batch_size
    )  # batch_sampler=train_batch_sampler,
    if k_samples is None or k_samples <= 0 or k_samples > total_num_train_samples:
        k_samples = total_num_train_samples
    logging.info(f"Number of total training examples: {total_num_train_samples}.")
    # batches_per_epoch = k_samples // batch_size

    valid_seq_len = seq_len
    valid_batch_size = batch_size

    valid_feats, valid_annots, valid_targs, valid_names, valid_genres = partitioner.get_split('valid')
    valid_loader = DataLoader(
        SnippetDataset(
            valid_feats,
            valid_targs,
            valid_seq_len,
            hop_size,
            single_targets=True,
            seq_len_dist_fun=frames_to_samples,
            squeeze=True,
        ),
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_pt_workers,
        pin_memory=pin_memory,
    )
    test_feats, test_annots, test_targs, test_names, test_genres = partitioner.get_split('test')
    test_loader = DataLoader(
        ListDataset(test_feats, test_targs, test_annots, test_names, test_genres),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return test_loader, train_loader, valid_loader


def generate_plots():
    entries = match_files()
    annots = [entry.get_gems_9() for entry in entries]
    genres = [entry.get_genre() for entry in entries]

    gems_df = generate_dataframe(annots, genres)
    plot_data(gems_df, 'EMMA')


def generate_dataframe(annots, genres):
    d = []
    for cur_genre, cur_mat in zip(genres, annots):
        for gems_idx in range(9):
            d.append(
                {
                    'Score': cur_mat[gems_idx],
                    'Categories': GEMS_9[gems_idx],
                    'Genre': GENRE_MAP[cur_genre]
                }
            )
    gems_df = pd.DataFrame(d)
    return gems_df


def plot_data(gems_df, set_title='xx'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_path = os.path.join(DATA_PATH, "../../figures/")
    os.makedirs(fig_path, exist_ok=True)

    # overall plot of gems 9 distribution in dataset:
    plt.figure(figsize=(8, 6), dpi=200)
    sns.violinplot(gems_df, x='Categories', y='Score', cut=0)
    plt.xticks(rotation=45)
    plt.ylim([0, 55])
    plt.title(f'GEMS-9 distribution in {set_title}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, set_title.lower().replace(' ', '_') + '_overal_gems9_dist.png'))
    # plt.show()
    # per genre plot of gems 9 distribution
    plt.figure(figsize=(16, 6), dpi=200)
    sns.violinplot(gems_df, x='Categories', y='Score', hue='Genre', cut=0)
    plt.ylim([0, 55])
    plt.title(f'GEMS-9 per Genre in {set_title}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, set_title.lower().replace(' ', '_') + '_genre_gems9_dist.png'))
    # plt.show()


if __name__ == '__main__':
    generate_plots()
