import logging
import time
from collections import OrderedDict
from typing import Any

import hydra
import numpy as np
import torch
from catalyst import dl, utils
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader

from mood_tagger import get_architecture
from mood_tagger.data import load_data, FeatureSetup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEAT_KEY = 0
TARG_KEY = 1


class CustomRunner(dl.Runner):
    def __init__(self, feature_key: Any, target_key: Any, *args, **kwargs):
        self.feature_key = feature_key
        self.target_key = target_key

        super().__init__(*args, **kwargs)

    @property
    def logger(self) -> Any:
        return logger

    def predict_batch(self, batch):
        # model inference step
        # print(f"inference batch len: {len(batch[self.feature_key])} \n"
        #       f"inference sample shape {batch[self.feature_key].shape}")
        # dict:
        return self.model(batch[self.feature_key].to(self.engine.device))

    def handle_batch(self, batch):
        # model train/valid step
        # dict:

        x = batch[self.feature_key]
        y = batch[self.target_key]

        # print(f"train batch len: {len(batch[self.feature_key])} \n"
        #       f"train sample shape {batch[self.feature_key][0].shape}")
        logits = self.model(x)
        self.batch = {"features": x, "targets": y, "logits": logits}


def run_training(cfg: DictConfig) -> None:
    time_t0 = time.time()
    architecture, architecture_file = get_architecture(cfg["model"]["architecture"])
    use_audio_in = cfg["model"]["audio-inputs"]
    gpu_num = 0
    k_samples = cfg["training"]["k_samples"]
    batch_size = cfg["training"]["batch_size"]
    val_size = cfg["training"]["validation_size"]
    num_pt_workers = cfg["training"]["num_data_threads"]
    max_epochs = cfg["training"]["max_num_epochs"]

    valid_fm_thresh = cfg["training"]["valid_fm_thresh"]
    memory_map = cfg["training"]["memory_map"]
    patience = cfg["training"]["patience"]
    refinements = cfg["training"]["refinements"]

    seq_len = cfg["training"]["sequence-length"]
    hop_size = cfg["training"]["sequence-hop"]
    feat_sample_rate = cfg["features"]["sample_rate"]
    feat_frame_rate = cfg["features"]["frame_rate"]
    feat_window_size = cfg["features"]["window_size"]

    learning_rate = cfg["training"]["learning-rate"]

    num_key_classes = 24  # 24 keys ?
    num_spec_bins = 100
    num_channels = 1
    sample_rate = 44100

    fft_hop_size = feat_sample_rate / feat_frame_rate

    snippet_duration_s = (fft_hop_size * (seq_len - 1) + feat_window_size) / feat_sample_rate
    snippet_hop_s = fft_hop_size * (hop_size - 1) / feat_sample_rate

    model = architecture(audio_input=use_audio_in, num_key_classes=num_key_classes, debug=True)
    if use_audio_in:
        in_size = (batch_size, int((seq_len - 1) * (feat_sample_rate / feat_frame_rate) + feat_window_size))
    else:
        in_size = (batch_size, seq_len, num_spec_bins)
    output = model.forward(torch.Tensor(np.zeros(in_size)))
    print(output.shape)

    model = architecture(audio_input=use_audio_in, num_key_classes=num_key_classes)

    feat_settings = FeatureSetup(
        "mood_feat", feat_sample_rate, 1, feat_frame_rate, feat_window_size, 100, 30, False, "log", "log_1"
    )

    test_loader, train_loader, valid_loader = load_data(batch_size,
                                                        feat_settings,
                                                        gpu_num,
                                                        hop_size,
                                                        k_samples,
                                                        num_pt_workers,
                                                        seq_len,
                                                        use_audio_in,
                                                        val_size,
                                                        valid_fm_thresh,
                                                        model.is_sequential())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    from catalyst.utils.torch import get_available_engine

    engine = get_available_engine()
    engine.dispatch_batches = False
    runner = CustomRunner(FEAT_KEY, TARG_KEY, engine=engine)

    loaders = OrderedDict(
        {
            "train": train_loader,
            "valid": valid_loader,
            # "infer": test_loader,
        }
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15])
    early_stopping = True

    callbacks = [dl.CriterionCallback(input_key="logits", target_key="targets", metric_key="loss"),
                 dl.BackwardCallback(metric_key="loss"), dl.OptimizerCallback(metric_key="loss"),
                 dl.CheckpointCallback("./logs", loader_key="valid", metric_key="loss", minimize=True, topk=3),

                 ]
    if early_stopping:
        callbacks.append(
            dl.EarlyStoppingCallback(patience=patience, loader_key="valid", metric_key="loss", minimize=True))

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        scheduler=scheduler,
        logdir="./logs",
        num_epochs=max_epochs,  # if not early_stopping else -1,
        verbose=True,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=callbacks,
        # timeit=True,
    )

    # model evaluation
    # model.debug = True
    metrics = runner.evaluate_loader(
        model=model,
        loader=test_loader,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1, 3)),
            dl.PrecisionRecallF1SupportCallback(input_key="logits", target_key="targets", num_classes=num_key_classes),
        ],
    )
    print(f"Metrics for test {metrics}")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    run_training(cfg)


if __name__ == "__main__":
    my_app()
