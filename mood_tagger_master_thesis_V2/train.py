import logging
import time
from collections import OrderedDict
from typing import Any
import os
import shutil

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from catalyst import dl, utils
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim

from catalyst.engines.torch import GPUEngine


from __init__  import get_architecture, test_model
from data import load_data, FeatureSetup, NUM_CLASSES
#export PYTHONPATH="$PWD/"
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


def run_training(cfg: DictConfig, GEMS_9 = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']) -> None:
    torch.cuda.empty_cache()
    time_t0 = time.time()
    architecture, architecture_file = get_architecture(cfg.model.architecture)
    use_audio_in = cfg.model.audio_inputs
    gpu_num = 0
    k_samples = cfg.training.k_samples
    batch_size = cfg.training.batch_size
    val_size = cfg.training.validation_size
    num_pt_workers = cfg.training.num_data_threads
    max_epochs = cfg.training.max_num_epochs

    memory_map = cfg.training.memory_map
    patience = cfg.training.patience
    refinements = cfg.training.refinements

    seq_len = cfg.training.sequence_length
    hop_size = cfg.training.sequence_hop
    feat_sample_rate = cfg.features.sample_rate
    feat_frame_rate = cfg.features.frame_rate
    feat_window_size = cfg.features.window_size

    learning_rate = cfg.training.learning_rate
    transform = cfg.training.transformation
    scale = cfg.datasets.scale

    oversampling = cfg.datasets.oversampling
    tolerance = cfg.datasets.oversampling_tolerance

    hydra_base_dir = HydraConfig.get().runtime.output_dir
    catalyst_out_dir = os.path.join(hydra_base_dir, 'catalyst')
    model_out_dir = os.path.join(hydra_base_dir, 'models')
    os.makedirs(catalyst_out_dir)
    os.makedirs(model_out_dir)
    print(f"Output dir: {hydra_base_dir}")

    shutil.copy(architecture_file, os.path.join(model_out_dir, os.path.split(architecture_file)[1]))
    if isinstance(GEMS_9, str):
        num_classes = 1
    else:
        num_classes = NUM_CLASSES
    num_spec_bins = 100
    num_channels = 1

    fft_hop_size = feat_sample_rate / feat_frame_rate

    #    Sequence Length: The number of samples (or time steps) contained in each frame of the spectrogram.
    #    Hop Size: The number of samples between the starting points of consecutive frames (segments) in the spectrogram.
    #    Sample Rate: The number of samples per second in the audio signal.

                            #+ window size weil letztes sample kein overlap hat
    snippet_duration_s = (fft_hop_size * (seq_len - 1) + feat_window_size) / feat_sample_rate
    print(f'snippet duration{snippet_duration_s} ')
    snippet_hop_s = fft_hop_size * (hop_size ) / feat_sample_rate

    model = architecture(audio_input=use_audio_in, num_classes=num_classes, debug=True, **cfg.features)
    if use_audio_in:
        in_size = (batch_size, int((seq_len - 1) * (feat_sample_rate / feat_frame_rate) + feat_window_size))
    else:
        in_size = (batch_size, seq_len, cfg.features.freq_bins)
    output = model.forward(torch.Tensor(np.zeros(in_size)))
    print(output.shape)

    model = architecture(audio_input=use_audio_in, num_classes=num_classes, **cfg.features)

    feat_settings = FeatureSetup(
        "mood_feat", feat_sample_rate, 1, feat_frame_rate, feat_window_size, cfg.features.freq_bins, 30, False, "log", "log_1"
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
                                                        model.is_sequential(),
                                                        transform = transform,
                                                        scale = scale,
                                                        mode = "train",
                                                        oversampling = oversampling,
                                                        tolerance = tolerance,
                                                        train_y = GEMS_9) 

    criterion = nn.MSELoss() #torch.nn.SmoothL1Loss()   nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    from catalyst.utils.torch import get_available_engine

    # engine = get_available_engine()
    engine = GPUEngine()#fp16=False
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
                 dl.CheckpointCallback(model_out_dir, loader_key="valid", metric_key="loss", minimize=True, topk=3),

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
        logdir=catalyst_out_dir,
        num_epochs=max_epochs,
        verbose=True,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=callbacks,
    )
    if isinstance(GEMS_9, str):
        test_model(model, num_classes, test_loader, engine.device, transform = cfg.training.transformation, training = 'training', scale = scale, model_name = cfg.model.architecture , train_y = GEMS_9)
    else:
        test_model(model, num_classes, test_loader, engine.device, transform = cfg.training.transformation, training = 'training', scale = scale, model_name = cfg.model.architecture )


@hydra.main(version_base=None, config_path="configs", config_name="default")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    run_training(cfg)


if __name__ == "__main__":
    my_app()
