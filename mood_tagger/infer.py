from omegaconf import OmegaConf
import os
# from hydra import compose
import json

import torch

from mood_tagger import test_model, get_architecture
from mood_tagger.data import load_data, FeatureSetup, NUM_CLASSES


def run():
    # 16-48-26 > conv_lin
    # 19-52-45 > conv_relu (better)
    # 21-06-42 > conv_02 -> 6 seconds 150
    # 21-11-13 > conv_02 -> 16s 500 seqlen

    run_path = '/home/rvogl/python-workspace/humrec_mood_tagger/outputs/2023-01-31/19-52-45'
    hydra_run_path = os.path.join(run_path, '.hydra')
    config_path = os.path.join(hydra_run_path, 'config.yaml')
    # overrides_path = os.path.join(hydra_run_path, 'overrides.yaml')
    cfg = OmegaConf.load(config_path)
    # overrides = OmegaConf.load(overrides_path)

    catalyst_out_dir = os.path.join(run_path, 'catalyst')
    model_out_dir = os.path.join(run_path, 'models')

    model_json_file_path = os.path.join(model_out_dir, 'model.storage.json')

    with open(model_json_file_path, 'r') as json_file:
        data = json.load(json_file)

    best_model_path = data['storage'][0]['logpath']

    architecture, architecture_file = get_architecture(cfg.model.architecture, model_out_dir)

    model = architecture(audio_input=cfg.model.audio_inputs, num_classes=NUM_CLASSES, debug=False, **cfg.features)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    feat_settings = FeatureSetup(
        "mood_feat", cfg.features.sample_rate, 1, cfg.features.frame_rate, cfg.features.window_size, 100, 30, False, "log", "log_1"
    )
    gpu_num = 0
    test_loader, train_loader, valid_loader = load_data(cfg.training.batch_size,
                                                        feat_settings,
                                                        gpu_num,
                                                        cfg.training.sequence_hop,
                                                        cfg.training.k_samples,
                                                        cfg.training.num_data_threads,
                                                        cfg.training.sequence_length,
                                                        cfg.model.audio_inputs,
                                                        cfg.training.validation_size,
                                                        model.is_sequential())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_model(model, NUM_CLASSES, test_loader, device, plot=True, model_name=cfg.model.architecture)


if __name__ == "__main__":
    run()
