
#%%
from omegaconf import OmegaConf
import os
# from hydra import compose
import json
import numpy as np
import torch

from __init__ import test_model, get_architecture
from data import load_data, FeatureSetup, NUM_CLASSES

def run(run_path = None):
    # 16-48-26 > conv_lin
    # 19-52-45 > conv_relu (better)
    # 21-06-42 > conv_02 -> 6 seconds 150
    # 21-11-13 > conv_02 -> 16s 500 seqlen
    if run_path is None:
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-12-12/09-57-31' #best oversampling aug percentile 0.39
        run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-12-06/00-23-04' #best oversampling aug
        #run_path = "/home/ykinoshita/humrec_mood_tagger/outputs/2023-12-05/15-03-09"
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-11-22/10-07-19'

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

    model = architecture(audio_input=cfg.model.audio_inputs, num_classes=len(cfg.datasets.labels), debug=False, **cfg.features)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    feat_settings = FeatureSetup(
        "mood_feat", cfg.features.sample_rate, 1, cfg.features.frame_rate, cfg.features.window_size, cfg.features.freq_bins, 30, False, "log", "log_1"
    )
    gpu_num = 0
    scale = cfg.datasets.scale
    if cfg.model.architecture == 'FCN_6_layer_classifier':
        task = 'classification'
    else:
        task = 'regression'
    test_loader, train_loader, valid_loader, train_annot, targets_all   = load_data(cfg.training.batch_size,
                                                        feat_settings,
                                                        gpu_num,
                                                        cfg.training.sequence_hop,
                                                        cfg.training.k_samples,
                                                        cfg.training.num_data_threads,
                                                        cfg.training.sequence_length,
                                                        cfg.model.audio_inputs,
                                                        cfg.training.validation_size,
                                                        model.is_sequential(),
                                                        scale = scale,
                                                        mode = "test",
                                                        train_y = cfg.datasets.labels,
                                                        task = task) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #TODO metrics
    fft_hop_size = cfg.features.sample_rate / cfg.features.frame_rate
    snippet_duration_s = (fft_hop_size * (cfg.training.sequence_length - 1) + cfg.features.window_size) /cfg.features.sample_rate
    if cfg.datasets.loss_func != 'dense_weight':
        alpha = 0
    else:
        alpha = cfg.datasets.dense_weight_alpha

    csv_information = {
                'Model' : cfg.model.architecture,
                #'resampling' : cfg.resampling,
                'Labels' : [cfg.datasets.labels],
                'lr' : cfg.training.learning_rate,
                'loss_function' : cfg.datasets.loss_func,
                'dense_weight_alpha': alpha ,
                'batch size' : cfg.training.batch_size,
                'snippet_duration_s' : snippet_duration_s,
                'oversampling': cfg.datasets.oversampling,
                'oversampling_tolerance': cfg.datasets.oversampling_tolerance,
                'oversampling_ratio' : cfg.datasets.oversampling_ratio,
                'oversampling_method': cfg.datasets.oversampling_method,
                'data_augmentation': cfg.datasets.data_augmentation


            }
    test_model(model, NUM_CLASSES, test_loader, device, plot=True, model_name=cfg.model.architecture, transform = cfg.training.transformation, scale = scale , csv_information = csv_information, task = task, testing = True)


if __name__ == "__main__":
    
    run()

# %%
