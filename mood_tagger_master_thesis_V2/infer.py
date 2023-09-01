from omegaconf import OmegaConf
import os
# from hydra import compose
import json

import torch

from __init__ import test_model, get_architecture
from data import load_data, FeatureSetup, NUM_CLASSES


def run(run_path = None):
    # 16-48-26 > conv_lin
    # 19-52-45 > conv_relu (better)
    # 21-06-42 > conv_02 -> 6 seconds 150
    # 21-11-13 > conv_02 -> 16s 500 seqlen
    if run_path is None:
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-09-01/17-43-02' #allconv01 oversampling tol 20 Standard scaler
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-09-01/15-02-58' #allconv01 oversampling tol 20
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-09-01/14-22-24' #allconv01 oversampling tol 20 Robust scaling
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-09-01/14-17-25'#allconv01 oversampling tol 20 MinMax scaling
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-09-01/13-44-13' #allconv01 oversampling tol 20 AbsMax scaling

        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-08-30/19-43-25' #transfer learning correct oversampling tol 20
        #run_path = /home/ykinoshita/humrec_mood_tagger/outputs/2023-08-30/18-10-28 #musicnn correct oversampling tol 20
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-08-30/17-52-00' #allconv01 correct oversampling tol 20
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-08-16/17-45-34' #allconv02 correct oversampling tol 15
        #run_path =  '/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-31/15-01-14' 
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-31/13-31-05'  #transfer learnin parallel
        #'/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-19/14-01-40' #allconv01 15 sec log
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-18/16-44-58' #allconv complex log
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-18/16-49-44' #allconv log
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-14/14-22-23' #log #musicnn
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-12/17-13-56' #log
        #run_path ='/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-12/14-58-45'#allconvcomplex2
        #run_path ='/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-12/14-11-21'#musicnn mse loss lr 0.001
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-12/11-14-34' #allconvcomplex lr 0.001
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-12/11-06-43' #allconvcomplex lr 0.01
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-12/11-00-26' #allconv01 lr 0.01
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-12/10-54-40' #allconv complex
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-12/10-04-12' #musicnn smooth l1 loss
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-07-12/09-52-39' #musicnn mse loss HERE TL
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-06-17/10-11-44' #short mse loss
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-06-21/13-30-01' #dilconv
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-06-21/13-16-05' #allconv01
        #run_path = '/home/ykinoshita/humrec_mood_tagger/outputs/2023-06-21/13-24-33' #allconv02
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
    scale = cfg.datasets.scale
    test_loader, train_loader, valid_loader = load_data(cfg.training.batch_size,
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
                                                        mode = "test")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #TODO metrics
    test_model(model, NUM_CLASSES, test_loader, device, plot=True, model_name=cfg.model.architecture, transform = cfg.training.transformation, scale = scale)


if __name__ == "__main__":
    run()
