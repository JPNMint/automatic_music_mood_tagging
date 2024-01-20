import logging
import time
from collections import OrderedDict
from typing import Any
import os
import shutil
import pandas as pd
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
import torch.nn.functional as F
from catalyst import dl, utils
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim

from catalyst.engines.torch import GPUEngine


from denseweight import DenseWeight

from custom_loss_func import dense_weight_loss_single, dense_weight_loss, dense_weight_loss_tuned, RMSELoss, dense_weight_loss_tuned_RMSE ,dense_weight_loss_RMSE,RMSLELoss
from __init__  import get_architecture, test_model, test_model_fold
from data import load_data, FeatureSetup #, NUM_CLASSES
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


def run_training_fold(cfg: DictConfig, GEMS_9 = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness'], fold = False, stratified = False, skip = False, manual_seed = False, seed = 10) -> None:
    torch.cuda.empty_cache()
    time_t0 = time.time()


    if isinstance(fold, int) and fold is not False:
        print(f"Cross validating output! {fold} fold")
    else:
        print('No cross validation!')

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
    os_ratio = cfg.datasets.oversampling_ratio
    loss_func = cfg.datasets.loss_func
    oversampling_method = cfg.datasets.oversampling_method
    data_augmentation = cfg.datasets.data_augmentation

    if oversampling == True:

        if oversampling_method not in ['average_density_oversampling', 'average_density_oversampling_V2','density_oversampling_augmentation_V2','density_oversampling_augmentation', 'non_average_density_oversampling_augmentation', 'non_average_density_oversampling' ]:
            print('Invalid oversampling method!')
            return
    if oversampling != False:

        cur_method = cfg.datasets.oversampling_method+f'_{cfg.datasets.oversampling_tolerance}'
    else:
        cur_method = "No_oversampling"
    if data_augmentation > 0:
        cur_method = cur_method+f'_{data_augmentation}'
    for i in range(fold):
        print(f"Fold {i+1}!")
        if manual_seed == True:
            cur_seed = seed
        hydra_base_dir = HydraConfig.get().runtime.cwd #.
        if stratified == False and manual_seed != True:
            hydra_base_fold_dir = os.path.join(hydra_base_dir,f'output_fold/{cfg.model.architecture}/{cur_method}/{i+1}')
        elif stratified == True and manual_seed != True:
            hydra_base_fold_dir = os.path.join(hydra_base_dir,f'output_fold_stratified/{cfg.model.architecture}/{cur_method}/{i+1}')
        elif stratified == False and manual_seed == True:
            print("Creating output_fold_seeds folder")
            hydra_base_fold_dir = os.path.join(hydra_base_dir,f'output_fold_seeds/{cfg.model.architecture}/{cur_method}/{i+1}')
        elif stratified == True and manual_seed == True:
            hydra_base_fold_dir = os.path.join(hydra_base_dir,f'output_fold_stratified_seeds/{cfg.model.architecture}/{cur_method}/{i+1}')
        print(f"using output folder {hydra_base_fold_dir}")
        catalyst_out_dir = os.path.join(hydra_base_fold_dir, 'catalyst')
        model_out_dir = os.path.join(hydra_base_fold_dir, 'models')
        if not isinstance(GEMS_9, str) and cfg.model.architecture != 'vgg_freeze':
            if os.path.exists(catalyst_out_dir):
                shutil.rmtree(catalyst_out_dir)
            os.makedirs(catalyst_out_dir)
            if os.path.exists(model_out_dir):
                shutil.rmtree(model_out_dir)
            os.makedirs(model_out_dir)

            shutil.copy(architecture_file, os.path.join(model_out_dir, os.path.split(architecture_file)[1]))
        # if isinstance(GEMS_9, str):
        #     num_classes = 1
        # else:
        #     num_classes = NUM_CLASSES
        num_classes = len(GEMS_9)
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


    

        model = architecture(audio_input=use_audio_in, num_classes=num_classes, data_augmentation = data_augmentation, mode = 'train',**cfg.features)
        model.cuda()
        if use_audio_in:
            in_size = (batch_size, int((seq_len - 1) * (feat_sample_rate / feat_frame_rate) + feat_window_size))
        else:
            in_size = (batch_size, seq_len, cfg.features.freq_bins)
        output = model.forward(torch.Tensor(np.zeros(in_size)))
        print(output.shape)

        #model = architecture(audio_input=use_audio_in, num_classes=num_classes, data_augmentation = data_augmentation, mode = 'train',**cfg.features)
        #model = architecture(audio_input=use_audio_in, num_classes=num_classes, data_augmentation = data_augmentation, mode = 'train',**cfg.features)

        ##########Finetuning pretrained model
        if cfg.model.architecture == 'musicnn_arch_finetune':
            print(f"Loading Model: 'musicnn_arch_finetune'")
            run_path_pretrained = '/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/architectures/SotA' #musicnn mse loss
            # hydra_run_path = os.path.join(run_path_pretrained, '.hydra')
            # config_path = os.path.join(hydra_run_path, 'config.yaml')
            # cfg = OmegaConf.load(config_path)


            pretrained_model  = architecture(audio_input=cfg.model.audio_inputs, num_classes=9, debug=False, **cfg.features)
            print(f'Pretrained model: {cfg.model.architecture}')
            pretrained_dict = torch.load('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/architectures/SotA/musicnn_sota.pth')
            del pretrained_dict['dense2.weight']
            del pretrained_dict['dense2.bias']

            
            for name, param in pretrained_model.named_parameters():
                print(name)
                if name not in  ['dense1.weight', 'bn.weight', 'dense1.bias', 'bn.bias']:
                    param.requires_grad = False

            model_dict = pretrained_model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            pretrained_model.load_state_dict(pretrained_dict)
            pretrained_model.postprocess =  torch.nn.Sequential(
                                torch.nn.Linear(200,100),
                                torch.nn.Dropout(p = 0.5),
                                torch.nn.Linear(100,50),
                                torch.nn.Dropout(p = 0.5),
                                torch.nn.Linear(50,25),
                                torch.nn.Linear(25,num_classes)                       
                                    )
            model = pretrained_model
            

            
            



        feat_settings = FeatureSetup(
            "mood_feat", feat_sample_rate, 1, feat_frame_rate, feat_window_size, cfg.features.freq_bins, 30, False, "log", "log_1"
        )

        ######loading data
        if cfg.model.architecture == 'FCN_6_layer_classifier':
            task = 'classification'
        else:
            task = 'regression'
        test_loader, train_loader, valid_loader, train_annot, targets_all = load_data(batch_size,
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
                                                            oversampling_method = oversampling_method,
                                                            tolerance = tolerance,
                                                            os_ratio = os_ratio,
                                                            train_y = GEMS_9,
                                                            data_augmentation = data_augmentation,
                                                            alpha = cfg.datasets.dense_weight_alpha,
                                                            task = task,
                                                            fold = i+1,
                                                            stratified = stratified,
                                                            manual_seed = manual_seed,
                                                            cur_seed = seed) 
        


        


        if loss_func == 'dense_weight':
            print(f'Custom Loss function {loss_func}!')
            #TODO SEE HERE IF CORRECT
            if oversampling == True:
                criterion = dense_weight_loss(cfg.datasets.dense_weight_alpha, targets_all = train_annot)
            else:
                criterion = dense_weight_loss(cfg.datasets.dense_weight_alpha, targets_all = targets_all)

        elif loss_func == 'dense_weight_tuned':
            print(f'Custom Loss function {loss_func}!')
            criterion = dense_weight_loss_tuned(alpha = None, targets_all = train_annot)

        elif loss_func == 'dense_weight_single':
            print(f'Custom Loss function {loss_func}!')
            criterion = dense_weight_loss_single(cfg.datasets.dense_weight_alpha, targets_all = targets_all)

        elif loss_func == 'MAE':
            
            criterion = nn.L1Loss() #nn.MSELoss() #torch.nn.SmoothL1Loss()   nn.CrossEntropyLoss()
            print("MAE Loss Function!")
        elif loss_func == 'MSE':
            
            criterion = nn.MSELoss() #nn.MSELoss() #torch.nn.SmoothL1Loss()   nn.CrossEntropyLoss()
            print("MSE Loss Function!")

        elif loss_func == 'RMSE':
            
            criterion = RMSELoss() #nn.MSELoss() #torch.nn.SmoothL1Loss()   nn.CrossEntropyLoss()
            print("RMSE Loss Function!")
        
        elif loss_func == 'dense_weight_RMSE':
            print(f'Custom Loss function {loss_func}!')
            criterion = dense_weight_loss_RMSE(alpha = None, targets_all = train_annot)

        elif loss_func == 'dense_weight_tuned_RMSE':
            print(f'Custom Loss function {loss_func}!')
            criterion = dense_weight_loss_tuned_RMSE(alpha = None, targets_all = train_annot)
        elif loss_func == 'RMSLE':
            print(f'Custom Loss function {loss_func}!')
            criterion = RMSLELoss()
        elif loss_func == 'Cross_entropy':
            print(f'Custom Loss function {loss_func}!')
            criterion = nn.BCELoss() #torch.nn.CrossEntropyLoss
        
        else:
            
            criterion = nn.L1Loss() #nn.MSELoss() #torch.nn.SmoothL1Loss()   nn.CrossEntropyLoss()
            print("No Loss Function set, use default!")
            loss_func = 'MAE'

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay  = 1e-7)# weight_decay  = 1e-9
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
        if loss_func != 'dense_weight':
            alpha = 0
        else:
            alpha = cfg.datasets.dense_weight_alpha
        if skip == False:
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
                        'data_augmentation': data_augmentation,
                        'fold': i+1


                    }

            test_model_fold(model, num_classes, test_loader, engine.device, transform = cfg.training.transformation, training = 'training', scale = scale, model_name = cfg.model.architecture, csv_information = csv_information , task = task, fold = i+1,  csv_method = cur_method, stratified = stratified, manual_seed = manual_seed, cur_seed = cur_seed)

            if i+1 == 10: #TODO here 10
                csv_file = pd.read_csv(f'/home/ykinoshita/humrec_mood_tagger/output_fold/{cfg.model.architecture}/{cur_method}/fold_metrics.csv')
                print('CV metrics!')
                rmse_cv = csv_file['RMSE']
                r2_cv = csv_file['R2']
                print(f'R2: {r2_cv.mean()}')
                print(f'RMSE : {rmse_cv.mean()}')
                



def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    # cite https://github.com/jiawei-ren/BalancedMSE/blob/main/tutorial/balanced_mse.ipynb

    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var) # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss
class BMCLoss(nn.Module):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
        print(self.noise_sigma)

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        target = target.cpu().detach()
        pred = pred.cpu().detach()
        return bmc_loss(pred, target, noise_var)



def bmc_loss_md(pred, target, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    I = torch.eye(pred.shape[-1])
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
    
    return loss


@hydra.main(version_base=None, config_path="configs", config_name="default")
def my_app_fold(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    run_training_fold(cfg, GEMS_9= cfg.datasets.labels, fold = 10, stratified = True, skip = False, manual_seed = True, seed = 50)


if __name__ == "__main__":
    my_app_fold()
