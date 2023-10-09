from omegaconf import OmegaConf
import os
from __init__ import test_model, get_architecture

import torch

run_path = '/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/architectures/SotA' #musicnn mse loss
hydra_run_path = os.path.join(run_path, '.hydra')
config_path = os.path.join(hydra_run_path, 'config.yaml')
# overrides_path = os.path.join(hydra_run_path, 'overrides.yaml')
cfg = OmegaConf.load(config_path)



#https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/16

# Load the pre-trained musicnn model
#musicnn_model = torch.load('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/architectures/SotA/musicnn_sota.pth')

# Assuming the output feature is the last layer of musicnn
# Extract the feature extractor part of the model (without the final prediction layers)


architecture, architecture_file = get_architecture(cfg.model.architecture, None)

model = architecture(audio_input=cfg.model.audio_inputs, num_classes=9, debug=False, **cfg.features)


pretrained_dict = torch.load('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/architectures/SotA/musicnn_sota.pth')
del pretrained_dict['dense2.weight']
del pretrained_dict['dense2.bias']
model_dict = model.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
model.load_state_dict(pretrained_dict)


# Set the model to evaluation mode
model.eval()

with torch.no_grad():
    musicnn_features = model(input_data)