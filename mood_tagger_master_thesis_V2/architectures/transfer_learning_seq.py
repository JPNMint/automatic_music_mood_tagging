import torch
from omegaconf import OmegaConf
import os
from __init__ import test_model, get_architecture
from architectures import LogMelSpec
import torchaudio



class Net(torch.nn.Module):
    MONO = True
    SAMPLE_RATE = 44100
    DROPOUT_P = 0.1

    CHANNEL_BASE = 20

    DEFAULT_CLASSES = 9

    N_FFT = 8192
    FPS = 5

    def __init__(
        self,
        audio_input: bool,
        sample_rate: int = SAMPLE_RATE,
        dropout_p: float = DROPOUT_P,
        channel_base: int = CHANNEL_BASE,
        num_classes: int = DEFAULT_CLASSES,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self._sequential = False
        self.debug = debug

        self.audio_input = audio_input
        self.sample_rate = sample_rate
        self.channel_base = channel_base
        self.num_classes = num_classes

        self.dropout_p = dropout_p
        self.frequency_bins = 200
        self.hop_length = int(self.sample_rate / self.FPS)

        self.mel_spec_transform = LogMelSpec(
            sample_rate=self.sample_rate, n_fft=self.N_FFT, frequency_bins=self.frequency_bins, hop_size=self.hop_length
        )


        run_path_pretrained = '/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/architectures/SotA' #musicnn mse loss
        hydra_run_path = os.path.join(run_path_pretrained, '.hydra')
        config_path = os.path.join(hydra_run_path, 'config.yaml')
        cfg = OmegaConf.load(config_path)

        architecture, architecture_file = get_architecture('musicnn_transfer', None)

        self.pretrained_model  = architecture(audio_input=cfg.model.audio_inputs, num_classes=9, debug=False, **cfg.features)
        print(f'Pretrained model: musicnn_transfer')
        pretrained_dict = torch.load('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/architectures/SotA/musicnn_sota.pth')
        del pretrained_dict['dense2.weight']
        del pretrained_dict['dense2.bias']

        model_dict = self.pretrained_model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.pretrained_model.load_state_dict(pretrained_dict)
        # Set the model to evaluation mode
        self.pretrained_model.eval()
        print('Freeze pretrained model!')
        for param in self.pretrained_model .parameters():
            param.requires_grad = False
            



        # Dense
        dense_channel = 200
        self.dense1 = torch.nn.Linear(200, dense_channel)
        self.bn = torch.nn.BatchNorm1d(dense_channel)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.dense2 = torch.nn.Linear(dense_channel, num_classes)

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:

        if self.debug:
            print(x_input.shape)
        if self.audio_input:
            spec = self.mel_spec_transform(x_input)
        else:
            spec = x_input
        if self.debug:
            print(spec.shape)
        
        #x = spec[:, None, :, :]
        pre_trained_output = self.pretrained_model(x_input)

        

        ### 
        # research transfer learning concat
        #out = torch.cat([scores, pre_trained_output], 1)
       
       #dense layers
        out = self.relu(self.bn(self.dense1(pre_trained_output)))
        out = self.dropout(out)
        out = self.dense2(out)

        return out.view(-1, self.num_classes)

    def is_sequential(self):
        return self._sequential









##https://stackoverflow.com/questions/66096478/how-train-the-pytorch-model-with-another-freeze-model
##https://stackoverflow.com/questions/71364119/how-to-combine-two-trained-models-using-pytorch

