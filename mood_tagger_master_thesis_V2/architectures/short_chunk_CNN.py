import torch.nn as nn
import torch
from architectures import LogMelSpec
import torchaudio

class Net(torch.nn.Module):
    MONO = True
    SAMPLE_RATE = 16000
    DROPOUT_P = 0.1

    CHANNEL_BASE = 128

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

        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=512,
                                                         f_min=0.0,
                                                         f_max=8000.0,
                                                         n_mels=128)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)
        # TODO: log freq scaling?
        # TODO: Batch norm before ELU??

        self.layer1 = Conv_2d(1, self.channel_base, pooling=2)
        self.layer2 = Conv_2d(self.channel_base, self.channel_base, pooling=2)
        self.layer3 = Conv_2d(self.channel_base, self.channel_base*2, pooling=2)
        self.layer4 = Conv_2d(self.channel_base*2, self.channel_base*2, pooling=2)
        self.layer5 = Conv_2d(self.channel_base*2, self.channel_base*2, pooling=2)
        self.layer6 = Conv_2d(self.channel_base*2, self.channel_base*2, pooling=2)
        self.layer7 = Conv_2d(self.channel_base*2, self.channel_base*4, pooling=2)
        # Dense


        self.dense = nn.Sequential(
            nn.Linear(self.channel_base*4, self.channel_base*4),
            nn.BatchNorm1d(self.channel_base*4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.channel_base*4, self.num_classes),
            


        )

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:

        x = self.spec(x_input.cuda())
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)



        for cur_layer in self.dense:
            x = cur_layer(x)
            if self.debug:
                print(x.shape)
        scores = x
        return scores.view(-1, self.num_classes)

    def is_sequential(self):
        return self._sequential

class Conv_2d(nn.Module):
    
    ##cite https://github.com/minzwon/sota-music-tagging-models/blob/master/training/modules.py
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out
    

#info

# seq hop 4, seq len 18, batch 26