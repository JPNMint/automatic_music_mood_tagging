import torch

from architectures import LogMelSpec

import torchaudio
class Net(torch.nn.Module):
    MONO = True
    DROPOUT_P = 0.1

    CHANNEL_BASE = 20# 20 

    DEFAULT_CLASSES = 9

    N_FFT = 512 #8192
    FPS = 5


    #cnn param
    
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
        print(self.sample_rate)
        #self.mel_spec_transform = LogMelSpec(
        #    sample_rate=self.sample_rate, n_fft=self.N_FFT, frequency_bins=self.frequency_bins, hop_size=self.hop_length
        #)
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                n_fft=self.N_FFT,
                                                                f_min=0.0,
                                                                f_max=8000.0,
                                                                n_mels=128)
        shape = 3 

        padding=3//2
        pooling = 2
        self.to_db = torchaudio.transforms.AmplitudeToDB() ##TRY FROM ORIGINAL MODEL
        self.spec_bn = torch.nn.BatchNorm2d(1)

        # TODO: log freq scaling?
        # TODO: Batch norm before ELU??

        self.cnn = torch.nn.Sequential(
            # 1st layer
            torch.nn.Conv2d(1, channel_base, shape, stride = 1, padding = padding),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pooling),
            ### need pytorch tutorial tbh
            
            # 2nd layer
            torch.nn.Conv2d(channel_base, channel_base, shape, stride = 1, padding = padding),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pooling),

            # 3rd layer
            torch.nn.Conv2d(channel_base, channel_base*2, shape, stride = 1, padding = padding),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pooling),

            # 4th layer
            torch.nn.Conv2d(channel_base*2, channel_base*2, shape, stride = 1, padding = padding),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pooling),


            # 5th  layer
            torch.nn.Conv2d(channel_base*2, channel_base*2, shape, stride = 1, padding = padding),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pooling),



            # 6th layer
            torch.nn.Conv2d(channel_base*2, channel_base*2, shape, stride = 1, padding = padding),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pooling),
            # 7th layer
            torch.nn.Conv2d(channel_base*2, channel_base*4, shape, stride = 1, padding = padding),
            torch.nn.BatchNorm2d(channel_base*4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pooling)
            ,
            #8th 

           # torch.nn.Conv2d(channel_base*4, channel_base*8, shape, stride = 1, padding = padding),
            #torch.nn.BatchNorm2d(channel_base*8),
            #torch.nn.ReLU(),
            #torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
            #,

            #reshape_tensor()
            #,
            # Dense

            #torch.nn.ReLU()

            #torch.nn.Conv2d(channel_base*4, self.num_classes, shape, stride = 1, padding = padding),
            #torch.nn.BatchNorm2d(self.num_classes),
            #torch.nn.ReLU()



        )

        
        # Dense
        self.dense1 = torch.nn.Linear(channel_base*4, channel_base*4)
        self.bn = torch.nn.BatchNorm1d(channel_base*4)
        self.dense2 = torch.nn.Linear(channel_base*4, self.num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()


    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        if self.debug:
            print(x_input.shape)
        if self.audio_input:
            spec = self.mel_spec_transform(x_input)

        else:
            spec = x_input
        spec = self.to_db(spec)
        spec = spec.unsqueeze(1)
        spec = self.spec_bn(spec)
        spec = spec.squeeze(1)
        if self.debug:
            print(spec.shape)
        x = spec[:, None, :, :]
        for cur_layer in self.cnn:
            x = cur_layer(x)
            if self.debug:
                print(f'Current layer {cur_layer}')
                print(x.shape)


        x = x.squeeze(2)
        if x.size(-1) != 1:
            x = torch.nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        scores = x #(32,9,1,1) sollte sein
        return scores.view(-1, self.num_classes)

    def is_sequential(self):
        return self._sequential

class reshape_tensor(torch.nn.Module):
    def forward(self, conv_output):
        conv_output = conv_output.view(conv_output.size(0), -1)
        return conv_output