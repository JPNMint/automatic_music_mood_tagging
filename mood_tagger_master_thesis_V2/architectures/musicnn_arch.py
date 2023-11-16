import torch

from architectures import LogMelSpec

import torchaudio

class Net(torch.nn.Module):
    MONO = True
    SAMPLE_RATE = 16000 #16000 #44100
    DROPOUT_P = 0.1

    CHANNEL_BASE = 20
    CHANNEL_BASE_V = 204
    
    DEFAULT_CLASSES = 9

    N_FFT = 1024 #512 
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
        self.frequency_bins = 96
        self.hop_length = int(self.N_FFT/2)#int(self.sample_rate / self.FPS)

        self.spec = torchaudio.transforms.MelSpectrogram(
                                                        sample_rate=sample_rate,
                                                         n_fft=512,
                                                         f_min=0.0,
                                                         f_max=8000.0,
                                                         n_mels=96
        )
    
        self.to_db = torchaudio.transforms.AmplitudeToDB() 
        self.spec_bn = torch.nn.BatchNorm2d(1)


        # Pons front-end    
        m1 = Conv_V(1, 204, (int(0.7*self.frequency_bins), 7))
        m2 = Conv_V(1, 204, (int(0.4*self.frequency_bins), 7))

        m3 = Conv_H(1, 51, 129)
        m4 = Conv_H(1, 51, 65)
        m5 = Conv_H(1, 51, 33)

        self.layers = torch.nn.ModuleList([m1, m2, m3, m4, m5])


        # Pons mid-end
        midend_channel = 64
        self.layer1 = Conv_1d(561, midend_channel, 7, 1, 1)
        self.layer2 = Conv_1d(midend_channel, midend_channel, 7, 1, 1)
        self.layer3 = Conv_1d(midend_channel, midend_channel, 7, 1, 1)

        # Dense
        dense_channel = 200
        self.dense1 = torch.nn.Linear((561+(midend_channel*3))*2, dense_channel)
        self.bn = torch.nn.BatchNorm1d(dense_channel)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.postprocess = torch.nn.Sequential(
                            torch.nn.Linear(200,100),
                            torch.nn.Dropout(p = 0.5),
                            torch.nn.Linear(100,50),
                            torch.nn.Dropout(p = 0.5),
                            torch.nn.Linear(50,25),
                            torch.nn.Linear(25,num_classes)                       
                                )


    def forward(self, x_input: torch.Tensor) -> torch.Tensor:

        if self.debug:
            print(x_input.shape)
        if self.audio_input:
            specto = self.spec(x_input.cuda())
        else:
            specto = x_input
        if self.debug:
            print(specto.shape)
        
        specto = self.to_db(specto)
        specto = specto.unsqueeze(1)
        specto = self.spec_bn(specto)




        # Pons front-end
        out = []
        for layer in self.layers:
            out.append(layer(specto))
            #print(layer(specto).shape)
        out = torch.cat(out, dim=1)



        # Pons mid-end
        length = out.size(2)
        res1 = self.layer1(out)
        #print(out.shape)
        res2 = self.layer2(res1) + res1
        #print(res2.shape)
        res3 = self.layer3(res2) + res2
        out = torch.cat([out, res1, res2, res3], 1)

        
        mp = torch.nn.MaxPool1d(length)(out)
        avgp = torch.nn.AvgPool1d(length)(out)

        out = torch.cat([mp, avgp], dim=1)
        out = out.squeeze(2)
        ## dense
        out = self.relu(self.bn(self.dense1(out)))
        out = self.dropout(out)
        print(out.shape)
        #out = torch.nn.Sigmoid()(out)
        x = out
        for cur_layer in self.postprocess:
            x = cur_layer(x)
        
        return x #scores.view(-1, self.num_classes)

    def is_sequential(self):
        return self._sequential




    

class Conv_V(torch.nn.Module):
    # vertical convolution
    def __init__(self, input_channels, output_channels, filter_shape):
        super(Conv_V, self).__init__()
        self.conv = torch.nn.Conv2d(input_channels, output_channels, filter_shape,
                              padding=(0, filter_shape[1]//2))
        self.bn = torch.nn.BatchNorm2d(output_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        freq = x.size(2)
        out = torch.nn.MaxPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        return out




class Conv_H(torch.nn.Module):
    # horizontal convolution
    def __init__(self, input_channels, output_channels, filter_length):
        super(Conv_H, self).__init__()
        self.conv = torch.nn.Conv1d(input_channels, output_channels, filter_length,
                              padding=filter_length//2)
        self.bn = torch.nn.BatchNorm1d(output_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        freq = x.size(2)
        out = torch.nn.AvgPool2d((freq, 1), stride=(freq, 1))(x)
        out = out.squeeze(2)
        out = self.relu(self.bn(self.conv(out)))
        return out
    

    

class Conv_1d(torch.nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_1d, self).__init__()
        self.conv = torch.nn.Conv1d(input_channels, output_channels, shape, stride=stride, padding=shape//2) #
        self.bn = torch.nn.BatchNorm1d(output_channels)
        self.relu = torch.nn.ReLU()
        self.mp = torch.nn.MaxPool1d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out
