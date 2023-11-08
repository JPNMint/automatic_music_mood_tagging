import torch

from architectures import LogMelSpec


class Net(torch.nn.Module):
    MONO = True
    SAMPLE_RATE = 44100
    DROPOUT_P = 0.1

    CHANNEL_BASE = 64

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
        self.cnn_dict = {}


        self.cnn0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(in_channels=channel_base, out_channels=channel_base*2, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            torch.nn.Conv2d(in_channels=channel_base*2, out_channels=1*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1 * channel_base),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            
            torch.nn.Conv2d(in_channels=1*channel_base, out_channels=1, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.ReLU(),


        )
        
        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(in_channels=channel_base, out_channels=channel_base*2, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            torch.nn.Conv2d(in_channels=channel_base*2, out_channels=1*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1 * channel_base),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            
            torch.nn.Conv2d(in_channels=1*channel_base, out_channels=1, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.ReLU(),


        )


        self.cnn2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(in_channels=channel_base, out_channels=channel_base*2, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            torch.nn.Conv2d(in_channels=channel_base*2, out_channels=1*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1 * channel_base),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            
            torch.nn.Conv2d(in_channels=1*channel_base, out_channels=1, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.ReLU(),


        )


        self.cnn3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(in_channels=channel_base, out_channels=channel_base*2, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            torch.nn.Conv2d(in_channels=channel_base*2, out_channels=1*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1 * channel_base),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            
            torch.nn.Conv2d(in_channels=1*channel_base, out_channels=1, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.ReLU(),


        )

        self.cnn4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(in_channels=channel_base, out_channels=channel_base*2, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            torch.nn.Conv2d(in_channels=channel_base*2, out_channels=1*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1 * channel_base),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            
            torch.nn.Conv2d(in_channels=1*channel_base, out_channels=1, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.ReLU(),


        )



        self.cnn5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(in_channels=channel_base, out_channels=channel_base*2, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            torch.nn.Conv2d(in_channels=channel_base*2, out_channels=1*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1 * channel_base),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            
            torch.nn.Conv2d(in_channels=1*channel_base, out_channels=1, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.ReLU(),


        )
        self.cnn6 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(in_channels=channel_base, out_channels=channel_base*2, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            torch.nn.Conv2d(in_channels=channel_base*2, out_channels=1*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1 * channel_base),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            
            torch.nn.Conv2d(in_channels=1*channel_base, out_channels=1, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.ReLU(),


        )
        self.cnn7 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(in_channels=channel_base, out_channels=channel_base*2, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            torch.nn.Conv2d(in_channels=channel_base*2, out_channels=1*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1 * channel_base),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            
            torch.nn.Conv2d(in_channels=1*channel_base, out_channels=1, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.ReLU(),


        )
        self.cnn8 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(in_channels=channel_base, out_channels=channel_base*2, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            torch.nn.Conv2d(in_channels=channel_base*2, out_channels=1*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1 * channel_base),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            
            torch.nn.Conv2d(in_channels=1*channel_base, out_channels=1, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.ReLU(),


        )


    def forward(self, x_input: torch.Tensor) -> torch.Tensor:

        #look into x_input to understand this
        #how to oversample for each 

        #try allconv_01_class first, then try to implement into multitask classifier

        if self.debug:
            print(x_input.shape)
        if self.audio_input:
            spec = self.mel_spec_transform(x_input)
        else:
            spec = x_input
        if self.debug:
            print(spec.shape)
        x_original = spec[:, None, :, :]
        x_dict = {}
        counter = 0
        cnns = [self.cnn0,self.cnn1,self.cnn2,self.cnn3,self.cnn4,self.cnn5,self.cnn6,self.cnn7,self.cnn8]
        x = x_original #keep original
        for network in cnns:
            x = x_original #keep original
            for cur_layer in network:
                x = cur_layer(x)
            x_dict[counter] = x.view(-1, 1)
            counter += 1
        
        # for cur_layer in self.cnn1:
        #     x = cur_layer(x)

        # x = x_original #keep original
        # for cur_layer in self.cnn2:
        #     x = cur_layer(x)
        # x = x_original #keep original
        # for cur_layer in self.cnn3:
        #     x = cur_layer(x)
        # x = x_original #keep original
        # for cur_layer in self.cnn4:
        #     x = cur_layer(x)
        # x = x_original #keep original
        # for cur_layer in self.cnn5:
        #     x = cur_layer(x)
        # x = x_original #keep original
        # for cur_layer in self.cnn6:
        #     x = cur_layer(x)
        # x = x_original #keep original
        # for cur_layer in self.cnn7:
        #     x = cur_layer(x)
        # x = x_original #keep original
        # for cur_layer in self.cnn8:
        #     x = cur_layer(x)


 #safe output
        # for key, value in self.cnn_dict.items():
        # output = x_dict
        x_keys = list(x_dict.keys())
        output = x_dict[0] #cnn0
        for i in range(self.num_classes-1):##get rest
            output = torch.cat([output, x_dict[i]], dim = 1)
        
 
        #need to concat here somewhere torch.cat([mp, avgp], dim=1)
        #scores = x
        return output

    def is_sequential(self):
        return self._sequential
