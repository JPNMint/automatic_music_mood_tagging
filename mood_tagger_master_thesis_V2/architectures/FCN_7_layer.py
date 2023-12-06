import torch

from architectures import LogMelSpec, data_aug
import librosa
import torchaudio
from torchaudio.transforms import  TimeStretch


#cite https://github.com/minzwon/sota-music-tagging-models/blob/master/training/model.py#L9

#librosa 0.8.0 and numpy==1.24.4 worked before 
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
        data_augmentation: list = [],
        mode: str = 'train',
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
        self.mode = mode
        self.data_augmentation = data_augmentation

        self.mel_spec_transform = LogMelSpec(
            sample_rate=self.sample_rate, n_fft=self.N_FFT, frequency_bins=self.frequency_bins, hop_size=self.hop_length
        )
        
        # TODO: log freq scaling?
        # TODO: Batch norm before ELU??

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(in_channels=channel_base, out_channels=channel_base*2, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base*2),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Dropout2d(self.dropout_p),

            torch.nn.Conv2d(in_channels=channel_base*2, out_channels=2*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(2 * channel_base),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Dropout2d(self.dropout_p),

            torch.nn.Conv2d(in_channels=channel_base*2, out_channels=2*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(2 * channel_base),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Dropout2d(self.dropout_p),


            torch.nn.Conv2d(in_channels=channel_base*2, out_channels=1*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1 * channel_base),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.Dropout2d(self.dropout_p),

            
            torch.nn.Conv2d(in_channels=1*channel_base, out_channels=1*channel_base, kernel_size=(1, 1), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(1 * channel_base),
            # torch.nn.MaxPool2d((2, 2)),
            torch.nn.Dropout2d(self.dropout_p),


            torch.nn.Conv2d(in_channels=1*channel_base, out_channels=self.num_classes, kernel_size=(1, 1), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(self.num_classes),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.ReLU(),


        )

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        #CHECK HOW TO CONCAT
        # if self.mode == 'train' and len(self.data_augmentation) != 0:
        #     #should be random draw
        #     #depending on chain length 
        #     if 'time_stretch' in self.data_augmentation:
        #         x_input_aug = librosa.effects.time_stretch(x_input.cpu().detach().numpy(), rate = 1.0008) #shifted 25 ms bei 30 sekunden

        #     if 'pitch_shift' in self.data_augmentation:
                
        #         x_input_aug = librosa.effects.pitch_shift(x_input, self.sample_rate, n_steps=1)
        #x_input = torch.from_numpy(x_input_aug) #.to(torch.device("cuda:0"))
        #new_x = torch.cat((x_input, x_input_aug), dim=0)

        if self.debug:
            print(x_input.shape)
        if self.audio_input:
            spec = self.mel_spec_transform(x_input)
        else:
            spec = x_input
        if self.debug:
            print(spec.shape)

        #make own function for this


        x = spec[:, None, :, :]

        # testing = data_aug(x, data_augmentation = self.data_augmentation)
        # x = torch.cat((x[spec.shape[0]:, :], testing), dim=0)

        for cur_layer in self.cnn:
            x = cur_layer(x)
            if self.debug:
                print(x.shape)

        scores = x
        return scores.view(-1, self.num_classes)

    def is_sequential(self):
        return self._sequential
