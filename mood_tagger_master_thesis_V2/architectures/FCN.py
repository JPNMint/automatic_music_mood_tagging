import torch

from architectures import LogMelSpec
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
##50 batch size is too big use 15
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

        # self.mel_spec_transform = LogMelSpec(
        #     sample_rate=self.sample_rate, n_fft=self.N_FFT, frequency_bins=self.frequency_bins, hop_size=self.hop_length
        # )
        self.spec = MelSpectrogram(sample_rate=16000,
                                                         n_fft=512,
                                                         f_min=0.0,
                                                         f_max=8000.0,
                                                         n_mels=96)
        self.to_db = AmplitudeToDB()
        # TODO: log freq scaling?
        # TODO: Batch norm before ELU??
        self.spec_bn = torch.nn.BatchNorm2d(1)
        pad = 'same'

        channel_base = 128
        self.cnn = torch.nn.Sequential(
        torch.nn.Conv2d(1, channel_base, kernel_size = 3, stride=1, padding= pad),
        torch.nn.BatchNorm2d(channel_base),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d((2,4)),

        torch.nn.Conv2d(channel_base, 3*channel_base, kernel_size = 3, stride=1, padding=pad),
        torch.nn.BatchNorm2d(3*channel_base),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d((4,5)),

        torch.nn.Conv2d(3*channel_base, 6*channel_base, kernel_size = 3, stride=1, padding=pad),
        torch.nn.BatchNorm2d(6*channel_base),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d((3,8)),

        torch.nn.Conv2d(6*channel_base, 16*channel_base, kernel_size = 3, stride=1, padding=pad),
        torch.nn.BatchNorm2d(16*channel_base),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d((2,8)),

        torch.nn.Conv2d(16*channel_base, 8*channel_base, kernel_size = 3, stride=1, padding=pad),
        torch.nn.BatchNorm2d(8*channel_base),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d((2,4)),
        # torch.nn.Conv2d(2*channel_base, num_classes, kernel_size = 3, stride=1, padding=pad),
        # torch.nn.BatchNorm2d(channel_base),
        # torch.nn.ReLU(),
        # torch.nn.MaxPool2d((4,8)),
        # torch.nn.Flatten()
        )


        self.dense3 = torch.nn.Linear(16*channel_base, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x_input: torch.Tensor) -> torch.Tensor:

        # if self.debug:
        #     print(x_input.shape)
        # if self.audio_input:
        #     spec = self.mel_spec_transform(x_input)
        # else:
        #     spec = x_input
        # if self.debug:
        #     print(spec.shape)
        # x = spec[:, None, :, :]

        x = self.spec(x_input)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)
        print(x.shape)

        for cur_layer in self.cnn:
            x = cur_layer(x)
            if self.debug:
                print(x.shape)
        # x = x.view(x.size(0), -1)

        # x = self.dropout(x)
        # x = self.dense3(x)


        scores = x
        return scores.view(-1, self.num_classes)

    def is_sequential(self):
        return self._sequential
