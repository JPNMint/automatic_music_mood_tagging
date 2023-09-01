import torch

from architectures import LogMelSpec


class Net(torch.nn.Module):
    MONO = True
    SAMPLE_RATE = 44100
    DROPOUT_P = 0.1

    CHANNEL_BASE = 20

    DEFAULT_CLASSES = 1

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
        
        # TODO: log freq scaling?
        # TODO: Batch norm before ELU??

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channel_base, kernel_size=(5, 5), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.Conv2d(in_channels=channel_base, out_channels=channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Dropout2d(self.dropout_p),
            torch.nn.Conv2d(in_channels=channel_base, out_channels=2*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(2 * channel_base),
            torch.nn.Conv2d(in_channels=2*channel_base, out_channels=2*channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(2 * channel_base),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Dropout2d(self.dropout_p),
            torch.nn.Conv2d(
                in_channels=2 * channel_base, out_channels=4 * channel_base, kernel_size=(3, 3), padding="same"
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(4 * channel_base),
            torch.nn.Conv2d(
                in_channels=4 * channel_base, out_channels=4 * channel_base, kernel_size=(3, 3), padding="same"
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(4 * channel_base),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Dropout2d(self.dropout_p),
            torch.nn.Conv2d(
                in_channels=4 * channel_base, out_channels=8 * channel_base, kernel_size=(3, 3), padding="same"
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(8 * channel_base),
            torch.nn.Dropout2d(self.dropout_p),
            torch.nn.Conv2d(
                in_channels=8 * channel_base, out_channels=8 * channel_base, kernel_size=(3, 3), padding="same"
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(8 * channel_base),
            torch.nn.Dropout2d(self.dropout_p),
            torch.nn.Conv2d(in_channels=8 * channel_base, out_channels=self.num_classes, kernel_size=(1, 1), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(self.num_classes),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.ReLU(),
        )

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        print(self.num_classes)
        if self.debug:
            print(x_input.shape)
        if self.audio_input:
            spec = self.mel_spec_transform(x_input)
        else:
            spec = x_input
        if self.debug:
            print(spec.shape)
        x = spec[:, None, :, :]
        for cur_layer in self.cnn:
            x = cur_layer(x)
            if self.debug:
                print(x.shape)
        scores = x
        return scores.view(-1, self.num_classes)

    def is_sequential(self):
        return self._sequential
