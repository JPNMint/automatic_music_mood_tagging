import torch

from mood_tagger.architectures import LogMelSpec


class Net(torch.nn.Module):
    DEFAULT_CLASSES = 9

    MONO = True
    SAMPLE_RATE = 44100
    N_FFT = 8192
    FPS = 5
    N_FREQ_BINS = 200

    DROPOUT_P = 0.1
    CHANNEL_BASE = 20

    def __init__(
        self,
        audio_input: bool,
        sample_rate: int = SAMPLE_RATE,
        window_size: int = N_FFT,
        dropout_p: float = DROPOUT_P,
        channel_base: int = CHANNEL_BASE,
        num_classes: int = DEFAULT_CLASSES,
        frame_rate: int = FPS,
        freq_bins: int = N_FREQ_BINS,
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
        self.n_fft = window_size
        self.fps = frame_rate

        self.dropout_p = dropout_p
        self.frequency_bins = freq_bins
        self.hop_length = int(self.sample_rate / self.fps)

        self.mel_spec_transform = LogMelSpec(
            sample_rate=self.sample_rate, n_fft=self.n_fft, frequency_bins=self.frequency_bins, hop_size=self.hop_length
        )

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channel_base, kernel_size=(5, 5), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.Conv2d(in_channels=channel_base, out_channels=channel_base, kernel_size=(3, 3), padding="same"),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(channel_base),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Dropout2d(self.dropout_p),
            torch.nn.Conv2d(
                in_channels=channel_base, out_channels=2 * channel_base, kernel_size=(3, 3), padding="same"
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(2 * channel_base),
            torch.nn.Conv2d(
                in_channels=2 * channel_base, out_channels=2 * channel_base, kernel_size=(3, 3), padding="same"
            ),
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
