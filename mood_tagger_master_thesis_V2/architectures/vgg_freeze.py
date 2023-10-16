#https://blog.paperspace.com/vgg-from-scratch-pytorch/

import torch

from architectures import LogMelSpec


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
        
        
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.preprocess = torch.nn.Sequential(
                            torch.nn.Linear(128,64),
                            torch.nn.Dropout(p = 0.5),
                            torch.nn.Linear(64,32),
                            torch.nn.Linear(32,num_classes)                       
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
        for cur_layer in self.model.preprocess:
            x = cur_layer(x)
            if self.debug:
                print(x.shape)
        scores = x
        return scores.view(-1, self.num_classes)


    def is_sequential(self):
        return self._sequential
