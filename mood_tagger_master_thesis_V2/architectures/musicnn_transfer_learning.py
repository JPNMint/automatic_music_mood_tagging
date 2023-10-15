import torch

from architectures import LogMelSpec


from musicnn.tagger import extractor
import warnings

warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)


class Net(torch.nn.Module):
    MONO = True
    SAMPLE_RATE = 16000
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
        
        # TODO: log freq scaling?
        # TODO: Batch norm before ELU??

        dense_channel = 200
        self.dense1 = torch.nn.Linear((561+(midend_channel*3))*2, dense_channel)
        self.bn = torch.nn.BatchNorm1d(dense_channel)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.dense2 = torch.nn.Linear(dense_channel, num_classes)


    def forward(self, x_input: torch.Tensor, embed_input: torch.Tensor) -> torch.Tensor:

        

        #taggram, tags, features = extractor(file_name, model='MTT_musicnn', extract_features=True)





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
