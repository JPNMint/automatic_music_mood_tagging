import torch
from torchaudio.transforms import MelSpectrogram


class LogMelSpec(torch.nn.Module):
    SPEC_LOG_MUL = 1
    SPEC_LOG_ADD = 1

    def __init__(self, sample_rate: int, n_fft: int, hop_size: int, frequency_bins: int,
                 spec_log_mul: float = SPEC_LOG_MUL, spec_log_add: float = SPEC_LOG_ADD):
        super(LogMelSpec, self, ).__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.frequency_bins = frequency_bins
        self.hop_size = hop_size
        self.spec_log_mul = spec_log_mul
        self.spec_log_add = spec_log_add

        self.mel_spec_layer = MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft,
                                             n_mels=self.frequency_bins, hop_length=self.hop_size,
                                             power=1)

    def forward(self, x_input):
        calc_mel_spec = self.mel_spec_layer(x_input)
        y_output = torch.log10(calc_mel_spec * self.spec_log_mul + self.spec_log_add)
        return y_output