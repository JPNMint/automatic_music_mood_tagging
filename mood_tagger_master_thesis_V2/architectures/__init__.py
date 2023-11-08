import torch
from torchaudio.transforms import MelSpectrogram, TimeStretch

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
        x_input = x_input.cuda()
        calc_mel_spec = self.mel_spec_layer(x_input)
        #print(x_input.device)
        # stretch = TimeStretch(n_freq=200).cuda()
        # calc_mel_spec = stretch(calc_mel_spec, 1.0008) 
        # calc_mel_spec = calc_mel_spec.cuda()
        y_output = torch.log10(calc_mel_spec * self.spec_log_mul + self.spec_log_add)

        return y_output.type(torch.cuda.FloatTensor)
    
from torch.nn.functional import interpolate
def data_aug(spec , data_augmentation):
        spec = spec[:spec.shape[0], :]
        if 'time_stretch' in data_augmentation:
            stretch = TimeStretch(n_freq=200).cuda()
            calc_mel_spec = stretch(spec, 1.20) #0008
            calc_mel_spec = calc_mel_spec.cuda()
        

        return calc_mel_spec.type(torch.cuda.FloatTensor)
