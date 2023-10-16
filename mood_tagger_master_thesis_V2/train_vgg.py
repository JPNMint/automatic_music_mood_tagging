
import numpy as np
import torch
from torch import nn, optim
from torchaudio.transforms import MelSpectrogram
import torchaudio
from data import load_data, FeatureSetup #, NUM_CLASSES
device = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")
GEMS_9 =['Wonder'] #, 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
model = torch.hub.load('harritaylor/torchvggish', 'vggish')
for param in model.parameters():
    param.requires_grad = False

model. = torch.nn.Sequential(
                            torch.nn.Linear(128,64),
                            torch.nn.Dropout(p = 0.5),
                            torch.nn.Linear(64,32),
                            torch.nn.Linear(32,len(GEMS_9))                       
                                )

use_audio_in = True
gpu_num = 0
k_samples = 1e5
batch_size = 40
val_size =  0.5
num_pt_workers =8
max_epochs = 99999999

memory_map = True
patience = 20
refinements = 3

seq_len = 150
hop_size = 40
feat_sample_rate = 44100
feat_frame_rate = 5
feat_window_size = 8192

learning_rate =  0.1 
transform = None
scale = None

oversampling = False
tolerance = 10
loss_func = 'MAE'

model = model.to(device)
feat_settings = FeatureSetup(
        "mood_feat", feat_sample_rate, 1, feat_frame_rate, feat_window_size, 200, 30, False, "log", "log_1"
    )
    ######loading data
test_loader, train_loader, valid_loader, targets_all = load_data(batch_size,
                                                        feat_settings,
                                                        gpu_num,
                                                        hop_size,
                                                        k_samples,
                                                        num_pt_workers,
                                                        seq_len,
                                                        use_audio_in,
                                                        val_size,
                                                        True,
                                                        transform = transform,
                                                        scale = scale,
                                                        mode = "train",
                                                        oversampling = oversampling,
                                                        tolerance = tolerance,
                                                        train_y = GEMS_9) 

if loss_func == 'dense_weight':
    print(f'Custom Loss function {loss_func}!')
    criterion = dense_weight_loss(1, targets_all = targets_all)

elif loss_func == 'MAE':
        
    criterion = torch.nn.L1Loss() #nn.MSELoss() #torch.nn.SmoothL1Loss()   nn.CrossEntropyLoss()
    print("MAE Loss Function!")
elif loss_func == 'MSE':
        
    criterion = torch.nn.MSELoss() #nn.MSELoss() #torch.nn.SmoothL1Loss()   nn.CrossEntropyLoss()
    print("MSE Loss Function!")
else:
        
    criterion = torch.nn.L1Loss() #nn.MSELoss() #torch.nn.SmoothL1Loss()   nn.CrossEntropyLoss()
    print("No Loss Function set, use default!")
    loss_func = 'MAE'

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

mel_spec_transform = LogMelSpec(
            sample_rate=44100, 
            n_fft=1024, 
            frequency_bins=96, 
            hop_size=512
        )
to_db = torchaudio.transforms.AmplitudeToDB() 
spec_bn = torch.nn.BatchNorm2d(1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(25):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        #x = inputs.to(device)
        spec =  mel_spec_transform(inputs)
        x = spec[:, None, :, :].to(device)
        
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(x)
        loss = criterion(outputs, labels)        
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(running_loss)
print('Finished Training')


class dense_weight_loss(nn.Module):
    # cite https://github.com/SteiMi/denseweight 
    # use of package denseweight to calculate weights
    def __init__(self ,alpha, targets_all):
        super(dense_weight_loss, self).__init__()
        #put it before as input? as in fit it not in the class?
        self.dw = DenseWeight(alpha=alpha)
        self.dw.fit(targets_all)
    def forward(self, predictions, targets):
        try:

            targs = targets.cpu().detach().numpy()
            weighted_targs = self.dw(targs)
            relevance = torch.from_numpy(weighted_targs).to(torch.device("cuda:0"))
        except ValueError:
            print(
                        'WARNING!)'
                    )
            relevance = torch.ones_like(targets)

        err = torch.pow(predictions - targets, 2)
        err_weighted = relevance * err
        mse = err_weighted.mean()

        return mse