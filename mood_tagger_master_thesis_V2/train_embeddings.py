import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pathlib as Path
import pandas as pd
from data import remove_special_chars
from sklearn.metrics import mean_squared_error, r2_score

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.file_list = os.listdir(data_folder)
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_folder, self.file_list[idx])
        data = np.load(file_path)
        
        if self.transform:
            data = self.transform(data)
        
        return data



class embeddingsNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(embeddingsNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        self.bnorm1 = torch.nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(hidden_size, 2*hidden_size)

        self.bnorm2 = torch.nn.BatchNorm1d(hidden_size*2)
        self.fc3 = nn.Linear(2*hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.drop = torch.nn.Dropout1d(0.1)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.bnorm1(out)
        out = self.drop(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.bnorm2(out)
        out = self.drop(out)

        out = self.fc3(out)
        out = self.relu(out)
        out = self.bnorm1(out)
        out = self.fc4(out)
        return out
import random as rand    
def load_data():
    embed_path = '/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/data/audio_embeddings'
    embed_npy = list(Path.Path(embed_path).glob('*.npy'))
    ANNOT_CSV_FILE = '/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/data/GEMS-INN_2023-01-30_expert.csv'

    emma_df = pd.read_csv(ANNOT_CSV_FILE, encoding="ISO-8859-1")
    emma_df.dropna(inplace=True)
    track_name = emma_df.title 
    artist_name = emma_df.artist

    # artist_name_ns = remove_special_chars(artist_name)
    # track_name_ns = remove_special_chars(track_name)
    train_feat = []
    test_feat = []
    valid_feat = []
    feat = np.empty((0,1728))
    counter = 0
    
    test_annot = []
    train_annot = []
    valid_annot = []


    rand.seed(42)
    num_samples = len(emma_df)
    indices = list(range(num_samples))
    rand.shuffle(indices)
    test_stop = int(num_samples * 0.15)
    valid_stop = test_stop + int(num_samples * 0.15)
    #get train test valid indices
    test_indices = indices[:test_stop]
    valid_indices = indices[test_stop:valid_stop]
    train_indices = indices[valid_stop:]

    train_counter =0
    test_counter =0
    valid_counter =0
    for i in embed_npy:

        name = i.stem
        #reset flag
        train = False
        test = False
        valid = False
        #look for feat in emma df for current song
        for col_idx, col_series in emma_df.iterrows():

            matched = False
            artist_name = col_series.artist
            track_name = col_series.title
            if artist_name.lower().startswith('the '):
                artist_name = artist_name[4:]

            artist_name_ns = remove_special_chars(artist_name)
            track_name_ns = remove_special_chars(track_name)


            audio_file_short = name[2:-4]
                # print(audio_file_short)
            if len(audio_file_short.split('_')) == 2:
                audio_artist_name, audio_track_name = audio_file_short.split('_')
                if audio_artist_name.lower().startswith(artist_name_ns.lower()) and \
                        track_name_ns[:7].lower().startswith(audio_track_name.lower()):
                    matched = True
            #if matched check which set it belongs to and flag it
            if matched:
                if col_idx in test_indices:
                    test = True
                    #feat = np.array(col_series[['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']])
                if col_idx in train_indices:
                    train = True
                    #feat = np.array(col_series[['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']])
                if col_idx in valid_indices:
                    valid = True
                annot = np.array(col_series[['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']])

        sample = np.load(f'/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/data/audio_embeddings/{name}.npy', allow_pickle=True)
        nr = sample.shape[1]
        sample = sample.T
        sample = np.split(sample.flatten(), nr)

        annot_con = [] #np.empty((0,9))
        #duplicate annot as one track can have multiple feats (windowing)
        for i in range(nr):
            annot_con.append(annot)
            #annot_con = np.vstack((annot_con, annot))

        if train:
            train_annot = train_annot+annot_con
            #train_annot = np.concatenate((train_annot, annot_con), axis=0)
            train_feat = train_feat + sample  #np.concatenate((train_feat, sample), axis=0)
            train_counter += 1
        if test:
            test_annot = test_annot+annot_con
            #test_annot = np.concatenate((test_annot, annot_con), axis=0)
            test_feat =  test_feat + sample  #np.concatenate((test_feat, sample), axis=0)
            test_counter += 1
        if valid:
            valid_annot = valid_annot+annot_con
            #valid_annot = np.concatenate((valid_annot, annot_con), axis=0)
            valid_feat =  valid_feat + sample #np.concatenate((valid_feat, sample), axis=0)
            valid_counter += 1

        #print(feat_con)
        
        # counter += 1 
        #feat = np.concatenate((annot, sample), axis=0)

        
        

        # if counter == 7:
        #     break

    assert len(train_annot) == len(train_feat)
    assert len(test_annot) == len(test_feat)
    assert len(valid_annot) == len(valid_feat)
    print(f"Number of train samples: {len(train_annot)}!")
    print(f"Number of test samples: {len(test_annot)}!")
    print(f"Number of valid samples: {len(valid_annot)}!")
    return train_annot, train_feat, test_annot, test_feat, valid_annot, valid_feat


class CustomDataset(Dataset):
    def __init__(self, feat, annots):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.annots = torch.tensor(annots, dtype=torch.float32)
    
    def __len__(self):
        return len(self.feat)
    
    def __getitem__(self, index):
        return self.feat[index], self.annots[index]
train_annot, train_feat, test_annot, test_feat, valid_annot, valid_feat = load_data()

# if artist_name.lower().startswith('the '):
#             artist_name = artist_name[4:]
train_dataset = CustomDataset(train_feat, train_annot)
test_dataset = CustomDataset(test_feat, test_annot)
valid_dataset = CustomDataset(valid_feat, valid_annot)
batch_size = 10

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/model/embed_model.pth'

num_epochs = 9999999
input_size = train_feat[0].shape[0]  # Replace with your input size
hidden_size = 64  # Replace with desired hidden layer size
output_size = 9   # Replace with your output size

model = embeddingsNN(input_size, hidden_size, output_size)
criterion = nn.L1Loss() #
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay  = 1e-7)
best_val_loss = float('inf')
min_valid_loss = np.inf

patience = 1000


for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()  
    model = model.to(device)
    # Training
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()


    valid_loss = 0.0
    model.eval()     # Optional when not using Model Specific layer
    for data, labels in valid_loader:
        data, labels = data.to(device), labels.to(device)
        
        output = model(data)
        loss = criterion(output,labels)
        valid_loss = loss.item() * data.size(0)
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), model_path)
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        counter = 0 
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered! No improvement.")
            break  # Stop training if no improvement after patience epochs


model = embeddingsNN(input_size, hidden_size, output_size)
model = model.to(device)
model.load_state_dict(torch.load(model_path)) 
model.eval()


# Make predictions on the test set

annotations = []
predictions = []


with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)  # Assuming 'device' is defined as the same device as your model
        outputs = model(inputs)

        predictions.append(outputs.cpu().numpy()[0])
        annotations.append(labels.cpu().numpy()[0])
errors = np.array(predictions) - np.array(annotations)

error_df = pd.DataFrame(errors)
    # Calculate Mean Squared Error (MSE)
mean_errors_df = error_df.mean()
mean_abs_errors_df = error_df.abs().mean()
mse_df = (error_df**2).mean()
rmse_df = np.sqrt(mse_df)
r2 = r2_score((annotations), (predictions))
formatted_out = np.round(predictions, 2)
formatted_annot = np.round(annotations, 2)
for i in range(len(predictions)):
        print(f"Annotation:{formatted_annot[i]} ") 
        print(f"Prediction:{formatted_out[i]} ")
        print(f"Mean: [13.28, 9.53, 13.17, 9.47, 15.06, 19.69, 12.22, 8.12, 3.24] \n")

evaluation = {
                'ME' : [np.mean(mean_errors_df)],
                'MAE' :  [np.mean(mean_abs_errors_df)],
                'MSE' : [np.mean(mse_df)],
                'RMSE' : [np.mean(rmse_df)],
                "R2" : r2
            }
print(f"\noverall: \nME: {np.mean(mean_errors_df):.2f} \nMAE: {np.mean(mean_abs_errors_df):.2f}\nMSE: {np.mean(mse_df):.2f}\nRMSE: {np.mean(rmse_df):.2f}\nR2: {r2:.2f}")