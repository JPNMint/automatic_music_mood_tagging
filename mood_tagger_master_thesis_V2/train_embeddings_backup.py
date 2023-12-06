import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pathlib as Path
import pandas as pd
from data import remove_special_chars
from sklearn.metrics import mean_squared_error, r2_score
import os


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
    


#def run_training()
gems9 = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
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
output_size = len(gems9)   # Replace with your output size
learning_rate = 0.001
oversampling = False
oversampling_tolerance = 0
oversampling_ratio = 0
oversampling_method = 'No oversampling'
data_augmentation = 0



model = embeddingsNN(input_size, hidden_size, output_size)
criterion = nn.L1Loss() #
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay  = 1e-7)
best_val_loss = float('inf')
min_valid_loss = np.inf

patience = 100


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
model_name ='Transfer_learning_embeddings'
evaluation = {
                'Model' : [model_name],
                'ME' : [np.mean(mean_errors_df)],
                'MAE' :  [np.mean(mean_abs_errors_df)],
                'MSE' : [np.mean(mse_df)],
                'RMSE' : [np.mean(rmse_df)],
                "R2" : r2
            }
    #metrics for detailed

csv_information = { 
                'Model' : model_name,
                #'resampling' : cfg.resampling,
                'Labels' : [['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']],
                'lr' : learning_rate,
                'loss_function' : 'MAE',
                'dense_weight_alpha': 0 ,
                'batch size' : batch_size,
                'snippet_duration_s' : 3,
                'oversampling': oversampling,
                'oversampling_tolerance': oversampling_tolerance,
                'oversampling_ratio' : oversampling_ratio,
                'oversampling_method': oversampling_method,
                'data_augmentation': data_augmentation


            }
MAE = ['Wonder MAE', 'Transcendence MAE', 'Nostalgia MAE', 'Tenderness MAE', 'Peacfulness MAE', 'Joy MAE', 'Power MAE', 'Tension MAE', 'Sadness MAE']
gems9_MAE = pd.DataFrame(columns=MAE)
gems9_pos = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
RMSE=  ['Wonder RMSE', 'Transcendence RMSE', 'Nostalgia RMSE', 'Tenderness RMSE', 'Peacfulness RMSE', 'Joy RMSE', 'Power RMSE', 'Tension RMSE', 'Sadness RMSE']
gems9_RMSE = pd.DataFrame(columns=RMSE)

if len(csv_information['Labels'][0])==1:
    position = gems9_pos.index(csv_information['Labels'][0][0])
    MAE_list = [np.nan] * 9
    MAE_list[position] = mean_abs_errors_df[0]
    gems9_MAE.loc[len(gems9_MAE)] = MAE_list
    RMSE_list = [np.nan] * 9
    RMSE_list[position] = rmse_df[0]
    gems9_RMSE.loc[len(gems9_RMSE)] = RMSE_list

if len(csv_information['Labels'][0]) == 9:


  
    gems9_MAE = pd.concat([gems9_MAE,pd.DataFrame([mean_abs_errors_df.values ], columns=MAE)], ignore_index=True) 


    gems9_RMSE = pd.concat([gems9_RMSE,pd.DataFrame([rmse_df.values], columns=RMSE)], ignore_index=True) 


csv_information.update(evaluation)

    #make copy for detailed csv
csv_information_detailed = csv_information

    ## if csv file does not exist, create blank file
import csv
if not os.path.isfile('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/performance/model_performance_list.csv'):
    header = ['Model', 'Labels', 'lr', 'dense_weight_alpha', 'snippet_duration_s', 'batch size',  'data_augumentation', 'ME', 'MAE', 'MSE', 'RMSE']
    with open('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/performance/model_performance_list.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header) 


csv_file = pd.read_csv('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/performance/model_performance_list.csv')
    
cur_model = csv_information['Model'][0]
cur_labels = f"{csv_information['Labels'][0]}"    
cur_lr = csv_information['lr']
cur_loss = csv_information['loss_function']
cur_alpha = csv_information['dense_weight_alpha']
cur_duration = csv_information['snippet_duration_s']
cur_batch = csv_information['batch size']
cur_os = csv_information['oversampling']
if not csv_information['data_augmentation']:
    cur_da = 0
    csv_information['data_augmentation']  = 0
else:
    cur_da = csv_information['data_augmentation']
        
if cur_os == False:
    cur_os_ratio = 0
    cur_os_tol = 0
    cur_os_method = 'No oversampling'
    csv_information['oversampling_ratio'] = 0
    csv_information['oversampling_tolerance'] = 0
    csv_information['oversampling_method'] = 'No oversampling'
    csv_information_detailed['oversampling_ratio'] = 0
    csv_information_detailed['oversampling_tolerance'] = 0
    csv_information_detailed['oversampling_method'] = 'No oversampling'


else:
    cur_os_ratio = csv_information['oversampling_ratio']
    cur_os_tol = csv_information['oversampling_tolerance']
    cur_os_method = csv_information['oversampling_method']
    cur_os_ratio = csv_information['oversampling_ratio']
    cur_os_tol = csv_information['oversampling_tolerance']
    cur_os_method = csv_information['oversampling_method']
    
    #create the dataframes for both csv
csv_information_df = pd.DataFrame(csv_information)
csv_information_detailed_df = pd.DataFrame(csv_information_detailed)

csv_information_detailed_df =  pd.concat([csv_information_detailed_df, gems9_MAE], axis=1)

csv_information_detailed_df =  pd.concat([csv_information_detailed_df, gems9_RMSE], axis=1)
    

    #write csv  
if not csv_file.empty:

        # if any(csv_file['Labels'].apply(lambda x: x == cur_labels)):!!!!!!!!!!!!!
        #     print('Triggered')!!!!!!!!!!
        
    condition = (
            (csv_file['Model'] == cur_model) &
            (csv_file['lr'] == cur_lr)  &
            (csv_file['Labels'] == cur_labels) &
            (csv_file['loss_function'] == cur_loss) &
            (csv_file['dense_weight_alpha'] == cur_alpha) &
            (csv_file['snippet_duration_s'] == cur_duration) &
            (csv_file['batch size'] == cur_batch) & 
            (csv_file['oversampling_ratio'] == cur_os_ratio) & 
            (csv_file['oversampling_tolerance'] == cur_os_tol) & 
            (csv_file['oversampling'] == cur_os) & 
            (csv_file['oversampling_method'] == cur_os_method ) &
            (csv_file['data_augmentation'] == cur_da )
            
            )



    if any(condition):
        csv_file = csv_file[~condition]
        print(csv_file)
            #csv_file[condition] = csv_information_df
        new_csv = pd.concat([csv_file, csv_information_df])

    else:
        new_csv = pd.concat([csv_file, csv_information_df])
else:
    new_csv = pd.concat([csv_file, csv_information_df])


new_csv.to_csv('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/performance/model_performance_list.csv', index=False)


  # #########SAVE ALL ERRORS FOR EACH LABEL IN CSV###############DETAILED SAVE

if not os.path.isfile('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/performance/model_performance_list_detailed.csv'):
    header = ['Model', 'Labels', 'lr', 'dense_weight_alpha', 'snippet_duration_s', 'batch size',  'data_augmentation', 'ME', 'MAE', 'MSE', 'RMSE',  'Wonder MAE', 'Transcendence MAE', 'Nostalgia MAE', 'Tenderness MAE', 'Peacfulness MAE', 'Joy MAE', 'Power MAE', 'Tension MAE', 'Sadness MAE', 'Wonder RMSE', 'Transcendence RMSE', 'Nostalgia RMSE', 'Tenderness RMSE', 'Peacfulness RMSE', 'Joy RMSE', 'Power RMSE', 'Tension RMSE', 'Sadness RMSE']
        
    with open('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/performance/model_performance_list_detailed.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header) 
    csv_information_detailed_df.to_csv('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/performance/model_performance_list_detailed.csv', index=False)
    
else:
    csv_file = pd.read_csv('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/performance/model_performance_list_detailed.csv')


        # if any(csv_file['Labels'].apply(lambda x: x == cur_labels)):!!!!!!!!!!!!!
        #     print('Triggered')!!!!!!!!!!
        
    condition = (
            (csv_file['Model'] == cur_model) &
            (csv_file['lr'] == cur_lr)  &
            (csv_file['Labels'] == cur_labels) &
            (csv_file['loss_function'] == cur_loss) &
            (csv_file['dense_weight_alpha'] == cur_alpha) &
            (csv_file['snippet_duration_s'] == cur_duration) &
            (csv_file['batch size'] == cur_batch) & 
            (csv_file['oversampling_ratio'] == cur_os_ratio) & 
            (csv_file['oversampling_tolerance'] == cur_os_tol) & 
            (csv_file['oversampling'] == cur_os) & 
            (csv_file['oversampling_method'] == cur_os_method ) &
            (csv_file['data_augmentation'] == cur_da )
            
            )

    print(csv_file)
    print(csv_information_detailed_df)
    new_csv = pd.concat([csv_file, csv_information_detailed_df])
    if any(condition):
        csv_file = csv_file[~condition]
            #csv_file[condition] = csv_information_df
        new_csv = pd.concat([csv_file, csv_information_detailed_df])

    else:
        new_csv = pd.concat([csv_file, csv_information_detailed_df])



    new_csv.to_csv('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/performance/model_performance_list_detailed.csv', index=False)






print(f"\noverall: \nME: {np.mean(mean_errors_df):.2f} \nMAE: {np.mean(mean_abs_errors_df):.2f}\nMSE: {np.mean(mse_df):.2f}\nRMSE: {np.mean(rmse_df):.2f}\nR2: {r2:.2f}")