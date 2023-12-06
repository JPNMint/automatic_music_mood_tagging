import pandas as pd
import numpy as np
import os
#import librosa
import torch
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from collections import defaultdict
from data.sets import SnippetDataset, FramesToSamples, ListDataset
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
from denseweight import DenseWeight
import math
from data.utils import GetKDE, bisection
from torchaudio.transforms import PitchShift
from data.create_augmentation import create_augmentation

SAMPLE_RATE = 44100

DATA_PATH = os.path.dirname(os.path.realpath(__file__))

# TODO move data out of here
AUDIO_PATH = os.path.join(DATA_PATH, 'audio')
CACHE_PATH = os.path.join(DATA_PATH, 'cache')
#ANNOT_CSV_FILE = os.path.join(DATA_PATH, "GEMS-INN_2023-01-30_expert.csv")
ANNOT_CSV_FILE = os.path.join(DATA_PATH, "GEMS-INN_2023-01-30_expert.csv")

os.makedirs(CACHE_PATH, exist_ok=True)

GEMS_9 = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
GEMS_9_ext = ['artist', 'title', 'Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']

NUM_CLASSES = len(GEMS_9)
GENRE_MAP = {'H': 'Hip-Hop', 'K': 'Classical', 'P': 'Pop'}


class FeatureSetup:
    # TODO fix this mess :)
    def __init__(
        self,
        name: str,
        sample_rate: int,
        channels: int,
        fps: float,
        frame_size: int,
        num_bins: int,
        fmin: float,
        diff: bool,
        freq_scale: str,
        mag_scale: str,
        **kwargs,
    ):
        self.name = name

        self.sample_rate = sample_rate
        self.channels = channels
        self.fps = fps
        self.hop_size = int(sample_rate / fps)
        self.frame_size = frame_size
        self.num_bins = num_bins
        self.fmin = fmin
        self.diff = diff
        self.freq_scale = freq_scale
        self.mag_scale = mag_scale

        self.feat_size = np.sum(num_bins) * (2 if diff else 1)

        self.__dict__.update(kwargs)


def remove_special_chars(name_with_chars):
    output_name = name_with_chars
    output_name = output_name.replace('3. Klavierkonzert, 3. Satz', '3KK3Sat')
    output_name = output_name.replace('3. Klaviersonate, 4. Satz', '3KS4Sat')
    output_name = output_name.replace('Klaviersonate No. 7', 'KSNo7')
    output_name = output_name.replace('Preludes op. 11 no. 1', 'Pre1')
    output_name = output_name.replace('Preludes op. 11 no. 15', 'Pre15')
    output_name = output_name.replace('Die 4 Jahreszeiten', '4Jahr')
    output_name = output_name.replace('Sinfonie', 'Sinf')
    output_name = output_name.replace(' ', '')
    output_name = output_name.replace(':', '-')
    output_name = output_name.replace('\'', '')
    output_name = output_name.replace('.', '')
    output_name = output_name.replace('!', '')
    output_name = output_name.replace('*', 'X')
    output_name = output_name.replace('&', 'and')
    output_name = output_name.replace('$', 'S')
    output_name = output_name.replace('\xfc', 'ue')
    output_name = output_name.replace(',', '')
    output_name = output_name.replace('(', '')
    output_name = output_name.replace(')', '')
    output_name = output_name.replace('é', 'e')
    output_name = output_name.replace('É', 'E')
    output_name = output_name.replace('ö', 'oe')
    output_name = output_name.replace('á', 'a')
    output_name = output_name.replace('ä', 'ae')

    return output_name

class augmented_Entry:
    def __init__(self, audio_file_name, data_row, train_y, AUG_CACHE_PATH, data_augmentation):
        self.genre_key = audio_file_name[:1]
        self.artist_name = data_row.artist
        self.track_name = data_row.title
        self.audio_file_name = audio_file_name
        self.data_row = data_row
        self.audio_data = None
        self.train_y = train_y,
        self.AUG_CACHE_PATH = AUG_CACHE_PATH

    def _get_cache_path(self):
        
        base_name, _ = os.path.splitext(self.audio_file_name)

        return os.path.join(self.AUG_CACHE_PATH, base_name.lower()+ '.npy') #+ '_augmented'

    # def _get_full_audio_path(self):
    #     return os.path.join(AUDIO_PATH, self.audio_file_name)

    def get_audio(self):
        if self.audio_data is None:
            cache_file = self.re_cache()

            self.audio_data = np.load(cache_file, mmap_mode= None)#'r')
        return self.audio_data.reshape(self.audio_data.shape[0],1)

    def re_cache(self):
        cache_file = self._get_cache_path()
        if not os.path.exists(cache_file):            
            create_augmentation(data_augmentation)
        return cache_file

    def get_gems_9(self):
        if isinstance(self.train_y, str):
            return self.data_row[[self.train_y]].to_numpy(dtype=np.float32)
        # if len(self.train_y) == 1:
        #     print('Only one variable!')
        #     print(f'y is {self.train_y}!')
        #     testing = self.data_row[self.train_y[0]]
        #     return self.data_row[[self.train_y[0]]].to_numpy(dtype=np.float32)
        else:
            return self.data_row[GEMS_9].to_numpy(dtype=np.float32)

    def get_name(self):
        return f"{self.artist_name} {self.track_name} augmented"

    def get_genre(self):
        return self.genre_key


class Entry:
    def __init__(self, audio_file_name, data_row, train_y):
        self.genre_key = audio_file_name[:1]
        self.artist_name = data_row.artist
        self.track_name = data_row.title
        self.audio_file_name = audio_file_name
        self.data_row = data_row
        self.audio_data = None
        self.train_y = train_y

    def _get_cache_path(self):
        
        base_name, _ = os.path.splitext(self.audio_file_name)

        return os.path.join(CACHE_PATH, base_name.lower() + '.npy')

    def _get_full_audio_path(self):
        return os.path.join(AUDIO_PATH, self.audio_file_name)

    def get_audio(self):
        if self.audio_data is None:
            cache_file = self.re_cache()

            self.audio_data = np.load(cache_file, mmap_mode= None)#'r')
        return self.audio_data

    def re_cache(self):
        cache_file = self._get_cache_path()
        if not os.path.exists(cache_file):
            full_file_name = self._get_full_audio_path()
            audio_data, sr = librosa.load(full_file_name, sr=SAMPLE_RATE)
            audio_data = audio_data[:, None]  # add feature dim
            np.save(cache_file, audio_data)
        return cache_file

    def get_gems_9(self):
        if isinstance(self.train_y, str):
            return self.data_row[[self.train_y]].to_numpy(dtype=np.float32)
        # if len(self.train_y) == 1:
        #     print('Only one variable!')
        #     print(f'y is {self.train_y}!')
        #     testing = self.data_row[self.train_y[0]]
        #     return self.data_row[[self.train_y[0]]].to_numpy(dtype=np.float32)
        else:
            return self.data_row[GEMS_9].to_numpy(dtype=np.float32)

    def get_name(self):
        return f"{self.artist_name} {self.track_name}"

    def get_genre(self):
        return self.genre_key



def match_files(train_y, task):
    if task == 'classification':
        emma_df = pd.read_csv('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/data/GEMS-INN_2023-01-30_expert_binary.csv', encoding="ISO-8859-1")
    else:

        emma_df = pd.read_csv(ANNOT_CSV_FILE, encoding="ISO-8859-1")
    emma_df.dropna(inplace=True)
    audio_files = os.listdir(AUDIO_PATH)
    audio_files = [cur_file for cur_file in audio_files if cur_file.endswith('.mp3')]
    not_found_count = 0
    matched = 0

    entries = []
    for col_idx, col_series in emma_df.iterrows():
        artist_name = col_series.artist
        track_name = col_series.title
        if artist_name.lower().startswith('the '):
            artist_name = artist_name[4:]

        artist_name_ns = remove_special_chars(artist_name)
        track_name_ns = remove_special_chars(track_name)

        found = False

        for cur_audio_file in audio_files:
            audio_file_short = cur_audio_file[2:-4]
            # print(audio_file_short)
            if len(audio_file_short.split('_')) == 2:
                audio_artist_name, audio_track_name = audio_file_short.split('_')
                if audio_artist_name.lower().startswith(artist_name_ns.lower()) and \
                        track_name_ns[:7].lower().startswith(audio_track_name.lower()):
                    # audio_track_name.lower() == track_name_ns[:7].lower():
                    # print(f'    found! {artist_name}  {track_name} == {audio_file_short}')
                    found = True
                    matched += 1
                    entries.append(Entry(cur_audio_file, col_series, train_y = train_y))
                    break
            else:
                print(f"problem with {audio_file_short} - more than 2 parts")
        if not found:
            print(f"not found! {artist_name} {track_name}")
            not_found_count += 1

    print(f"not found count :  {not_found_count}   - matched: {matched}")

    return entries

def match_files_augmentation(train_y, data_augmentation):
    emma_df = pd.read_csv(ANNOT_CSV_FILE, encoding="ISO-8859-1")
    emma_df.dropna(inplace=True)
    #TODO
    if data_augmentation in [1,2,4,8]:
        AUG_AUDIO_PATH = f'/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/data/cache_augmented_{data_augmentation}'
    audio_files = os.listdir(AUDIO_PATH)
    audio_files = [cur_file for cur_file in audio_files if cur_file.endswith('.mp3')]
    not_found_count = 0
    matched = 0

    entries = []
    for col_idx, col_series in emma_df.iterrows():
        artist_name = col_series.artist
        track_name = col_series.title
        if artist_name.lower().startswith('the '):
            artist_name = artist_name[4:]

        artist_name_ns = remove_special_chars(artist_name)
        track_name_ns = remove_special_chars(track_name)

        found = False

        for cur_audio_file in audio_files:
            audio_file_short = cur_audio_file[2:-4]
            # print(audio_file_short)
            if len(audio_file_short.split('_')) == 2:
                audio_artist_name, audio_track_name = audio_file_short.split('_')
                if audio_artist_name.lower().startswith(artist_name_ns.lower()) and \
                        track_name_ns[:7].lower().startswith(audio_track_name.lower()):
                    # audio_track_name.lower() == track_name_ns[:7].lower():
                    # print(f'    found! {artist_name}  {track_name} == {audio_file_short}')
                    found = True
                    matched += 1
                    entries.append(augmented_Entry(cur_audio_file, col_series, train_y = train_y, AUG_CACHE_PATH = AUG_AUDIO_PATH, data_augmentation = data_augmentation))
                    break
            else:
                print(f"problem with {audio_file_short} - more than 2 parts")
        if not found:
            print(f"not found! {artist_name} {track_name}")
            not_found_count += 1

    print(f"not found count :  {not_found_count}   - matched: {matched}")

    return entries


class RandomPartitioner:
    def __init__(self, feats, annots, targs, names, genres, seed, train_y,mode, tolerance, oversampling_method = None, 
                oversampling = False,test_split=0.15, valid_split=0.15, os_ratio = 1, data_augmentation = 'default', data_aug_data = {}, alpha = 1):
        self.feats = feats
        self.annots = annots
        self.targs = targs
        self.names = names
        self.genres = genres
        self.seed = seed
        self.tolerance = tolerance
        num_samples = len(feats)
        assert num_samples == len(annots) == len(targs) == len(names) == len(genres)
        import random as rand
        rand.seed(seed)
        indices = list(range(num_samples))
        rand.shuffle(indices)
        test_stop = int(num_samples * test_split)
        valid_stop = test_stop + int(num_samples * valid_split)
        self.test_indices = indices[:test_stop]
        self.valid_indices = indices[test_stop:valid_stop]
        self.train_indices = indices[valid_stop:]
        self.oversampling = oversampling
        self.oversampling_method = oversampling_method
        
        if data_augmentation == 'default':
            self.aug_feats  = data_aug_data['aug_feats']
            self.aug_annots = data_aug_data['aug_annots']
            self.aug_targs = data_aug_data['aug_targs']
            self.aug_names = data_aug_data['aug_names']
            self.aug_genres = data_aug_data['aug_genres']
            self.aug_train_indices = [num + len(feats) for num in self.train_indices]
            self.test = self.train_indices
            self.train_indices = self.train_indices + self.aug_train_indices

            self.feats = self.feats+self.aug_feats
            self.annots = np.concatenate((self.annots, self.aug_annots))

            self.targs = np.concatenate((self.targs, self.aug_targs))
            self.names = self.names + self.aug_names
            self.genres = self.aug_genres + self.aug_genres
        elif data_augmentation == 'oversampling':
            self.feats  = data_aug_data['aug_feats']
            self.annots = data_aug_data['aug_annots']
            self.targs = data_aug_data['aug_targs']
            self.names = data_aug_data['aug_names']
            self.genres = data_aug_data['aug_genres']
            #self.train_indices = [num + len(feats) for num in self.train_indices]
        else:
            pass
##############
        #self.test_names = [self.names[index] for index in self.test_indices]
        #self.train_names = [self.names[index] for index in self.train_indices]
###########
        ##oversampling
        smogn = False
        naive = False
        if self.oversampling == True:

            #####smogn approach#### 3.86 trancsendence ohne smogn

            ####CHECK THIS 
            #### PREDICTION IS REDICULOUSLY HIGH

            if smogn and isinstance(train_y, str):
                
                import smogn

                train_set = [self.annots[i] for i in self.train_indices] 
                train_set = pd.DataFrame(train_set, columns=[train_y])
                names = [self.names[i] for i in self.train_indices] 
                name_indices_set = pd.DataFrame(names, columns= ['names'])
                name_indices_set['indices'] = self.train_indices

                train_set['names'] = names
                #emma['indices'] = self.train_indices
                
                ## conduct smogn
                smogn_dataset = smogn.smoter(
                    
                    data = train_set,  ## pandas dataframe
                    y = train_y,  ## string ('header name')
                    samp_method = 'balance'
                )
                smogn_dataset = smogn_dataset.merge(name_indices_set, on='names', how='left')
                print(smogn_dataset)
                self.train_indices = smogn_dataset['indices']


            ### naive approach

            elif self.oversampling_method == 'naive':
                print(f'Oversampling by {self.oversampling_method}')

                meanvals = [13.282972972972974, 9.536189189189187, 13.177081081081083, 9.475270270270272, 15.062567567567566, 19.696594594594593, 12.22191891891892, 8.125810810810812, 3.2404324324324327]
                candidate = []
                for i in range(len(self.annots)): #each list in list 
                    for pos in range(len(self.annots[i])):
                        annot = self.annots[i][pos]
                        mean = meanvals[pos]
                        if np.abs(annot-mean) > self.tolerance:
                            if i not in candidate:
                                candidate.append(i)

                ## see if candidate is in train, get matching indices
                matching_indices = []
                for index, value in enumerate(candidate):
                    if value in self.train_indices: 
                        #allconv01
                        #without oversamp RMSE: 6.44
                        matching_indices.append(value) #tol 20 52 RMSE: 6.34 #tol 25 23 RMSE: 7.01 #tol30 RMSE: 7.52
                        #matching_indices.append(value) ## tol 10 RMSE: 7.01 ,tol 20, 104 RMSE: 7.58 #tol 25, 46 RMSE: 6.72 #tol30 RMSE: 6.70

                if mode == "train" :
                    print(f"Oversampling {len(matching_indices)} samples! (When training, does not affect testing), score tolerance: {self.tolerance}")
                self.train_indices.extend(matching_indices)
            elif self.oversampling_method == 'adaptive_density_oversampling' or self.oversampling_method == 'density_augmentation_oversampling':
                print(f'Oversampling by {self.oversampling_method}')
                print(f'Oversampling! Treshold is set to {tolerance} and ratio to {os_ratio}')
                train_annots = annots[self.train_indices]
                gdw = GetKDE(alpha=alpha)
                dense = gdw.fit(train_annots)


                threshold = dense<tolerance #bool

                print((dense>0.5).sum())
                print((dense<0.5).sum())
                new_len = math.floor(((dense>0.5).sum() - threshold.sum())/os_ratio)
                
                                
                
                w = (1-dense)[threshold] ##

                from itertools import compress
                dense_thresh_indices = list(compress(self.train_indices, threshold)) 
                                #np.array(self.train_indices)[threshold]

                import random
                oversampled = []
                candidates = []
                counter = 0
                while len(dense_thresh_indices)>0:
                    
                    if counter > new_len:
                        break

                    random_chosen = random.choices(dense_thresh_indices,weights = w)
                    
                    choice = random_chosen[0] #np.where(train_annots == random_chosen)[0][0]
                    candidates.append(choice)
                    self.train_indices = self.train_indices + candidates
                    oversampled.append(choice)
                    candidates = []
                    train_annots_new = annots[self.train_indices]
                    gdw_new = GetKDE(alpha=alpha)
                    dense = gdw_new.fit(train_annots_new)
                    #look into this, why does it only resample 330 at the end in tenderness
                    threshold = dense<tolerance
                    w = (1-dense)[threshold]
                    dense_thresh_indices = list(compress(self.train_indices, threshold)) 
                    counter += 1

            elif self.oversampling_method == 'adaptive_density_oversampling_V2' :
                print(f'Oversampling by {self.oversampling_method}')
                print(f'Oversampling! Treshold is set to {tolerance} and ratio to {os_ratio}')
                train_annots = annots[self.train_indices]
                gdw = GetKDE(alpha=alpha)
                dense = gdw.fit(train_annots)


                threshold = dense<tolerance #bool

                print((dense>0.5).sum())
                print((dense<0.5).sum())
                new_len = math.floor(((dense>0.5).sum() - threshold.sum())/os_ratio)
                
                                
                print(dense[161])
                w = (1-dense)[threshold] ##

                from itertools import compress
                dense_thresh_indices = list(compress(self.train_indices, threshold)) 
                                #np.array(self.train_indices)[threshold]

                import random
                oversampled = []
                candidates = []
                counter = 0
                #TODO add seed
                while len(dense_thresh_indices)>0:
                    
                    if counter > new_len:
                        break

                    random_chosen = random.choices(dense_thresh_indices,weights = w)
                    #choose from under 0.5, if already chosen, keep if under 0.2 else new choice
                    choice = random_chosen[0] #np.where(train_annots == random_chosen)[0][0]
                    chosen = False
                    counter = 0

                    while chosen == False:
                        counter += 1
                        # print(np.isin(choice, oversampled))
                        if np.isin(choice, oversampled):
                            
                            # print((1-w[dense_thresh_indices.index(choice)]))
                            
                            if (1-w[dense_thresh_indices.index(choice)]) < 0.15:
                                candidates.append(choice)
                                # print(f'{choice} chosen')
                                chosen = True
                            else:
                                # print(f'{choice } not chosen')
                                random_chosen = random.choices(dense_thresh_indices,weights = w)
                                choice = random_chosen[0]
                                if not np.isin(choice, self.train_indices):
                                    chosen = True
                        if counter == 30:
                            print('No samples left under density of 0.15')
                            break
                        else:
                            candidates.append(choice)
                            # print(f'{choice} chosen')
                            chosen = True




                    self.train_indices = self.train_indices + candidates
                    oversampled.append(choice)
                    candidates = []
                    train_annots_new = annots[self.train_indices]
                    gdw_new = GetKDE(alpha=alpha)
                    dense = gdw_new.fit(train_annots_new)
                    #look into this, why does it only resample 330 at the end in tenderness
                    threshold = dense<tolerance
                    w = (1-dense)[threshold]


                    dense_thresh_indices = list(compress(self.train_indices, threshold)) 


            elif self.oversampling_method == 'density_oversampling_V2':
                #this will only work with one label
                #y = np.array(self.annots)
                print(f'Oversampling by {self.oversampling_method}')
                print(f'Oversampling! Treshold is set to {tolerance} and ratio to {os_ratio}')
                train_annots = annots[self.train_indices]
                gdw = GetKDE()
                dense = gdw.fit(train_annots)
                threshold = dense<tolerance #th
                print((dense>0.5).sum())
                print((dense<0.5).sum())
                import random
                
                candidates = []
                w = (dense)[threshold]
                #train_annots_thresh = train_annots[threshold]

                from itertools import compress
                dense_thresh_indices = list(compress(self.train_indices, threshold)) 
                #add factor
                self.train_indices = self.train_indices + dense_thresh_indices 


            elif self.oversampling_method == 'density_oversampling':
                #this will only work with one label
                #y = np.array(self.annots)
                print(f'Oversampling by {self.oversampling_method}')
                print(f'Oversampling! Treshold is set to {tolerance} and ratio to {os_ratio}')
                train_annots = annots[self.train_indices]
                gdw = GetKDE(alpha=alpha)
                dense = gdw.fit(train_annots)
                threshold = dense<tolerance #th
                print((dense>0.5).sum())
                print((dense<0.5).sum())
                new_len = math.floor(((dense>0.5).sum() - threshold.sum())/os_ratio)
                import random
                
                candidates = []
                w = (1-dense)[threshold]
                #train_annots_thresh = train_annots[threshold]

                from itertools import compress
                dense_thresh_indices = list(compress(self.train_indices, threshold)) 
                #np.array(self.train_indices)[threshold]
                for i in range(new_len):
                    random_chosen = random.choices(dense_thresh_indices,weights = w)
                    #arange(a.size)
                    choice = random_chosen[0] #np.where(train_annots == random_chosen)[0][0]
                    candidates.append(choice)
                self.train_indices = self.train_indices + candidates
    
            elif self.oversampling_method == 'density_oversampling_V3':
                #this will only work with one label
                #y = np.array(self.annots)
                from itertools import compress
                print(f'Oversampling by {self.oversampling_method}')
                print(f'Oversampling! Treshold is set to {tolerance} and ratio to {os_ratio}')
                train_annots = annots[self.train_indices]
                gdw = GetKDE(alpha=alpha)
                dense = gdw.fit(train_annots)
                threshold = dense<tolerance #th
                print((dense>0.5).sum())
                print((dense<0.5).sum())

                
                dense_thresh_filter = (dense)[threshold]



                dense_thresh_indices = list(compress(self.train_indices, threshold)) #indices of under the threshhold
                q1 = np.percentile(dense_thresh_filter, 25)
                q2 = np.percentile(dense_thresh_filter, 50)
                q3 = np.percentile(dense_thresh_filter, 75)
                indices_smaller_than_q1 = [index for index, value in enumerate(dense_thresh_filter) if value < q1]
                indices_between_q1_and_q2 = [index for index, value in enumerate(dense_thresh_filter) if q1 <= value <= q2]
                indices_higher_than_q3 = [index for index, value in enumerate(dense_thresh_filter) if value > q3]
                smaller_q1 = [dense_thresh_indices[i] for i in indices_smaller_than_q1] 
                between_q1_and_q2 = [dense_thresh_indices[i] for i in indices_between_q1_and_q2] 
                higher_than_q3  = [dense_thresh_indices[i] for i in indices_higher_than_q3] 

                self.train_indices = self.train_indices + smaller_q1 *3 + between_q1_and_q2*2 + higher_than_q3*1

            elif self.oversampling_method == 'density_oversampling':
                from itertools import compress
                #this will only work with one label
                #y = np.array(self.annots)
                print(f'Oversampling by {self.oversampling_method}')
                print(f'Oversampling! Treshold is set to {tolerance} and ratio to {os_ratio}')
                train_annots = annots[self.train_indices]
                gdw = GetKDE()
                dense = gdw.fit(train_annots)
                threshold = dense<tolerance #th
                print((dense>0.5).sum())
                print((dense<0.5).sum())
                new_len = math.floor(((dense>0.5).sum() - threshold.sum())/os_ratio)
                import random
                
                candidates = []
                w = (1-dense)[threshold]
                #train_annots_thresh = train_annots[threshold]

                
                dense_thresh_indices = list(compress(self.train_indices, threshold)) 
                #np.array(self.train_indices)[threshold]
                for i in range(new_len):
                    random_chosen = random.choices(dense_thresh_indices,weights = w)
                    #arange(a.size)
                    choice = random_chosen[0] #np.where(train_annots == random_chosen)[0][0]
                    candidates.append(choice)
                self.train_indices = self.train_indices + candidates
    
            elif self.oversampling_method == 'average_density_oversampling' or self.oversampling_method == 'density_oversampling_augmentation':
                from itertools import compress
                train_annots = annots[self.train_indices]
                #gdw = GetKDE()
                self.dw_all ={}
                stacked_arrays = []
                for i in range(train_annots.shape[1]):
                    self.dw_cur = f'dw{i}'
                    self.dw_all[self.dw_cur] = GetKDE()#DenseWeight(alpha=1)
                    weighted_targs = self.dw_all[self.dw_cur].fit(train_annots[:, i])
                    stacked_arrays.append(weighted_targs)
                

                stacked_weight = np.stack(stacked_arrays, axis=1)

                average_dense = np.mean(stacked_weight, axis=1)
                #get average for that data point
                #oversample based on that 
                threshold = average_dense<tolerance #th
                dense_thresh_filter = (average_dense)[threshold]
                dense_thresh_indices = list(compress(self.train_indices, threshold)) #indices of under the threshhold

                self.train_indices = self.train_indices + dense_thresh_indices

    
            elif self.oversampling_method == 'average_density_oversampling_V2'or self.oversampling_method == 'density_oversampling_augmentation_V2':
                from itertools import compress
                train_annots = annots[self.train_indices]
                gdw = GetKDE()
                self.dw_all ={}
                stacked_arrays = []
                for i in range(train_annots.shape[1]):
                    self.dw_cur = f'dw{i}'
                    self.dw_all[self.dw_cur] = GetKDE(alpha=alpha)#DenseWeight(alpha=1)
                    weighted_targs = self.dw_all[self.dw_cur].fit(train_annots[:, i])
                    stacked_arrays.append(weighted_targs)
                

                stacked_weight = np.stack(stacked_arrays, axis=1)

                average_dense = np.mean(stacked_weight, axis=1)
                #get average for that data point
                #oversample based on that 
                threshold = average_dense<tolerance #th
                print((average_dense>0.5).sum())
                print((average_dense<0.5).sum())

                
                dense_thresh_filter = (average_dense)[threshold]

                dense_thresh_indices = list(compress(self.train_indices, threshold)) #indices of under the threshhold
                q1 = np.percentile(dense_thresh_filter, 25)
                q2 = np.percentile(dense_thresh_filter, 50)
                q3 = np.percentile(dense_thresh_filter, 75)
                indices_smaller_than_q1 = [index for index, value in enumerate(dense_thresh_filter) if value < q1]
                indices_between_q1_and_q2 = [index for index, value in enumerate(dense_thresh_filter) if q1 <= value <= q2]
                indices_higher_than_q3 = [index for index, value in enumerate(dense_thresh_filter) if value > q3]
                smaller_q1 = [dense_thresh_indices[i] for i in indices_smaller_than_q1] 
                between_q1_and_q2 = [dense_thresh_indices[i] for i in indices_between_q1_and_q2] 
                higher_than_q3  = [dense_thresh_indices[i] for i in indices_higher_than_q3] 

                self.train_indices = self.train_indices + smaller_q1 *3 + between_q1_and_q2*2 + higher_than_q3*1

            elif self.oversampling_method == 'average_density_oversampling_V3':
                from itertools import compress
                train_annots = annots[self.train_indices]
                gdw = GetKDE()
                self.dw_all ={}
                stacked_arrays = []
                for i in range(train_annots.shape[1]):
                    self.dw_cur = f'dw{i}'
                    self.dw_all[self.dw_cur] = GetKDE(alpha=alpha)#DenseWeight(alpha=1)
                    weighted_targs = self.dw_all[self.dw_cur].fit(train_annots[:, i])
                    stacked_arrays.append(weighted_targs)
                

                stacked_weight = np.stack(stacked_arrays, axis=1)

                average_dense = np.mean(stacked_weight, axis=1)
                #get average for that data point
                #oversample based on that 
                threshold = average_dense<tolerance #th
                print((average_dense>0.5).sum())
                print((average_dense<0.5).sum())

                
                dense_thresh_filter = (average_dense)[threshold]

                dense_thresh_indices = list(compress(self.train_indices, threshold)) #indices of under the threshhold
                q1 = np.percentile(dense_thresh_filter, 25)
                q2 = np.percentile(dense_thresh_filter, 50)
                q3 = np.percentile(dense_thresh_filter, 75)
                indices_smaller_than_q1 = [index for index, value in enumerate(dense_thresh_filter) if value < q1]
                indices_between_q1_and_q2 = [index for index, value in enumerate(dense_thresh_filter) if q1 <= value <= q2]
                indices_higher_than_q3 = [index for index, value in enumerate(dense_thresh_filter) if value > q3]
                smaller_q1 = [dense_thresh_indices[i] for i in indices_smaller_than_q1] 
                between_q1_and_q2 = [dense_thresh_indices[i] for i in indices_between_q1_and_q2] 
                higher_than_q3  = [dense_thresh_indices[i] for i in indices_higher_than_q3] 

                self.train_indices = self.train_indices + smaller_q1 *5 + between_q1_and_q2*4 + higher_than_q3*3


            elif self.oversampling_method == 'minimum_density_oversampling':
                from itertools import compress
                train_annots = annots[self.train_indices]
                #gdw = GetKDE()
                self.dw_all ={}
                stacked_arrays = []
                for i in range(train_annots.shape[1]):
                    self.dw_cur = f'dw{i}'
                    self.dw_all[self.dw_cur] = GetKDE(alpha=alpha)#DenseWeight(alpha=1)
                    weighted_targs = self.dw_all[self.dw_cur].fit(train_annots[:, i])
                    stacked_arrays.append(weighted_targs)
                

                stacked_weight = np.stack(stacked_arrays, axis=1)

                min_dense = np.min(stacked_weight, axis=1)
                #get average for that data point
                #oversample based on that 
                threshold = min_dense<tolerance #th
                dense_thresh_filter = (min_dense)[threshold]
                dense_thresh_indices = list(compress(self.train_indices, threshold)) #indices of under the threshhold

                self.train_indices = self.train_indices + dense_thresh_indices



        
#################

        test_list = []
        test_list.extend(self.test_indices)
        test_list.extend(self.valid_indices)
        test_list.extend(self.train_indices)
        #not going to be same len because of oversampling
        #assert test_list == indices

    def get_split(self, split_name):
        
        if split_name == 'train':
            # print(self.train_indices)
            # print([self.names[idx] for idx in self.train_indices])
            indices = self.train_indices
        elif split_name == 'valid':
            indices = self.valid_indices
        elif split_name == 'test':
            indices = self.test_indices
        else:
            raise ValueError(f"Unknown split name {split_name}")

        return ([self.feats[idx] for idx in indices],
                [self.annots[idx] for idx in indices],
                [self.targs[idx] for idx in indices],
                [self.names[idx] for idx in indices],
                [self.genres[idx] for idx in indices],
                indices)



def load_data(
        batch_size,
        feat_settings,
        gpu_num,
        hop_size,
        k_samples,
        num_pt_workers,
        seq_len,
        use_audio_in,
        val_size,
        sequential,
        train_y,
        transform = None,
        scale = None,
        mode = 'train',
        oversampling = False,
        oversampling_method = None,
        tolerance = 20,
        os_ratio = 1,
        data_augmentation = False,
        alpha = 1,
        task = 'regression'
    
        
):
    if scale == 'None':
        scale = None
    if scale:
        print(f'Loading Data! Scaling is set to {scale}!')
    else:
        print(f'Loading Data!')
    entries = match_files(train_y = train_y, task = task)
    if data_augmentation in [1,2,4,8] or oversampling_method == 'density_oversampling_augmentation' or oversampling_method == 'density_oversampling_augmentation_V2':
        aug_entries = match_files_augmentation(train_y = train_y, data_augmentation = data_augmentation)
        #use aug_entries seperatly and create train test sets
        #get indices of train and get the augmented data of it


    if transform != None:
        print(f'Transformation is set to {transform}!')
    if use_audio_in:
        feats = [entry.get_audio() for entry in entries]
        if data_augmentation in [1,2,4,8] or oversampling_method == 'density_oversampling_augmentation' or oversampling_method == 'density_oversampling_augmentation_V2':
            aug_feats = [entry.get_audio() for entry in aug_entries]####
            aug_annots = [entry.get_gems_9() for entry in aug_entries]
            aug_targs = aug_annots
            aug_names = [entry.get_name() for entry in aug_entries]
            aug_genres = [entry.get_genre() for entry in aug_entries]
    else:
        feats = [entry.get_audio() for entry in entries]  # TODO calculate spectrogram here.
        raise NotImplemented()

    if oversampling == 'True':
        oversampling = True
    annots = [entry.get_gems_9() for entry in entries]


    ##Scaling
    if scale == 'MaxAbs':
        transformer = MaxAbsScaler().fit(annots)
        annots = np.float32(transformer.transform(annots)) # inverse_transform(X)
        
        #save scaler for inversing
        filename_MaxAbsScaler = Path().resolve()/f"mood_tagger_master_thesis_V2/Scaler/MaxScaler.sav"
        pickle.dump(transformer, open(filename_MaxAbsScaler, 'wb'))
    if scale == 'MinMax':
        transformer = MinMaxScaler().fit(annots)
        annots = np.float32(transformer.transform(annots)) # inverse_transform(X)
        
        #save scaler for inversing
        filename_MinMaxScaler = Path().resolve()/f"mood_tagger_master_thesis_V2/Scaler/MinMaxScaler.sav"
        pickle.dump(transformer, open(filename_MinMaxScaler, 'wb'))

    if scale == 'RobustScaler':
        transformer = RobustScaler().fit(annots)
        annots = np.float32(transformer.transform(annots)) # inverse_transform(X)
        
        #save scaler for inversing
        filename_RobustScaler = Path().resolve()/f"mood_tagger_master_thesis_V2/Scaler/RobustScaler.sav"
        pickle.dump(transformer, open(filename_RobustScaler, 'wb'))

    if scale == 'StandardScaler':
        transformer = StandardScaler().fit(annots)
        annots = np.float32(transformer.transform(annots)) # inverse_transform(X)
        
        #save scaler for inversings
        filename_StandardScaler = Path().resolve()/f"mood_tagger_master_thesis_V2/Scaler/StandardScaler.sav"
        pickle.dump(transformer, open(filename_StandardScaler, 'wb'))


    targs = annots
    names = [entry.get_name() for entry in entries]
    genres = [entry.get_genre() for entry in entries]




    if transform == 'log':
        targs = np.log(np.array(targs)+1)
        annots = np.log(np.array(annots)+1)

    #Labels filter
    gems_label_pos = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
    label_idx = []

    for i in train_y:
        label_idx.append(gems_label_pos.index(i))
    old_targs = targs
    targs = np.take(targs, label_idx, axis = 1) #this worked
    annots = targs

    if data_augmentation in [1,2,4,8] and not (oversampling_method == 'density_oversampling_augmentation' or oversampling_method == 'density_oversampling_augmentation_V2'):
        aug_data = {
            'aug_feats':aug_feats, 
            'aug_annots': aug_annots, 
            'aug_targs' : aug_targs, 
            'aug_names' : aug_names, 
            'aug_genres': aug_genres
        }
        data_aug_info = 'default'
    elif data_augmentation in [1,2,4,8] and (oversampling_method == 'density_oversampling_augmentation' or oversampling_method == 'density_oversampling_augmentation_V2'):
        aug_data = {
            'aug_feats':aug_feats, 
            'aug_annots': aug_annots, 
            'aug_targs' : aug_targs, 
            'aug_names' : aug_names, 
            'aug_genres': aug_genres
        }
        data_aug_info = 'oversampling'
        # aug_partitioner = RandomPartitioner(feats, annots, targs, names, genres, 42, train_y, mode = mode, tolerance = tolerance, oversampling = oversampling, os_ratio = os_ratio, oversampling_method = oversampling_method,
        #                             data_augmentation = data_aug_info, data_aug_data = aug_data , alpha = alpha)
        # aug_train_feats, aug_train_annots, aug_train_targs, aug_train_names, aug_train_genres, aug_train_ix = aug_partitioner.get_split('train')
        data_aug_info = 'None'
    else:
        aug_data = {}
        data_aug_info = False
    partitioner = RandomPartitioner(feats, annots, targs, names, genres, 42, train_y, mode = mode, tolerance = tolerance, oversampling = oversampling, os_ratio = os_ratio, oversampling_method = oversampling_method,
                                    data_augmentation = data_aug_info, data_aug_data = aug_data , alpha = alpha)
    
    # if data_augmentation in [1,2,4]:
    #     aug_partitioner = RandomPartitioner(aug_feats, aug_annots, aug_targs, aug_names, aug_genres, 42, train_y, mode = mode, tolerance = tolerance, oversampling = oversampling, os_ratio = os_ratio, oversampling_method = oversampling_method)

    valid_subset = None
    if val_size is not None and 0.0 < val_size < 1.0:
        valid_subset = val_size
    pin_memory = gpu_num > -1


    train_feats, train_annots, train_targs, train_names, train_genres, train_ix = partitioner.get_split('train')
    #todo ADD Augmentation here prinz p 7
    if oversampling_method == 'density_oversampling_augmentation' or oversampling_method == 'density_oversampling_augmentation_V2':
        index_dict = defaultdict(list)
        for idx, item in enumerate(train_ix):
            index_dict[item].append(idx)

            # Find duplicates and their indices
        testing = {item: indices for item, indices in index_dict.items() if len(indices) > 1}
        duplicates =  {item: indices[1:] for item, indices in index_dict.items() if len(indices) > 1} #
        # get index of duplicate entries and get the corresponding augmented data!
        list_idx = duplicates.keys()
        aug_oversampling_names = [aug_names[i] for i in list_idx]
        aug_oversampling_targs = [aug_targs[i] for i in list_idx]
        aug_oversampling_annots = [aug_annots[i] for i in list_idx]
        aug_oversampling_feats = [aug_feats[i] for i in list_idx]

        #remove oversampled entries from the train set
        to_remove_idx = duplicates.values()
        to_remove_list = []
        for sublist in to_remove_idx:
            to_remove_list.extend(sublist)
        to_remove_list.sort(reverse=True)
        for idx in to_remove_list:
            del train_annots[idx]
            del train_feats[idx]
            del train_names[idx]
            del train_targs[idx]

        train_annots = train_annots + aug_oversampling_annots
        train_feats = train_feats + aug_oversampling_feats
        train_names = train_names + aug_oversampling_names
        train_targs = train_targs + aug_oversampling_targs

        # print(aug_names[356])
        # print(train_names[257])
        # print(train_names[115])
        # print(aug_annots[356])
        # print(train_annots[257])

    #check if duplicate exist in train_ix, get indices of duplicate and test

    frames_to_samples = FramesToSamples(hop_size=feat_settings.hop_size, window_size=feat_settings.frame_size)

    train_set = SnippetDataset(train_feats, train_targs,
                               seq_len,
                               hop_size,
                               single_targets=True,
                               seq_len_dist_fun=frames_to_samples,
                               squeeze=True,
                               )
    total_num_train_samples = len(train_set)
    # train_batch_sampler = InfiniteBatchSampler(data_size=total_num_train_samples, batch_size=batch_size,
    #                                            shuffle=True)
    train_loader = DataLoader(
        train_set, num_workers=num_pt_workers, pin_memory=pin_memory, batch_size=batch_size
    )  # batch_sampler=train_batch_sampler,
    if k_samples is None or k_samples <= 0 or k_samples > total_num_train_samples:
        k_samples = total_num_train_samples
    logging.info(f"Number of total training examples: {total_num_train_samples}.")
    # batches_per_epoch = k_samples // batch_size

    valid_seq_len = seq_len
    valid_batch_size = batch_size

    valid_feats, valid_annots, valid_targs, valid_names, valid_genres, valid_ix = partitioner.get_split('valid')
    valid_loader = DataLoader(
        SnippetDataset(
            valid_feats,
            valid_targs,
            valid_seq_len,
            hop_size,
            single_targets=True,
            seq_len_dist_fun=frames_to_samples,
            squeeze=True,
        ),
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_pt_workers,
        pin_memory=pin_memory,
    )
    test_feats, test_annots, test_targs, test_names, test_genres, test_ix = partitioner.get_split('test')


    test_loader = DataLoader(
        ListDataset(test_feats, test_targs, test_annots, test_names, test_genres),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    #assert if test does not include train samples
    matching_indices = []
    for index, value in enumerate(train_names):
        #if value == 'Tschaikowski 4. Sinfonie':
        #    print(True)
        if value in test_names:
            matching_indices.append(index)
    assert len(matching_indices) == 0

    matching_indices = []
    for index, value in enumerate(train_names):
        #if value == 'Tschaikowski 4. Sinfonie':
        #    print(True)
        if value in valid_names:
            matching_indices.append(index)
    assert len(matching_indices) == 0
    loss_annots = np.array(train_annots)

    return test_loader, train_loader, valid_loader, loss_annots, annots #


def generate_plots():
    entries = match_files()
    annots = [entry.get_gems_9() for entry in entries]
    genres = [entry.get_genre() for entry in entries]

    gems_df = generate_dataframe(annots, genres)
    plot_data(gems_df, 'EMMA')


def generate_dataframe(annots, genres):
    d = []
    for cur_genre, cur_mat in zip(genres, annots):
        for gems_idx in range(9):
            d.append(
                {
                    'Score': cur_mat[gems_idx],
                    'Categories': GEMS_9[gems_idx],
                    'Genre': GENRE_MAP[cur_genre]
                }
            )
    gems_df = pd.DataFrame(d)
    return gems_df


def plot_data(gems_df, set_title='xx'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_path = os.path.join(DATA_PATH, "../../figures/")
    os.makedirs(fig_path, exist_ok=True)

    # overall plot of gems 9 distribution in dataset:
    plt.figure(figsize=(8, 6), dpi=200)
    sns.violinplot(gems_df, x='Categories', y='Score', cut=0)
    plt.xticks(rotation=45)
    plt.ylim([0, 55])
    plt.title(f'GEMS-9 distribution in {set_title}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, set_title.lower().replace(' ', '_') + '_overal_gems9_dist.png'))
    # plt.show()
    # per genre plot of gems 9 distribution
    plt.figure(figsize=(16, 6), dpi=200)
    sns.violinplot(gems_df, x='Categories', y='Score', hue='Genre', cut=0)
    plt.ylim([0, 55])
    plt.title(f'GEMS-9 per Genre in {set_title}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, set_title.lower().replace(' ', '_') + '_genre_gems9_dist.png'))
    # plt.show()



if __name__ == '__main__':
    generate_plots()
