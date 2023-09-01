import pandas as pd
import numpy as np

ANNOT_CSV_FILE = 'mood_tagger_master_thesis/data/GEMS-INN_2023-01-30_expert.csv'


GEMS_9 = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
NUM_CLASSES = len(GEMS_9)
GENRE_MAP = {'H': 'Hip-Hop', 'K': 'Classical', 'P': 'Pop'}

emma_df = pd.read_csv(ANNOT_CSV_FILE, encoding="ISO-8859-1")
emma_df.dropna(inplace=True)




def create_dataset(variation, output_name, tolerance):
    emma_df_new = emma_df
    if variation == "each_mean":
        meanvals = emma_df[GEMS_9].mean()
        print(list(meanvals.values))
        for column in emma_df[GEMS_9]:        
            candidate = []
            index = 0
            for i in emma_df[GEMS_9][column]: #each entry
                #if i > meanvals[column]+tolerance: #if val is over 5 of mean value
                if np.abs(i-meanvals[column]) > tolerance:
                    candidate.append(index) #add index
                index += 1 #next index
            emma_df_new = pd.concat([emma_df_new,emma_df.iloc[candidate]])


    emma_df_new.to_csv(f'mood_tagger_master_thesis_V2/data/GEMS-INN_2023-01-30_expert_{output_name}_{tolerance}.csv', index=False) 
                    
create_dataset("each_mean", 'each_mean', 5)
