import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_binary(ANNOT_CSV_FILE = 'GEMS-INN_2023-01-30_expert.csv', plot = False):
        
    

    GEMS_9 = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
    NUM_CLASSES = len(GEMS_9)
    GENRE_MAP = {'H': 'Hip-Hop', 'K': 'Classical', 'P': 'Pop'}
    GEMS_9_ext = ['artist', 'title', 'Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
    emma_df = pd.read_csv(ANNOT_CSV_FILE, encoding="ISO-8859-1")
    emma_df.dropna(inplace=True)
    emma_df = emma_df[GEMS_9_ext]
    emma_df_binary = emma_df[['artist', 'title']]
    # Highly skewed distribution: If the skewness value is less than −1 or greater than +1.

    # Moderately skewed distribution: If the skewness value is between −1 and −½ or between +½ and +1.
    # Approximately symmetric distribution: If the skewness value is between −½ and +½.
    skewnewss = emma_df[GEMS_9].skew()
    skew_thresh = [10, 20 , 25]
    skew_list = []
    for value in skewnewss:
        if abs(value) > 1:
            skew_list.append(skew_thresh[0])
        elif (abs(value) < 1 and abs(value) > 0.5):
            skew_list.append(skew_thresh[1])
        else:
            skew_list.append(skew_thresh[2])
    i = 0 
    for col in emma_df.columns:
        
        if col not in ['artist', 'title']:
            #could be more elegant
            threshold = np.percentile(emma_df[col], skew_list[i])
            print(f'Emotion {col} will use {skew_list[i]}th percentile as threshold')
            i += 1
            #print(threshold)
            emma_df_binary[col] = (emma_df[col] > threshold).astype(int)
            #test = emma_df[col]
            #print(test[(emma_df[col] < threshold)])

    # Show the updated DataFrame with binary labels
    #print(emma_df_binary.head())
    
    if plot:
        for col in emma_df_binary.columns:
            if col not in ['artist', 'title']:
                plt.figure(figsize=(6, 4))
                emma_df_binary[col].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
                plt.title(f'Distribution of {col}')
                plt.xlabel('Binary Label')
                plt.ylabel('Frequency')
                plt.xticks(rotation=0)
                plt.show()

    # plt.figure(figsize=(10, 6))
    # emma_df_only= emma_df[GEMS_9]
    # for col in emma_df_only.columns:
    #     plt.figure(figsize=(8, 5))
    #     colors = np.where(emma_df_only[col] < emma_df_only[col].quantile(0.25), 'salmon', 'skyblue')
    #     plt.bar(np.arange(len(emma_df_only[col])), emma_df_only[col], color=colors)
    #     plt.axhline(y=emma_df_only[col].quantile(0.25), color='red', linestyle='--', label=f'{col} 25th percentile')
    #     plt.title(f'Distribution of {col} with Values Below 25th Percentile Highlighted')
    #     plt.xlabel('Data Points')
    #     plt.ylabel('Scores')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    emma_df_binary.to_csv('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/data/GEMS-INN_2023-01-30_expert_binary.csv', index=False)
if __name__ == '__main__':
    create_binary()



