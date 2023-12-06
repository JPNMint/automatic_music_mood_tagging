from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Mp3Compression, LowPassFilter, AddBackgroundNoise, Gain, GainTransition, Limiter
import numpy as np
import librosa
import pathlib as Path
import random

def create_augmentation(transformations):
    strength = 1


    bgPath = '/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/data/Background noise'
    bg_folder = list(Path.Path(bgPath).glob('*.wav'))
    aug_cache_path = f'/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/data/cache_augmented_{transformations}'
    audio_path = '/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/data/audio'
  
    print(f'Initiatin transformation!')
    print(f'Audio path is {audio_path}!')
    print(f'Cache path is {aug_cache_path}!')


    #cache_path = '/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/data/cache'
    audio_folder = list(Path.Path(audio_path).glob('*.mp3'))
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(p=0.5),
    ])
    if transformations not in [1, 2, 4, 7]:
        raise ValueError("You can only select 1, 2 or 4 transformation chains!")
    randombg = random.choices(bg_folder)
    mp3_strength = {0.5:56, 1:40, 1.5:32, 2:24}
    bg_strength = {0.5:27, 1:18, 1.5:13, 2:10}
    available_transformations = {
         #implement equasition if result is bad
        'time_stretch_pitch_shift' : pitch_time(strength),
        # 'pitch_shift' : PitchShift(min_semitones=-4, max_semitones=4, p=1),
        'mp3compression' : Mp3Compression(min_bitrate=mp3_strength[strength], max_bitrate= mp3_strength[strength] ,backend= 'lameenc', p=1),
        'lowpass' : LowPassFilter(min_cutoff_freq=20,max_cutoff_freq=150,min_rolloff = 6*strength, max_rolloff=6*strength, p=1),
        'bgnoise' : AddBackgroundNoise(randombg,min_snr_db=bg_strength[strength],max_snr_db=bg_strength[strength], p=1),
        'timeshift' : Shift(min_shift=0.05*strength,max_shift=0.05*strength, shift_unit = 'seconds', p=1.0),
        'gain' : GainTransition(min_gain_db=-4*strength,    max_gain_db=4*strength,    p=1.0),
        'limiter' : Limiter(min_threshold_db=-60.0,    max_threshold_db=-6.0,    threshold_mode="relative_to_signal_peak",    max_release = 0.1,    p=1.0),

    }
    


    print(f'Number of transformations is {transformations}!')
    random.seed(2425)
    for i in audio_folder:
        song, sr = librosa.load(i, sr=44100)
        name = i.stem
        print(f'Transforming {name}!')
        filepath = f'{aug_cache_path}/{name.lower()}'

        choices = []
        
    
        keys = list(available_transformations.keys())
        
        for i in range(transformations):
                #random.shuffle(bg_folder)
                randombg = random.choice(bg_folder)
                #keys = random.shuffle(keys)
                random_choice = random.choice(keys)
                choices.append(random_choice)
                keys.remove(random_choice)
        print(f'Chosen transformation is {choices}!')

        augment_funcs = []
        for i in choices:
            if i == 'time_stretch_pitch_shift':
                aug_dict = available_transformations[i]
                for k , val in aug_dict.items():
                    augment_funcs.append(val)
            else:     
                augment_funcs.append(available_transformations[i])

        augment = Compose(augment_funcs)

        augmented_samples = augment(samples=song, sample_rate=sr)
        
        print('Transformed!')
        np.save(filepath, augmented_samples)
        import soundfile as sf

def pitch_time(strength=1):
    
    st = round(random.uniform(0, 1), 2)
    sp = (np.sqrt(strength**2)-np.sqrt(st/4) )/ np.sqrt(1/5)
    print(f'time stretch {1-sp/2}')
    print(f'pitch shift {1-st/2}')
    transformation = {
        'time_stretch' : TimeStretch(min_rate=1-st/2, max_rate=1+st/2, p=1),
        'pitch_shift' : PitchShift(min_semitones=-sp/2, max_semitones=sp/2, p=1),
    }
    return transformation


if __name__ == '__main__':
    for i in [1,2,4,7]:
        create_augmentation(i)



