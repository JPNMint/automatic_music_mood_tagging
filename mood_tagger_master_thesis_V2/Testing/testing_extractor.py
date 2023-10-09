from musicnn_pons.extractor import extractor

file_name = '/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/data/audio/H_2Pac_AllEyez.mp3'
#'/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/musicnn_pons/joram-moments_of_clarity-08-solipsism-59-88.mp3'
taggram, tags, features = extractor(file_name, model='MTT_musicnn', extract_features=True)





print(list(features.keys()))

print(features['penultimate'])