import yaml
from train import run_training

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="default")

def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    run_training(cfg, GEMS_9= cfg.datasets.labels)



# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#     documents = yaml.full_load(file)

#     for item, doc in documents.items():
#         print(item, ":", doc)
#     print(documents['datasets']['labels'])
#     gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
#     #for i in gems:
#     #    documents['datasets']['labels'] = i
#     for i in gems:
#         documents['datasets']['labels'] = [i]
#         with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
#             yaml.dump(documents, file)

#         my_app()
    



def run():

    values = [0.5,1,1.5,2,3,4]
    oversamples = [True, False]
    for o in oversamples:

        for k in values:
            
            
            with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
                documents = yaml.full_load(file)

                for item, doc in documents.items():
                    print(item, ":", doc)
                print(documents['datasets']['labels'])
                gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
                #for i in gems:
                #    documents['datasets']['labels'] = i
                documents['datasets']['dense_weight_alpha'] = k
                documents['datasets']['oversampling'] = o
                documents['datasets']['loss_func'] = 'dense_weight'
                for i in gems:
                    documents['datasets']['labels'] = [i]
                    with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
                        yaml.dump(documents, file)

                    my_app()


# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#     documents = yaml.full_load(file)
#     documents['datasets']['oversampling'] = True
#     documents['datasets']['loss_func'] = 'MAE'
#     for item, doc in documents.items():
#         print(item, ":", doc)
#     print(documents['datasets']['labels'])
#     gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
#         #for i in gems:
#         #    documents['datasets']['labels'] = i
#     for i in gems:
#         documents['datasets']['labels'] = [i]
#         with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
#             yaml.dump(documents, file)

#         my_app()






# ########ALL MODEL
# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#     documents = yaml.full_load(file)
#     documents['datasets']['oversampling'] = False
#     documents['datasets']['oversampling_tolerance'] = 0.5
#     documents['datasets']['oversampling_ratio'] = 2
#     documents['datasets']['oversampling_method'] = 'None'
#     documents['datasets']['dense_weight_alpha'] = 2
#     documents['training']['batch_size'] = 50
#     documents['datasets']['loss_func'] = 'MAE'
#     documents['model']['architecture'] = 'allconv_01'
#     for item, doc in documents.items():
#         print(item, ":", doc)
#     print(documents['datasets']['labels'])
#     gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
#             #for i in gems:
#             #    documents['datasets']['labels'] = i

#     documents['datasets']['labels'] = gems
#     with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
#         yaml.dump(documents, file)

#     my_app()



# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#     documents = yaml.full_load(file)
#     documents['datasets']['oversampling'] = False
#     documents['datasets']['oversampling_tolerance'] = 0.5
#     documents['datasets']['oversampling_ratio'] = 2
#     documents['datasets']['oversampling_method'] = 'None'
#     documents['datasets']['dense_weight_alpha'] = 2

#     documents['datasets']['loss_func'] = 'MAE'
#     documents['model']['architecture'] = 'multitask_model'
#     for item, doc in documents.items():
#         print(item, ":", doc)
#     print(documents['datasets']['labels'])
#     gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
#             #for i in gems:
#             #    documents['datasets']['labels'] = i

#     documents['datasets']['labels'] = gems
#     with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
#         yaml.dump(documents, file)

#     my_app()




# for i in [1,2,3,4]:

#     with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#         documents = yaml.full_load(file)
#         documents['datasets']['oversampling'] = False
#         documents['datasets']['oversampling_tolerance'] = 0.5
#         documents['datasets']['oversampling_ratio'] = 2
#         documents['datasets']['oversampling_method'] = 'None'
#         documents['datasets']['dense_weight_alpha'] = i

#         documents['datasets']['loss_func'] = 'dense_weight'
#         documents['model']['architecture'] = 'multitask_model'
#         for item, doc in documents.items():
#             print(item, ":", doc)
#         print(documents['datasets']['labels'])
#         gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
#                 #for i in gems:
#                 #    documents['datasets']['labels'] = i

#         documents['datasets']['labels'] = gems
#         with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
#             yaml.dump(documents, file)

#         my_app()

#     with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#         documents = yaml.full_load(file)
#         documents['datasets']['oversampling'] = False
#         documents['datasets']['oversampling_tolerance'] = 0.5
#         documents['datasets']['oversampling_ratio'] = 2
#         documents['datasets']['oversampling_method'] = 'None'
#         documents['datasets']['dense_weight_alpha'] = i

#         documents['datasets']['loss_func'] = 'dense_weight'
#         documents['model']['architecture'] = 'allconv_01'
#         for item, doc in documents.items():
#             print(item, ":", doc)
#         print(documents['datasets']['labels'])
#         gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
#             #for i in gems:
#             #    documents['datasets']['labels'] = i

#         documents['datasets']['labels'] = gems
#         with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
#             yaml.dump(documents, file)

#         my_app()



# ####################SINGLE MODELS
# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#     documents = yaml.full_load(file)
#     documents['datasets']['oversampling'] = False
#     documents['datasets']['oversampling_tolerance'] = 0.5
#     documents['datasets']['oversampling_ratio'] = 2
#     documents['datasets']['oversampling_method'] = 'adaptive_density_oversampling_V2'
#     documents['datasets']['dense_weight_alpha'] = 2

#     documents['datasets']['loss_func'] = 'MAE'
#     documents['model']['architecture'] = 'allconv_01'
#     for item, doc in documents.items():
#         print(item, ":", doc)
#     print(documents['datasets']['labels'])
#     gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
#         #for i in gems:
#         #    documents['datasets']['labels'] = i
#     for i in gems:
#         documents['datasets']['labels'] = [i]
#         with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
#             yaml.dump(documents, file)

#         my_app()
# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#     documents = yaml.full_load(file)
#     documents['datasets']['oversampling'] = False
#     documents['datasets']['oversampling_tolerance'] = 0.5
#     documents['datasets']['oversampling_ratio'] = 2
#     documents['datasets']['oversampling_method'] = 'adaptive_density_oversampling_V2'
#     documents['datasets']['dense_weight_alpha'] = 2

#     documents['datasets']['loss_func'] = 'dense weight'
#     documents['model']['architecture'] = 'allconv_01'
#     for item, doc in documents.items():
#         print(item, ":", doc)
#     print(documents['datasets']['labels'])
#     gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
#         #for i in gems:
#         #    documents['datasets']['labels'] = i
#     for i in gems:
#         documents['datasets']['labels'] = [i]
#         with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
#             yaml.dump(documents, file)

#         my_app()




for i in [1,2,4]:
    with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
        documents = yaml.full_load(file)
        documents['datasets']['oversampling'] = False


        documents['datasets']['loss_func'] = 'dense_weight_tuned'
        documents['datasets']['dense_weight_alpha'] = 2
        documents['datasets']['data_augmentation'] = i
        documents['model']['architecture'] = 'FCN_4_layer'
        for item, doc in documents.items():
            print(item, ":", doc)
        print(documents['datasets']['labels'])
        gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
        documents['datasets']['labels'] = gems
        with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
            yaml.dump(documents, file)

            my_app()
for i in [1,2,4]:
    with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
        documents = yaml.full_load(file)
        documents['datasets']['oversampling'] = False


        documents['datasets']['loss_func'] = 'dense_weight_tuned'
        documents['datasets']['dense_weight_alpha'] = 2
        documents['datasets']['data_augmentation'] = i
        documents['model']['architecture'] = 'FCN_5_layer'
        for item, doc in documents.items():
            print(item, ":", doc)
        print(documents['datasets']['labels'])
        gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']


        documents['datasets']['labels'] = gems
        with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
            yaml.dump(documents, file)

            my_app()

for i in [1,2,4]:
    with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
        documents = yaml.full_load(file)
        documents['datasets']['oversampling'] = False


        documents['datasets']['loss_func'] = 'dense_weight_tuned'
        documents['datasets']['dense_weight_alpha'] = 2
        documents['datasets']['data_augmentation'] = i
        documents['model']['architecture'] = 'FCN_6_layer'
        for item, doc in documents.items():
            print(item, ":", doc)
        print(documents['datasets']['labels'])
        gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']


        documents['datasets']['labels'] = gems
        with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
            yaml.dump(documents, file)

            my_app()


for i in [1,2,4]:
    with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
        documents = yaml.full_load(file)
        documents['datasets']['oversampling'] = False


        documents['datasets']['loss_func'] = 'dense_weight'
        documents['datasets']['dense_weight_alpha'] = 1.5
        documents['datasets']['data_augmentation'] = i
        documents['model']['architecture'] = 'FCN_4_layer'
        for item, doc in documents.items():
            print(item, ":", doc)
        print(documents['datasets']['labels'])
        gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
        documents['datasets']['labels'] = gems
        with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
            yaml.dump(documents, file)

            my_app()
for i in [1,2,4]:
    with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
        documents = yaml.full_load(file)
        documents['datasets']['oversampling'] = False


        documents['datasets']['loss_func'] = 'dense_weight'
        documents['datasets']['dense_weight_alpha'] = 1.5
        documents['datasets']['data_augmentation'] = i
        documents['model']['architecture'] = 'FCN_5_layer'
        for item, doc in documents.items():
            print(item, ":", doc)
        print(documents['datasets']['labels'])
        gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']


        documents['datasets']['labels'] = gems
        with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
            yaml.dump(documents, file)

            my_app()

for i in [1,2,4]:
    with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
        documents = yaml.full_load(file)
        documents['datasets']['oversampling'] = False


        documents['datasets']['loss_func'] = 'dense_weight'
        documents['datasets']['dense_weight_alpha'] = 1.5
        documents['datasets']['data_augmentation'] = i
        documents['model']['architecture'] = 'FCN_6_layer'
        for item, doc in documents.items():
            print(item, ":", doc)
        print(documents['datasets']['labels'])
        gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']


        documents['datasets']['labels'] = gems
        with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
            yaml.dump(documents, file)

            my_app()