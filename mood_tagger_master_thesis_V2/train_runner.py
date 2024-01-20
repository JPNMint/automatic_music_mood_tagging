import yaml
from train import run_training
from train_cv import my_app_fold, run_training_fold


import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from train_embeddings import run_training_transfer_learning
@hydra.main(version_base=None, config_path="configs", config_name="default")

def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    run_training_fold(cfg, GEMS_9= cfg.datasets.labels, fold = 10, stratified= True)
    #run_training(cfg, GEMS_9= cfg.datasets.labels)


def my_app_strat(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    run_training_fold(cfg, GEMS_9= cfg.datasets.labels, fold = 10, stratified= True)
    #run_training(cfg, GEMS_9= cfg.datasets.labels)



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
    



def run(loss_func = 'dense_weight', model = 'FCN_6_layer', oversamples = [False], tolerance =0.7,method='adaptive_density_oversampling', os_ratio = 1, gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness'] ):

    values = [1]
    for o in oversamples:

        for k in values:
            
            
            with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
                documents = yaml.full_load(file)
                if model == 'short_chunk_CNN':
                    documents['training']['batch_size'] = 26

                    documents['model']['architecture'] = 'short_chunk_CNN'
                    documents['training']['sequence_hop'] = 4
                    documents['training']['sequence_length'] = 18
                else:
                    documents['training']['batch_size'] = 50

                    documents['model']['architecture'] = model
                    documents['training']['sequence_hop'] = 40
                    documents['training']['sequence_length'] = 150

                for item, doc in documents.items():
                    print(item, ":", doc)
                print(documents['datasets']['labels'])

                #for i in gems:
                #    documents['datasets']['labels'] = i
                documents['datasets']['dense_weight_alpha'] = k
                documents['datasets']['oversampling'] = o
                documents['datasets']['loss_func'] = loss_func
                documents['datasets']['oversampling_tolerance'] = tolerance
                documents['datasets']['oversampling_method'] = method
                documents['datasets']['oversampling_ratio'] = os_ratio
                
                for i in gems:
                    documents['datasets']['labels'] = [i]
                    with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
                        yaml.dump(documents, file)

                    my_app()


# run('MSE',oversamples = [True],tolerance=0.7, method = 'density_oversampling_V2')



# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#     documents = yaml.full_load(file)
#     documents['datasets']['oversampling'] = True
#     documents['datasets']['loss_func'] = 'MAE'
#     documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation'
#     documents['datasets']['data_augmentation'] = 8
#     documents['datasets']['oversampling_tolerance'] = 0.6
#     documents['model']['architecture'] = 'FCN_6_layer'
#     for item, doc in documents.items():
#         print(item, ":", doc)
#     print(documents['datasets']['labels'])
#     gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
#         #for i in gems:
#         #    documents['datasets']['labels'] = i
#     for i in gems:
#         documents['datasets']['labels'] = gems
#         with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
#             yaml.dump(documents, file)

#         my_app()
#




########################################################
#base
        
# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#         documents = yaml.full_load(file)
#         documents['datasets']['oversampling'] = False
#         documents['datasets']['loss_func'] = 'MAE'
#         documents['datasets']['oversampling_method'] = 'None'
#         documents['datasets']['data_augmentation'] = 0
#         documents['datasets']['oversampling_tolerance'] = 0
#         documents['training']['patience'] = 100

#         documents['model']['architecture'] = 'FCN_6_layer'
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
# for i in [1,2,4,8]:
#     with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#             documents = yaml.full_load(file)
#             documents['datasets']['oversampling'] = False
#             documents['datasets']['loss_func'] = 'MAE'
#             documents['datasets']['oversampling_method'] = 'None'
#             documents['datasets']['data_augmentation'] = i
#             documents['datasets']['oversampling_tolerance'] = 0
#             documents['training']['patience'] = 100

#             documents['model']['architecture'] = 'FCN_6_layer'
#             for item, doc in documents.items():
#                 print(item, ":", doc)
#             print(documents['datasets']['labels'])
#             gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
#                 #for i in gems:
#                 #    documents['datasets']['labels'] = i

#             documents['datasets']['labels'] = gems
#             with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
#                 yaml.dump(documents, file)

#             my_app()
# ###DOMR
        
for i in [0.5,0.6,0.7]:
    with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
            documents = yaml.full_load(file)
            documents['datasets']['oversampling'] = True
            documents['datasets']['loss_func'] = 'MAE'
            documents['datasets']['oversampling_method'] = 'average_density_oversampling'
            documents['datasets']['data_augmentation'] = 0
            documents['datasets']['oversampling_tolerance'] = i
            documents['training']['patience'] = 100

            documents['model']['architecture'] = 'FCN_6_layer'
            for item, doc in documents.items():
                print(item, ":", doc)
            print(documents['datasets']['labels'])
            gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
                #for i in gems:
                #    documents['datasets']['labels'] = i

            documents['datasets']['labels'] = gems
            with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
                yaml.dump(documents, file)

            my_app()



# # ###FCN6 DOMR percentile
for i in [0.5,0.6,0.7]:
    with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
            documents = yaml.full_load(file)
            documents['datasets']['oversampling'] = True
            documents['datasets']['loss_func'] = 'MAE'
            documents['datasets']['oversampling_method'] = 'average_density_oversampling_V2'
            documents['datasets']['data_augmentation'] = 0
            documents['datasets']['oversampling_tolerance'] = i
            documents['training']['patience'] = 100

            documents['model']['architecture'] = 'FCN_6_layer'
            for item, doc in documents.items():
                print(item, ":", doc)
            print(documents['datasets']['labels'])
            gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
                #for i in gems:
                #    documents['datasets']['labels'] = i

            documents['datasets']['labels'] = gems
            with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
                yaml.dump(documents, file)

            my_app()


# #FCN6 DOMR+  0.6 aug 2, 4

for tol in [0.5,0.6,0.7]:
    for aug in [1, 2,4]:
        with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
                documents = yaml.full_load(file)
                documents['datasets']['oversampling'] = True
                documents['datasets']['loss_func'] = 'MAE'
                documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation'
                documents['datasets']['data_augmentation'] = aug
                documents['datasets']['oversampling_tolerance'] = tol
                documents['training']['patience'] = 100

                documents['model']['architecture'] = 'FCN_6_layer'
                for item, doc in documents.items():
                    print(item, ":", doc)
                print(documents['datasets']['labels'])
                gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
                    #for i in gems:
                    #    documents['datasets']['labels'] = i

                documents['datasets']['labels'] = gems
                with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
                    yaml.dump(documents, file)

                my_app()
# for tol in [0.7]:
#     for aug in [1,2,4]:
#         with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#                 documents = yaml.full_load(file)
#                 documents['datasets']['oversampling'] = True
#                 documents['datasets']['loss_func'] = 'MAE'
#                 documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation'
#                 documents['datasets']['data_augmentation'] = aug
#                 documents['datasets']['oversampling_tolerance'] = tol
#                 documents['training']['patience'] = 100

#                 documents['model']['architecture'] = 'FCN_6_layer'
#                 for item, doc in documents.items():
#                     print(item, ":", doc)
#                 print(documents['datasets']['labels'])
#                 gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
#                     #for i in gems:
#                     #    documents['datasets']['labels'] = i

#                 documents['datasets']['labels'] = gems
#                 with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
#                     yaml.dump(documents, file)

                # my_app()

# ############### DMR+ percentile
        
for tol in [0.6, 0.7, 0.8]:
    for aug in [1,2,4]:
        with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
                documents = yaml.full_load(file)
                documents['datasets']['oversampling'] = True
                documents['datasets']['loss_func'] = 'MAE'
                documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation_V2'
                documents['datasets']['data_augmentation'] = aug
                documents['datasets']['oversampling_tolerance'] = tol
                documents['training']['patience'] = 100

                documents['model']['architecture'] = 'FCN_6_layer'
                for item, doc in documents.items():
                    print(item, ":", doc)
                print(documents['datasets']['labels'])
                gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']

                documents['datasets']['labels'] = gems
                with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
                    yaml.dump(documents, file)

                my_app()

# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#         documents = yaml.full_load(file)
#         documents['datasets']['oversampling'] = True
#         documents['datasets']['loss_func'] = 'MAE'
#         documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation_V2'
#         documents['datasets']['data_augmentation'] = 1
#         documents['datasets']['oversampling_tolerance'] = 0.7
#         documents['training']['patience'] = 100

#         documents['model']['architecture'] = 'FCN_6_layer'
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

# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#         documents = yaml.full_load(file)
#         documents['datasets']['oversampling'] = True
#         documents['datasets']['loss_func'] = 'MAE'
#         documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation_V2'
#         documents['datasets']['data_augmentation'] = 1
#         documents['datasets']['oversampling_tolerance'] = 0.8
#         documents['training']['patience'] = 100

#         documents['model']['architecture'] = 'FCN_6_layer'
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


# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#         documents = yaml.full_load(file)
#         documents['datasets']['oversampling'] = True
#         documents['datasets']['loss_func'] = 'MAE'
#         documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation_V2'
#         documents['datasets']['data_augmentation'] = 2
#         documents['datasets']['oversampling_tolerance'] = 0.6
#         documents['training']['patience'] = 100

#         documents['model']['architecture'] = 'FCN_6_layer'
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

# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#         documents = yaml.full_load(file)
#         documents['datasets']['oversampling'] = True
#         documents['datasets']['loss_func'] = 'MAE'
#         documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation_V2'
#         documents['datasets']['data_augmentation'] = 2
#         documents['datasets']['oversampling_tolerance'] = 0.7
#         documents['training']['patience'] = 100

#         documents['model']['architecture'] = 'FCN_6_layer'
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

# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#         documents = yaml.full_load(file)
#         documents['datasets']['oversampling'] = True
#         documents['datasets']['loss_func'] = 'MAE'
#         documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation_V2'
#         documents['datasets']['data_augmentation'] = 2
#         documents['datasets']['oversampling_tolerance'] = 0.8
#         documents['training']['patience'] = 100

#         documents['model']['architecture'] = 'FCN_6_layer'
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

# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#         documents = yaml.full_load(file)
#         documents['datasets']['oversampling'] = True
#         documents['datasets']['loss_func'] = 'MAE'
#         documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation_V2'
#         documents['datasets']['data_augmentation'] = 4
#         documents['datasets']['oversampling_tolerance'] = 0.6
#         documents['training']['patience'] = 100

#         documents['model']['architecture'] = 'FCN_6_layer'
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

# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#         documents = yaml.full_load(file)
#         documents['datasets']['oversampling'] = True
#         documents['datasets']['loss_func'] = 'MAE'
#         documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation_V2'
#         documents['datasets']['data_augmentation'] = 4
#         documents['datasets']['oversampling_tolerance'] = 0.7
#         documents['training']['patience'] = 100

#         documents['model']['architecture'] = 'FCN_6_layer'
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

# with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
#         documents = yaml.full_load(file)
#         documents['datasets']['oversampling'] = True
#         documents['datasets']['loss_func'] = 'MAE'
#         documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation_V2'
#         documents['datasets']['data_augmentation'] = 4
#         documents['datasets']['oversampling_tolerance'] = 0.8
#         documents['training']['patience'] = 100

#         documents['model']['architecture'] = 'FCN_6_layer'
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
# # for i in [0.3,0.4,0.5, 0.6, 0.7]:


# #     with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
# #             documents = yaml.full_load(file)
# #             documents['datasets']['oversampling'] = True
# #             documents['datasets']['loss_func'] = 'MAE'
# #             documents['datasets']['oversampling_method'] = 'average_density_oversampling_V2'
# #             documents['datasets']['data_augmentation'] = 0
# #             documents['datasets']['oversampling_tolerance'] = i
# #             documents['datasets']['dense_weight_alpha'] = 0
# #             documents['training']['patience'] = 100

# #             documents['model']['architecture'] = 'FCN_6_layer'
# #             for item, doc in documents.items():
# #                 print(item, ":", doc)
# #             print(documents['datasets']['labels'])
# #             gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
# #                 #for i in gems:
# #                 #    documents['datasets']['labels'] = i
# #             documents['datasets']['labels'] = gems
# #             with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
# #                 yaml.dump(documents, file)

# #             my_app()
    
# # for i in [0.5,0.6,0.7, 0.4,0.3]:
    
# #     with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
# #         documents = yaml.full_load(file)
# #         documents['datasets']['oversampling'] = True
# #         documents['datasets']['loss_func'] = 'MAE'
# #         documents['datasets']['oversampling_method'] = 'density_oversampling_augmentation'
# #         documents['datasets']['data_augmentation'] = 2
# #         documents['datasets']['oversampling_tolerance'] = i
# #         documents['model']['architecture'] = 'FCN_6_layer'
# #         for item, doc in documents.items():
# #             print(item, ":", doc)
# #         print(documents['datasets']['labels'])
# #         gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
# #             #for i in gems:
# #             #    documents['datasets']['labels'] = i
# #         for i in gems:
# #             documents['datasets']['labels'] = gems
# #             with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
# #                 yaml.dump(documents, file)

# #             my_app()

# # for i in [0.2,0.3,0.1]:
# #     with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
# #         documents = yaml.full_load(file)
# #         documents['datasets']['oversampling'] = True
# #         documents['datasets']['loss_func'] = 'MAE'
# #         documents['model']['architecture'] = 'FCN_6_layer'
# #         documents['datasets']['oversampling_method'] = 'minimum_density_oversampling'
# #         documents['datasets']['oversampling_tolerance'] = i

        
# #         for item, doc in documents.items():
# #             print(item, ":", doc)
# #         print(documents['datasets']['labels'])
# #         gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
# #             #for i in gems:
# #             #    documents['datasets']['labels'] = i

# #         documents['datasets']['labels'] = gems
# #         with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
# #             yaml.dump(documents, file)

# #         my_app()

# # for i in [0.6,0.7,0.8]:
# #     with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
# #         documents = yaml.full_load(file)
# #         documents['datasets']['oversampling'] = True
# #         documents['datasets']['loss_func'] = 'MAE'
# #         documents['model']['architecture'] = 'FCN_6_layer'
# #         documents['datasets']['oversampling_method'] = 'average_density_oversampling_V2'
# #         documents['datasets']['oversampling_tolerance'] = i

        
# #         for item, doc in documents.items():
# #             print(item, ":", doc)
# #         print(documents['datasets']['labels'])
# #         gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
# #             #for i in gems:
# #             #    documents['datasets']['labels'] = i

# #         documents['datasets']['labels'] = gems
# #         with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
# #             yaml.dump(documents, file)

# #         my_app()


# #     with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
# #         documents = yaml.full_load(file)
# #         documents['datasets']['oversampling'] = True
# #         documents['datasets']['loss_func'] = 'MAE'
# #         documents['model']['architecture'] = 'FCN_6_layer'
# #         documents['datasets']['oversampling_method'] = 'average_density_oversampling'
# #         documents['datasets']['oversampling_tolerance'] = i

        
# #         for item, doc in documents.items():
# #             print(item, ":", doc)
# #         print(documents['datasets']['labels'])
# #         gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
# #             #for i in gems:
# #             #    documents['datasets']['labels'] = i

# #         documents['datasets']['labels'] = gems
# #         with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
# #             yaml.dump(documents, file)

# #         my_app()

# # with open(r"/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'r') as file:
# #     documents = yaml.full_load(file)
# #     documents['datasets']['oversampling'] = True
# #     documents['datasets']['loss_func'] = 'MAE'
# #     documents['datasets']['dense_weight_alpha'] = 0
# #     documents['model']['architecture'] = 'FCN_6_layer'
# #     documents['oversampling_method'] = 'average_density_oversampling_V2'
# #     documents['oversampling_tolerance'] = 0.3

    
# #     for item, doc in documents.items():
# #         print(item, ":", doc)
# #     print(documents['datasets']['labels'])
# #     gems = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
# #         #for i in gems:
# #         #    documents['datasets']['labels'] = i

# #     documents['datasets']['labels'] = gems
# #     with open("/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/configs/default.yaml", 'w') as file:
# #         yaml.dump(documents, file)

# #     my_app()

# # run('MSE',oversamples = [True],tolerance=0.5, method = 'density_oversampling_V3')
# # run('MSE',oversamples = [True],tolerance=0.5, method = 'density_oversampling_V2')

# # run('MAE',oversamples = [True],tolerance=0.7, method = 'density_oversampling_V3')
# # run('MAE',oversamples = [True],tolerance=0.7, method = 'density_oversampling_V2')
# # run('MAE',oversamples = [True],tolerance=0.5, method = 'density_oversampling_V3')
# # run('MAE',oversamples = [True],tolerance=0.5, method = 'density_oversampling_V2')

