import os
import typing

import numpy as np
import torch
import importlib.util
from sklearn.metrics import auc, precision_recall_fscore_support, roc_auc_score, average_precision_score
from pathlib import Path
import pickle
from itertools import chain
from data import GEMS_9, plot_data, generate_dataframe

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
ARCH_PATH = os.path.join(ROOT_PATH, 'architectures')


def get_architecture(architecture_name: str, from_path: str = None) -> typing.Tuple[typing.Type, str]:
    """ select architecture """
    arch_file_name = architecture_name + '.py'

    if from_path is None:
        architecture_file = os.path.join(ARCH_PATH, arch_file_name)
        namespace = dict()
        namespace['architecture'] = None
        exec(f'from architectures.{architecture_name} import Net as architecture', namespace)
        return namespace['architecture'], architecture_file
    else:
        architecture_file = os.path.join(from_path, arch_file_name)
        arch_spec = importlib.util.spec_from_file_location(architecture_name, architecture_file)
        arch_module = importlib.util.module_from_spec(arch_spec)
        arch_spec.loader.exec_module(arch_module)
        return arch_module.Net, architecture_file
    


def calcAUC(predictions, annotations, n_classes):
    # Calculate average AUC-ROC across all classes
    #n_classes = predictions.shape[1]
    np_predictions = np.array(predictions)
    np_annotations= np.array(annotations)
    auc_scores = []
    for i in range(n_classes):
        for l in range(len(np_predictions)):
            #hier ist der fehler mach neues array und dann roc auc score drauf
            auc = roc_auc_score(np_annotations[l][i],np_predictions[l][i])
        auc = roc_auc_score(np_annotations[:, i], np_predictions[:, i])
        auc_scores.append(auc)

    # Calculate the average AUC-ROC
    avg_auc = np.mean(auc_scores)

    print("AUC-ROC scores for each class:", auc_scores)
    print("Average AUC-ROC:", avg_auc)

    return roc_aucs, pr_aucs


def test_model(model, num_classes, test_loader, device, csv_information, plot=False, model_name=None, transform = None, training = 'test', scale = None , train_y = None):
    if transform == 'log':
        print('Output data is log transform, applying np.exp to output.')
    if scale == 'None':
        scale = None
    if scale:
        print(f'Testing! Scaling is set to {scale}!')
    num_examples = len(test_loader)
    mean_abs_errors = np.zeros((num_classes,))
    mean_errors = np.zeros((num_classes,))
    mean_squared_errors = np.zeros((num_classes,))
    maximums = np.zeros((num_classes,))
    minimums = np.ones((num_classes,)) * 100
    model.to(device)
    predictions = []
    prediction_names = []
    prediction_genres = []
    annotations = []
    actual = []
    if scale == 'MaxAbs':
        filename_MaxAbsScaler = Path().resolve()/f"mood_tagger_master_thesis_V2/Scaler/MaxScaler.sav"
        loaded_scaler = pickle.load(open(filename_MaxAbsScaler, 'rb'))
    if scale == 'MinMax':
        filename_MinMaxScaler = Path().resolve()/f"mood_tagger_master_thesis_V2/Scaler/MinMaxScaler.sav"
        loaded_scaler = pickle.load(open(filename_MinMaxScaler, 'rb'))

    if scale == 'RobustScaler':
        filename_RobustScaler = Path().resolve()/f"mood_tagger_master_thesis_V2/Scaler/RobustScaler.sav"
        loaded_scaler = pickle.load(open(filename_RobustScaler, 'rb'))

    if scale == 'StandardScaler':
        filename_StandardScaler = Path().resolve()/f"mood_tagger_master_thesis_V2/Scaler/StandardScaler.sav"
        loaded_scaler = pickle.load(open(filename_StandardScaler, 'rb'))
    
    for test_feat, test_targ, test_annot, test_name, test_genre in test_loader:
        with torch.no_grad():
            test_feat = test_feat.to(device)
            
            if transform == 'log': 
                model_outs = np.exp(model(test_feat[:, :, 0]).cpu().numpy()[0])
            elif transform == 'reciprocal':
                model_outs = np.reciprocal(model(test_feat[:, :, 0]).cpu().numpy()[0])
            else:
                model_outs = model(test_feat[:, :, 0]).cpu().numpy()[0]
            
            error = test_targ[0].numpy() - model_outs
        
        if training == 'training':
            annotations.append(np.exp(test_targ.cpu().numpy()[0]))
        else:
            annotations.append(test_targ.cpu().numpy()[0])

        


        predictions.append(model_outs)
        
        #print(f"Annotation:{test_targ.cpu().numpy()[0]} ") 
        
        #print(f"Prediction:{formatted_out} ")
        #print(f"Mean: [13.28, 9.53, 13.17, 9.47, 15.06, 19.69, 12.22, 8.12, 3.24] \n")
        prediction_names.append(test_name[0])
        prediction_genres.append(test_genre[0])
        actual.append(test_targ.cpu().numpy()[0])

    from sklearn.metrics import r2_score
    
   # testing = loaded_scaler.inverse_transform(predictions) 
    if scale is not None: 
        predictions = loaded_scaler.inverse_transform(predictions)
        
        #print(predictions)
        actual = loaded_scaler.inverse_transform(actual)
    formatted_out = np.round(predictions, 2)
    formatted_annot = np.round(actual, 2)
    
    for i in range(len(predictions)):
        print(f"Annotation:{formatted_annot[i]} ") 
        print(f"Prediction:{formatted_out[i]} ")
        print(f"Mean: [13.28, 9.53, 13.17, 9.47, 15.06, 19.69, 12.22, 8.12, 3.24] \n")

    errors = np.array(predictions) - np.array(actual)


    import pandas as pd
    if isinstance(train_y, str):
        error_df = pd.DataFrame(errors)#, columns=[train_y]
    else:
        error_df = pd.DataFrame(errors, columns=train_y)
    print(error_df)
    mean_errors_df = error_df.mean()
    mean_abs_errors_df = error_df.abs().mean()
    mse_df = (error_df**2).mean()
    rmse_df = np.sqrt(mse_df)
    max_df = error_df.max()
    min_df = error_df.min()

    r2 = r2_score(np.array(actual), np.array(predictions))
    print(mse_df)
    index_list = ['mean error', 'm abs error', 'mse', 'rmse', 'maximums', 'minimums']

    #concatenated_df = pd.concat([mean_errors_df, mean_abs_errors_df, mse_df, rmse_df, max_df, min_df], keys=index_list)

    data = {
                'Model' : model_name,
                'ME' : mean_errors_df.values,
                'MAE' :  mean_abs_errors_df.values,
                'MSE' : mse_df.values,
                'RMSE' : rmse_df.values,
                'Min' : min_df.values,
                "Max" : max_df.values,
                "R2" : r2
            }
    df = pd.DataFrame(data)


    def ff(array):
        return np.array2string(array, precision=2, separator=' \t ', suppress_small=True)



    # print(df) 
    # print('R2!')
    # print(r2_score(np.array(actual), np.array(predictions)))
    df['model'] = model_name


    
    #metrics for normal csv
    evaluation = {
                'Model' : [model_name],
                'ME' : [np.mean(mean_errors_df)],
                'MAE' :  [np.mean(mean_abs_errors_df)],
                'MSE' : [np.mean(mse_df)],
                'RMSE' : [np.mean(rmse_df)],
                "R2" : r2
            }
    #metrics for detailed
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

    #add metric for normal csv
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




    print(f"\noverall: \nME: {np.mean(mean_errors_df):.2f} \nMAE: {np.mean(mean_abs_errors_df):.2f}\nMSE: {np.mean(mse_df):.2f}\nRMSE: {np.mean(rmse_df):.2f}")

    if plot:
        df_pred = generate_dataframe(predictions, prediction_genres)
        plot_data(df_pred, f'Test Predictions {model_name}')

        df_an = generate_dataframe(annotations, prediction_genres)
        plot_data(df_an, f'Test Annotations')
        print('plotted')
        pass
    each = {'ME': mean_errors_df,'MAE' : mean_abs_errors_df, 'MSE' : mse_df, 'RMSE' :rmse_df, 'maximum': maximums, 'minimum': minimums}
    overall = {'ME': np.mean(mean_errors_df), 'MAE': np.mean(mean_abs_errors_df), 'MSE' : np.mean(mse_df), 'RMSE': np.mean(rmse_df)}
    return each, overall
