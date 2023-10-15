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
    mean_errors_df = error_df.mean()
    mean_abs_errors_df = error_df.abs().mean()
    mse_df = (error_df**2).mean()
    rmse_df = np.sqrt(mse_df)
    max_df = error_df.max()
    min_df = error_df.min()
    index_list = ['mean error', 'm abs error', 'mse', 'rmse', 'maximums', 'minimums']
    #concatenated_df = pd.concat([mean_errors_df, mean_abs_errors_df, mse_df, rmse_df, max_df, min_df], keys=index_list)

    data = {
                'Model' : model_name,
                'ME' : mean_errors_df.values,
                'MAE' :  mean_abs_errors_df.values,
                'MSE' : mse_df.values,
                'RMSE' : rmse_df.values,
                'Min' : min_df.values,
                "Max" : max_df.values
            }
    df = pd.DataFrame(data)


    def ff(array):
        return np.array2string(array, precision=2, separator=' \t ', suppress_small=True)



    print(df) 
    df['model'] = model_name


    

    evaluation = {
                'Model' : [model_name],
                'ME' : [np.mean(mean_errors_df)],
                'MAE' :  [np.mean(mean_abs_errors_df)],
                'MSE' : [np.mean(mse_df)],
                'RMSE' : [np.mean(rmse_df)]
            }
    

    csv_information.update(evaluation)

    csv_information_df = pd.DataFrame(csv_information)

    ## if csv file does not exist, create blank file
    import csv
    if not os.path.isfile('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/performance/model_performance_list.csv'):
        header = ['Model', 'Labels', 'lr', 'custom_loss', 'dense_weight_alpha', 'snippet_duration_s', 'batch size', 'ME', 'MAE', 'MSE', 'RMSE']
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




    #TODO 
    if not csv_file.empty:
        #TODO HIER IST DER FEHLER
        ## Abrunden von Werten die sehr niedrig
        # if any(csv_file['Labels'].apply(lambda x: x == cur_labels)):!!!!!!!!!!!!!
        #     print('Triggered')!!!!!!!!!!
        condition = (csv_file['Model'] == cur_model) & (csv_file['lr'] == cur_lr)  & (csv_file['Labels'] == cur_labels) & ( csv_file['loss_function'] == cur_loss) & (csv_file['dense_weight_alpha'] == cur_alpha) & (csv_file['snippet_duration_s'] == cur_duration) & (csv_file['batch size'] == cur_batch)

        if any(condition):
            csv_file = csv_file[~condition]
            print(csv_file)
            #csv_file[condition] = csv_information_df
            new_csv = pd.concat([csv_file, csv_information_df])

        else:
            new_csv = pd.concat([csv_file, csv_information_df])
    else:
        new_csv = pd.concat([csv_file, csv_information_df])

    print(Path.cwd())
    new_csv.to_csv('/home/ykinoshita/humrec_mood_tagger/mood_tagger_master_thesis_V2/performance/model_performance_list.csv', index=False)



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
