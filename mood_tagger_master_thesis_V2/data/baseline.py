import pandas as pd
import numpy as np
import os

# TODO move data out of here





def mean_value_predictor(Filename = 'GEMS-INN_2023-01-30_expert.csv'):
    

    def ff(array):
        return np.array2string(array, precision=2, separator=' \t ', suppress_small=True)    

    DATA_PATH = os.path.dirname(os.path.realpath(__file__))

    ANNOT_CSV_FILE = os.path.join(DATA_PATH, Filename)
    GEMS_9 = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
    NUM_CLASSES = len(GEMS_9)


    emma_df = pd.read_csv(ANNOT_CSV_FILE, encoding="ISO-8859-1")
    emma_df.dropna(inplace=True)

    mean_value_features = emma_df[GEMS_9].mean() #get mean values for each label
    print(f"Mean values: \n{mean_value_features} ")

    error = emma_df[GEMS_9] - mean_value_features #true value - mean values

    abs_error = error.apply(abs) #get absolute values of error
    mean_abs_errors = abs_error.sum()/len(abs_error) #divide by len to obtain mean value


    mean_errors = error.sum()/len(error) #divide by len to obtain mean value

    squared_errors = error**2

    mean_squared_errors = squared_errors.sum()/len(squared_errors) #divide by len to obtain mean value

    rmse = np.sqrt(mean_squared_errors)

    maximums = error.max() #max value of each col
    minimums = error.min() #min value of each col


    print(f"for {len(error)} test samples:\n"
          f"          {' '.join(GEMS_9)} \n"
          f"mean err: {ff(mean_errors.values)} \n"
          f"m abs er: {ff(mean_abs_errors.values)} \n"
          f"m squared er: {ff(mean_squared_errors.values)} \n"
          f"root m squared er: {ff(rmse.values)} \n"
          f"maximums: {ff(maximums.values)} \n"
          f"minimums: {ff(minimums.values)} ")

    print(f"\noverall: \nME: {np.mean(mean_errors.values):.2f} \nMAE: {np.mean(mean_abs_errors.values):.2f} \nMSE: {np.mean(mean_squared_errors.values):.2f}\nRMSE: {np.mean(rmse.values):.2f}")

if __name__ == "__main__":
    mean_value_predictor()
