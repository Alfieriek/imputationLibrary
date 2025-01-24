import numpy as np
import random
import math

def inputData(training_df, sliding_window_size=0.1):
    imputed_df = training_df.copy()
    for index, row in imputed_df.iterrows():
        if row.isna().sum() >= 1:  # Check if there's at least one NaN in the row
            nan_columns = row[row.isna()].index  # Get column names with NaN values
            for col in nan_columns:
                row_index = imputed_df.index.get_loc(index)
                rand_row = row_index
                window_size = sliding_window_size
                while np.isnan(imputed_df.iloc[rand_row, imputed_df.columns.get_loc(col)]):
                    lower_bound = max(0, row_index - math.ceil((imputed_df.shape[0] - 1) * window_size))
                    upper_bound = min(row_index + math.ceil((imputed_df.shape[0] - 1) * window_size), training_df.shape[0] - 1)
                    rand_row = random.randint(lower_bound, upper_bound)
                    window_size += 0.02
                imputed_df.iloc[row_index, imputed_df.columns.get_loc(col)] = imputed_df.iloc[rand_row, imputed_df.columns.get_loc(col)]
    return imputed_df

def inputTrainingData(training_df, sliding_window_size = 0.1):
    return inputData(training_df, sliding_window_size)

def inputTestData(test_df, training_df, ignore_index=False, sliding_window_size = 0.1):
    imputed_df = training_df.copy()
    for index, row in test_df.iterrows():
        imputed_df = imputed_df.append(row, ignore_index=ignore_index)
        if (len(row)-row.count())>=1:
            imputed_df = inputData(training_df, sliding_window_size)
    return imputed_df[-test_df.shape[0]:]

def input(df_training, df_test, sliding_window_size = 0.1):
    return inputTrainingData(df_training), inputTestData(df_test, df_training)