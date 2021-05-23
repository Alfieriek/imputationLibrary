from scipy import interpolate
import numpy as np

def inputData(x, y, x_new, derivate_order=0, k = 3):
    tck = interpolate.splrep(x, y, s=derivate_order, k=k)
    y_new = interpolate.splev(x_new, tck, der=0)
    return y_new

def inputTrainingData(df, derivate_order=0):
    training_df = df.copy()
    for col, data in training_df.iteritems():
        complete_df = training_df[col].dropna()
        if len(complete_df) == 1:
            k = 0
        elif len(complete_df) == 2:
            k = 1
        elif len(complete_df) == 3:
            k = 2
        else:
            k = 3
        x_new = np.setdiff1d(training_df.index, complete_df.index)
        if(len(x_new.tolist())>0) and k > 0:
            y_new = inputData(complete_df.index.values.tolist(), complete_df.iloc[:].values.tolist(), x_new.tolist(), derivate_order, k)
            for i in range(0, len(x_new)):
                training_df.at[x_new[i], col] = y_new[i]
    return training_df

def inputTestData(test_df, df, ignore_index=False, derivate_order=0):
    training_df = df.copy()
    for index, row in test_df.iterrows():
        training_df = training_df.append(row, ignore_index=ignore_index)
        if (len(row)-row.count())>=1:
            training_df = inputTrainingData(training_df, derivate_order)
    return training_df[-test_df.shape[0]:]

def input(training_df, test_df, derivate_order=0):
    df_training = inputTrainingData(training_df, derivate_order)
    df_test = inputTestData(test_df, training_df, derivate_order)
    return df_training, df_test