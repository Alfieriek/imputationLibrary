import pandas as pd
import numpy as np
from imputationLibrary import meanImputation

def test_inputTrainingData():
    df_before = pd.DataFrame([np.nan,1,2,np.nan,0,0,0,0])
    df_after = pd.DataFrame([0.5,1,2,0.5,0,0,0,0])
    result = meanImputation.inputTrainingData(df_before)
    assert df_after.equals(result)

def test_inputTestData():
    df_training = pd.DataFrame([0.5,1,2,0.5,0,0,0,0])
    df_test = pd.DataFrame([np.nan, 1, 1, np.nan, 1, 1, 0, 0])
    df_after = pd.DataFrame([0.5, 1.0, 1.0, 0.590909, 1.0, 1.0, 0.0, 0.0])
    result = meanImputation.inputTestData(df_test, df_training, ignore_index=True)
    result = result.reset_index()
    assert np.isclose(df_after[:][0], result[:][0], rtol=1e-05, atol=1e-05).all() == True