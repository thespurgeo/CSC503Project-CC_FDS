from imblearn.over_sampling import SMOTE

def ovsample(X,y):
    # Resample the minority class.
    sm = SMOTE(sampling_strategy='minority', random_state=7)

    # Fit the model to generate the data.
    #oversampled_trainX, oversampled_trainY = sm.fit_resample(X, y)
    Xn,yn=sm.fit_sample(X,y)
    return Xn,yn
   # oversampled_train = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)
   # oversampled_train.columns = normalized_df.columns

