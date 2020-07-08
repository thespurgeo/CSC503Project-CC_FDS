from imblearn.under_sampling import RandomUnderSampler
def undersample(X,y):
    # Undersample the majority class.
    sm = RandomUnderSampler(sampling_strategy='majority', random_state=7)

    # Fit the model to generate the data.
    Xn,yn=sm.fit_sample(X,y)
    return Xn,yn
