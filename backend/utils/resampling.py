from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def resample_data(X, y):
    # Oversampling
    oversampler = RandomOverSampler(random_state=42)
    X_over, y_over = oversampler.fit_resample(X, y)
    
    print(f"\nOversampled class distribution:\n{y_over.value_counts()}")

    # Undersampling
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X_over, y_over)
    
    print(f"\nFinal resampled class distribution:\n{y_resampled.value_counts()}")

    return X_resampled, y_resampled