import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Loads the CCPP dataset from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def split_data(df, target_column='PE', test_size=0.2, random_state=42):
    """
    Splits the dataframe into training and testing sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Data split into Train ({X_train.shape[0]}) and Test ({X_test.shape[0]}) sets.")
    return X_train, X_test, y_train, y_test
