import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import joblib
import os


def check_binary(df):
    non_binary_cols = []
    for col in df.columns:
        for i in df[col].unique():
            if i not in [0,1]:
                non_binary_cols.append(col)
                break
    return non_binary_cols

def missing_values(df):
    null_columns = [col for col in df.columns if df[col].isnull().sum() > 0]

    if len(null_columns) == 0:
        print("No null values in the dataset")
        return df  # return unchanged df
    else:
        print("Null values are present in:", null_columns) 
        imputer = SimpleImputer(strategy='most_frequent')
        print("Imputing with most-frequent values")
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        return df_imputed

# VIF-based feature reduction
def calculate_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_data    
   

def preprocess_data():
    # Load raw data
    print("Loading raw data....")
    try:
        train_df = pd.read_csv('data/raw/Training.csv')
        test_df = pd.read_csv('data/raw/Testing.csv')
    except Exception as e:
        print("Data loading failed!")
        print(f"Error: {e}")
        return
    print("\nData Preprocessing started...\n")

    symptom_cols = [col for col in train_df.columns if col !="prognosis"]
    print("Initial columns:", len(symptom_cols))
    redundant_cols = []
    for i in symptom_cols:
        if train_df[i].var() ==0 or train_df[i].isnull().all():
            redundant_cols.append(i)
    print("Redundant columns:",len(redundant_cols))
    train_df = train_df.drop(columns=redundant_cols)
    print("Columns after removing redundant columns:",train_df.shape[-1])

    symptom_cols = [col for col in train_df.columns if col !="prognosis"]
    test_df = test_df[symptom_cols + ['prognosis']]

    #checking for binary columns, all columns except the target column
    print("checking for binary columns")
    bin_col = check_binary(train_df[symptom_cols])
    if len(bin_col) > 0:
        print("Non binary column other than target variable is present")
        print("Data not in correct format")
        return
    bin_col = check_binary(test_df[symptom_cols])
    if len(bin_col) > 0:
        print("Non binary column other than target variable is present")
        print("Data not in correct format")
        return    
    print("Checking missing values....")
    df_train_copy = train_df.copy()
    df_test_copy = test_df.copy()
    train_df = missing_values(df_train_copy)
    test_df = missing_values(df_test_copy)

    print("Checking for consistency")
    if train_df.columns.to_list() == test_df.columns.to_list():
        print("Train and Test data have same features")
    else:
        print("Train and Test data have different features")
        return
    
    print("Encoding target variable..")
    le = LabelEncoder()
    train_df['prognosis'] = le.fit_transform(train_df['prognosis'])
    test_df['prognosis'] = le.transform(test_df['prognosis'])
    
    print("Checking for multicollinearity....")

    threshold = 5
    dropped_features = []
    current_features = symptom_cols.copy()
    
    while True:
        vif_data = calculate_vif(train_df, current_features)
        high_vif = vif_data[vif_data['VIF'] > threshold].sort_values('VIF', ascending=False)
        if high_vif.empty:
            break
        to_drop = high_vif.iloc[0]['Feature']
        dropped_features.append(to_drop)
        current_features.remove(to_drop)
        print(f"Dropped {to_drop} with VIF {high_vif.iloc[0]['VIF']:.2f}")
    
    print("\nNumber of Features Dropped:", len(dropped_features))
    print("Number of Remaining Features:", len(current_features))
    # Create reduced datasets
    train_reduced = train_df[current_features + ['prognosis']]
    test_reduced = test_df[current_features + ['prognosis']]
    print(f"Reduced Training Shape: {train_reduced.shape}")
    print(f"Reduced Testing Shape: {test_reduced.shape}")

    
    # Save label encoder
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/label_encoder.pkl')
    print("Label encoder saved to models/label_encoder.pkl")
    
    # Save reduced datasets
    os.makedirs('data/processed', exist_ok=True)
    train_reduced.to_csv('data/processed/train_processed.csv', index=False)
    test_reduced.to_csv('data/processed/test_processed.csv', index=False)
    print("Reduced datasets saved to data/processed/")

if __name__ == "__main__":
    preprocess_data()