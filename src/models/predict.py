import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def predict():
    # Load test data and model
    print("Loading processed data...")
    try:
        test_df = pd.read_csv('data/processed/test_processed.csv')
        model = joblib.load('models/trained_model.pkl')
        le = joblib.load('models/label_encoder.pkl')
    except Exception as e:
        print(f"Error:{e}")
    print("Data Loaded for predictions!")
    # Prepare features
    symptom_cols = test_df.columns[:-1]  # Exclude prognosis
    X_test = test_df[symptom_cols]
    y_test = test_df['prognosis']
    # Predict
    y_test_pred = model.predict(X_test)
    pred_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Prediction Accuracy {pred_accuracy:.4f}")
    
    # Create predictions DataFrame
    predictions = pd.DataFrame({
        'prognosis_encoded': y_test_pred,
        'prognosis': le.inverse_transform(y_test_pred)
    })
    
    # Save predictions
    predictions.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    predict()