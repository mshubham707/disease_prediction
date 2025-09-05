from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import os

# Initialize Flask with explicit template and static folders
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Load model and label encoder
model_path = 'models/trained_model.pkl'
label_encoder_path = 'models/label_encoder.pkl'
train_reduced_path = 'data/processed/train_processed.csv'

model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

# Load symptom columns from training data
train_df = pd.read_csv(train_reduced_path)
symptom_cols = train_df.columns[:-1].tolist()  # Exclude prognosis

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html', symptoms=symptom_cols)

@app.route('/predict', methods=['POST'])
def predict():
    # Create binary input vector as DataFrame
    input_vector = np.zeros(len(symptom_cols))
    selected_symptoms = []
    for i, symptom in enumerate(symptom_cols):
        if symptom in request.form:
            input_vector[i] = 1
            selected_symptoms.append(symptom)
    
    # Convert to DataFrame with feature names
    input_df = pd.DataFrame([input_vector], columns=symptom_cols)
    
    # Predict
    pred_encoded = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    pred_disease = label_encoder.inverse_transform([pred_encoded])[0]
    confidence = np.max(pred_proba) * 100
    
    return render_template(
        'result.html',
        disease=pred_disease,
        confidence=f"{confidence:.2f}",
        selected_symptoms=selected_symptoms
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT env var or default to 5000
    app.run(host="0.0.0.0", port=port, debug=False)  # Bind to 0.0.0.0 for Spaces