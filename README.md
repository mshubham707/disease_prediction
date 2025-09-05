

````markdown
# Disease Prediction Project

## Overview
This project is a machine learning-based web application for predicting diseases from user-input symptoms. Built for a hackathon, it uses a dataset of 42 diseases with ~120 samples each, reduced to ~40-50 features via Variance Inflation Factor (VIF) to address multicollinearity. Four models (LogisticRegression, RandomForest, GradientBoosting, XGBoost) are tuned, achieving >97% accuracy, with the best model deployed in a Flask web app. The app features a modern, healthcare-themed UI with a search bar for easy symptom selection, making it user-friendly for early disease diagnosis.

## Features
- **Data Preprocessing**: VIF-based feature reduction (threshold=5) to remove multicollinearity, producing `train_processed.csv` and `test_processed.csv`.
- **Model Training**: GridSearchCV tunes four models, selecting the best (e.g., Logistic Regression) with >97% accuracy on 42-class classification.
- **Web App**: Flask app with a responsive UI, including:
  - Symptom selection form (~40–50 checkboxes).
  - Real-time symptom search bar (JavaScript).
  - Modern healthcare-themed design (blue/white palette, rounded elements).
  - Predictions with confidence scores (e.g., “Fungal infection, 95.23%”).
- **Deployment**: Hosted on Render for live access.

## Prerequisites
- Python 3.8+
- Virtual environment (recommended):  
  ```bash
  python -m venv dp_env
````

* Dependencies (in `requirements.txt`):

  ```
  pandas
  numpy
  scikit-learn
  xgboost
  statsmodels
  flask
  ```

## Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/disease-prediction.git
   cd disease-prediction
   ```

2. **Set Up Virtual Environment (optional)**:

   ```bash
   python -m venv dp_env
   .\dp_env\Scripts\activate  # Windows
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Directory Structure**:

   ```
   project_root/
   ├── data/
   │   ├── raw/                    # train.csv, test.csv
   │   ├── processed/              # train_reduced_vif.csv, test_reduced_vif.csv
   ├── models/                     # trained_model.pkl, label_encoder.pkl
   ├── predictions/                # predictions.csv
   ├── Notebooks/                  # Jupyter notebooks for exploration
   ├── src/                        # Python scripts and Flask app
   ├── static/                     # CSS for Flask UI
   ├── templates/                  # HTML templates for Flask
   ├── requirements.txt
   ├── README.md
   ```

## Usage

### Preprocess Data

Run `preprocess.py` to encode prognosis and reduce features via VIF:

```bash
python src/data/preprocess.py
```

**Outputs**:

* `data/processed/train_reduced_vif.csv`
* `data/processed/test_reduced_vif.csv`
* `models/label_encoder.pkl`

### Train Model

Run `train.py` to tune models and save the best:

```bash
python src/models/train.py
```

**Outputs**:

* `models/trained_model.pkl`

### Generate Predictions

Run `predict.py` for test set predictions:

```bash
python src/models/predict.py
```

**Outputs**:

* `predictions/predictions.csv` (42 rows, prognosis\_encoded, prognosis)

### Run Flask App

Start the web app:

```bash
python src/app.py
```

**Access**: [http://127.0.0.1:5000](http://127.0.0.1:5000)
**Features**: Select symptoms, use search bar, view predictions with confidence.

## Deployment

* **Live URL**: \[Your Render URL, e.g., [https://disease-prediction.onrender.com](https://disease-prediction.onrender.com)]

**Steps**:

1. Push to GitHub:

   ```bash
   git push origin main
   ```
2. Create a Web Service on Render (free tier).
3. Settings:

   * Runtime: Python
   * Build Command: `pip install -r requirements.txt`
   * Start Command: `python src/app.py`

Deploy and access the URL.

## Results

* **Data**: Reduced from 132 to \~40–50 features via VIF (threshold=5), handling multicollinearity.
* **Models**: Best model (Logistic Regression) achieves >97% accuracy on validation and test sets (42 classes, balanced).
* **UI**: Modern Flask app with search bar, responsive symptom grid, and healthcare-themed design (blue/white palette).
* **Output**: `predictions.csv` with accurate disease predictions for test set (42 rows).

## Notebooks

* `data_preparation.ipynb`: Data cleaning and initial preprocessing.
* `exploratory_data_analysis.ipynb`: Visualizations and VIF analysis.
* `model_training.ipynb`: Model tuning and evaluation.

## Contact

For questions, contact \mshubham707@gmail.com.

## Acknowledgments

Built for a hackathon to demonstrate end-to-end ML pipeline skills, from preprocessing to deployment, as part of **PW Skills Data Analytics certification**.
