
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def train_model():
    # Load data
    try:
        print("Loading data......")
        train_df = pd.read_csv('data/processed/train_processed.csv')
        symptom_cols = train_df.columns[:-1]
        X = train_df[symptom_cols]
        y = train_df['prognosis']

    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data Loaded Successfully")

    # Split for tuning
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42)

    # Define models and grids
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'param_grid': {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'param_grid': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        }
    }

    # Tune models
    best_models = {}
    best_scores = {}
    for name, config in models.items():
        print(f"Tuning {name}...")
        grid_search = GridSearchCV(config['model'], config['param_grid'], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        best_scores[name] = grid_search.best_score_
        print(f"Best {name} Parameters:", grid_search.best_params_)
        print(f"Best {name} CV Accuracy:", grid_search.best_score_)
    print("Model Tuning Completed")
    # Select best model
    best_model_name = max(best_scores, key=best_scores.get)
    best_model = best_models[best_model_name]
    print(f"Selected {best_model_name} with CV Accuracy: {best_scores[best_model_name]:.4f}")

    # Train on full data
    best_model.fit(X, y)

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/trained_model.pkl')
    print(f"Best model ({best_model_name}) saved to models/trained_model.pkl")

if __name__ == "__main__":
    train_model()
