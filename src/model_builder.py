import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import numpy as np


def train_ghost_risk_model(df: pd.DataFrame, model_output_path: str) -> dict:
    """
    Trains a classification model to predict the 'is_ghost' label
    and saves the model object (Step 3).
    """
    print("\n--- STEP 3: Training Risk Model ---")

    # Drop rows where satellite data failed (999.0 from GEE)
    df = df[df['ndvi_change_metric'] != 999.0].copy()

    # 1. Feature Engineering and Selection
    features = ['total_loan_usd', 'cpi_score', 'ndvi_change_metric']

    # Filter out rows where the target (is_ghost) is NaN (shouldn't happen after audit, but safe)
    # Corrected redundant 'inplace=True' argument
    df.dropna(subset=['is_ghost'], inplace=True)
    X = df[features]
    y = df['is_ghost']

    # Simple imputation: fill NaN in features with the mean of the training data
    # (Especially needed for 'total_loan_usd' proxy)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mean_imputer = X_train.mean()
    X_train = X_train.fillna(mean_imputer)
    X_test = X_test.fillna(mean_imputer)

    # 2. Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 3. Predict and Score
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Save the model
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

    # 4. Apply score to the full (imputed) dataset
    # Must use the mean calculated from the training set to avoid data leakage
    X_full = df[features].fillna(mean_imputer)
    df['ghost_risk_score'] = model.predict_proba(X_full)[:, 1]

    metrics = {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'feature_importance': dict(zip(features, model.feature_importances_))
    }

    # IMPORTANT: The DF is modified with the score and returned for the final index step
    return metrics, df


def create_final_index(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Saves the final project index, sorted by risk score (Step 4)."""
    print("\n--- STEP 4: Creating Final Green Ghost Index ---")

    final_index = df.sort_values(by='ghost_risk_score', ascending=False)

    final_index = final_index[['project_id', 'project_name', 'country', 'latitude', 'longitude',
                               'ghost_risk_score', 'is_ghost', 'funded_capacity_mw',
                               'project_type', 'total_loan_usd', 'audit_status']]

    # Save the key dataset as CSV
    final_index.to_csv(output_path, index=False)
    print(f"Final Green Ghost Index saved to CSV: {output_path}")

    # NEW: Save as JSON for the HTML map
    json_path = output_path.replace('.csv', '.json')
    # Filter for projects with valid coordinates
    map_data = final_index[final_index['latitude'].notna() & final_index['longitude'].notna()].copy()
    # Select only the columns needed for the map visualization
    map_data = map_data[['project_name', 'country', 'latitude', 'longitude', 'ghost_risk_score']]
    map_data.to_json(json_path, orient='records', indent=4)
    print(f"Final Green Ghost Index saved to JSON for map: {json_path}")

    return final_index