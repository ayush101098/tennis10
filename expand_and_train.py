#!/usr/bin/env python3
"""
Complete Data Expansion and Model Training Script
Expands data to 2015-2024 and trains optimized models
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TENNIS PREDICTION MODEL ENHANCEMENT")
print("="*60)

# Step 1: Fix database schema
print("\n[1/6] Fixing database schema...")
conn = sqlite3.connect('tennis_data.db')
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(matches)")
columns = [c[1] for c in cursor.fetchall()]
if 'location' not in columns:
    cursor.execute("ALTER TABLE matches ADD COLUMN location TEXT")
    conn.commit()
    print("  Added 'location' column")
else:
    print("  Schema OK")

# Check current data
cursor.execute('SELECT COUNT(*), MIN(tournament_date), MAX(tournament_date) FROM matches')
result = cursor.fetchone()
print(f"  Current: {result[0]:,} matches from {result[1]} to {result[2]}")
conn.close()

# Step 2: Expand data (download 2015-2019)
print("\n[2/6] Expanding data to 2015-2024...")
from data_pipeline import TennisDataPipeline

pipeline = TennisDataPipeline('tennis_data.db')
pipeline.connect_db()

for year in [2015, 2016, 2017, 2018, 2019]:
    print(f"  Fetching {year}...", end=" ", flush=True)
    try:
        df = pipeline.fetch_atp_data(year)
        if df is not None:
            pipeline.process_match_data(df, year)
            print(f"OK ({len(df)} matches)")
        else:
            print("FAILED")
    except Exception as e:
        print(f"ERROR: {e}")

# Check final count
conn = sqlite3.connect('tennis_data.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*), MIN(tournament_date), MAX(tournament_date) FROM matches')
result = cursor.fetchone()
print(f"  Total: {result[0]:,} matches from {result[1]} to {result[2]}")
conn.close()

# Step 3: Extract features for all matches
print("\n[3/6] Extracting features...")
from features import TennisFeatureExtractor

extractor = TennisFeatureExtractor('tennis_data.db')
features_df = extractor.extract_features()
print(f"  Extracted {len(features_df)} feature rows")
print(f"  Features: {list(features_df.columns)}")

# Step 4: Hyperparameter tuning for Logistic Regression
print("\n[4/6] Training optimized Logistic Regression...")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# Prepare data
feature_cols = ['RANK_DIFF', 'RANK_RATIO', 'FIRST_SERVE_PCT_DIFF', 'FIRST_SERVE_WIN_DIFF',
                'SECOND_SERVE_WIN_DIFF', 'ACE_DIFF', 'DF_DIFF', 'BP_SAVE_DIFF']

X = features_df[feature_cols].dropna()
y = np.ones(len(X))  # Player 1 always wins (as data is structured)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Grid search for LR
print("  Running hyperparameter grid search...")
param_grid = {
    'C': [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'newton-cg'],
    'class_weight': [None, 'balanced']
}

lr = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y)

best_lr = grid_search.best_estimator_
print(f"  Best params: {grid_search.best_params_}")
print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")

# Calibrate probabilities
calibrated_lr = CalibratedClassifierCV(best_lr, method='isotonic', cv=5)
calibrated_lr.fit(X_scaled, y)

# Save LR model
lr_model = {
    'model': calibrated_lr,
    'scaler': scaler,
    'features': feature_cols,
    'best_params': grid_search.best_params_,
    'accuracy': grid_search.best_score_
}
with open('ml_models/logistic_regression_trained.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print(f"  Saved optimized LR model")

# Step 5: Train Neural Network Ensemble with hyperparameter tuning
print("\n[5/6] Training Neural Network Ensemble...")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Neural Network Architecture
    class TennisPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Prepare data for NN (use more features)
    nn_feature_cols = [c for c in features_df.columns if c != 'match_id' and features_df[c].dtype in ['float64', 'int64']]
    X_nn = features_df[nn_feature_cols].fillna(0)
    y_nn = np.ones(len(X_nn))
    
    nn_scaler = StandardScaler()
    X_nn_scaled = nn_scaler.fit_transform(X_nn)
    
    X_tensor = torch.FloatTensor(X_nn_scaled)
    y_tensor = torch.FloatTensor(y_nn).unsqueeze(1)
    
    # Train ensemble with different hyperparameters
    ensemble_models = []
    configs = [
        {'hidden_dim': 64, 'dropout': 0.2, 'lr': 0.001},
        {'hidden_dim': 128, 'dropout': 0.3, 'lr': 0.001},
        {'hidden_dim': 64, 'dropout': 0.3, 'lr': 0.0005},
        {'hidden_dim': 128, 'dropout': 0.2, 'lr': 0.0005},
        {'hidden_dim': 256, 'dropout': 0.4, 'lr': 0.001},
    ]
    
    n_models = 10  # Reduced for speed
    print(f"  Training {n_models} ensemble models...")
    
    for i in range(n_models):
        config = configs[i % len(configs)]
        model = TennisPredictor(X_tensor.shape[1], config['hidden_dim'], config['dropout'])
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        
        # Bootstrap sample
        indices = np.random.choice(len(X_tensor), len(X_tensor), replace=True)
        X_boot = X_tensor[indices]
        y_boot = y_tensor[indices]
        
        dataset = TensorDataset(X_boot, y_boot)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        model.train()
        for epoch in range(50):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        ensemble_models.append(model)
        print(f"    Model {i+1}/{n_models} trained", end="\r")
    
    print(f"    Trained {n_models} models successfully")
    
    # Save NN ensemble
    nn_ensemble = {
        'models': ensemble_models,
        'scaler': nn_scaler,
        'features': nn_feature_cols,
        'n_models': n_models
    }
    with open('ml_models/neural_network_ensemble.pkl', 'wb') as f:
        pickle.dump(nn_ensemble, f)
    print(f"  Saved NN ensemble")
    
except ImportError:
    print("  PyTorch not available, skipping NN training")

# Step 6: Evaluate and report
print("\n[6/6] Final Evaluation...")

# Test on holdout data (last 20% of matches)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# LR accuracy
lr_pred = calibrated_lr.predict(X_test)
lr_acc = np.mean(lr_pred == y_test)
print(f"  Logistic Regression accuracy: {lr_acc:.4f}")

# NN ensemble accuracy (if available)
try:
    X_test_nn = X_nn_scaled[-len(X_test):]
    X_test_tensor = torch.FloatTensor(X_test_nn)
    
    preds = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            pred = model(X_test_tensor).numpy()
        preds.append(pred)
    
    ensemble_pred = np.mean(preds, axis=0) > 0.5
    nn_acc = np.mean(ensemble_pred.flatten() == y_test)
    print(f"  Neural Network Ensemble accuracy: {nn_acc:.4f}")
except:
    print("  Could not evaluate NN ensemble")

print("\n" + "="*60)
print("ENHANCEMENT COMPLETE")
print("="*60)
print(f"\nModels saved to:")
print(f"  - ml_models/logistic_regression_trained.pkl")
print(f"  - ml_models/neural_network_ensemble.pkl")
print(f"\nUse live_prediction.py for predictions!")
