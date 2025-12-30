#!/usr/bin/env python3
"""
Train optimized models on Jeff Sackmann's comprehensive ATP data
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING OPTIMIZED MODELS ON SACKMANN DATA")
print("="*60)

# Connect to database
conn = sqlite3.connect('tennis_data.db')

# Check data
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*), MIN(tournament_date), MAX(tournament_date) FROM matches")
count, min_date, max_date = cursor.fetchone()
print(f"\nData: {count:,} matches from {min_date} to {max_date}")

# Extract features directly from matches table (Sackmann data has stats in matches table)
print("\n[1/5] Extracting features...")

query = """
SELECT 
    m.match_id,
    m.tournament_date,
    m.surface,
    m.tourney_level,
    m.round,
    m.best_of,
    m.winner_id,
    m.loser_id,
    m.winner_rank,
    m.loser_rank,
    m.winner_rank_points,
    m.loser_rank_points,
    m.winner_age,
    m.loser_age,
    m.winner_ht,
    m.loser_ht,
    m.w_ace,
    m.w_df,
    m.w_svpt,
    m.w_1stIn,
    m.w_1stWon,
    m.w_2ndWon,
    m.w_SvGms,
    m.w_bpSaved,
    m.w_bpFaced,
    m.l_ace,
    m.l_df,
    m.l_svpt,
    m.l_1stIn,
    m.l_1stWon,
    m.l_2ndWon,
    m.l_SvGms,
    m.l_bpSaved,
    m.l_bpFaced,
    m.minutes
FROM matches m
WHERE m.winner_rank IS NOT NULL 
AND m.loser_rank IS NOT NULL
AND m.w_svpt IS NOT NULL
AND m.l_svpt IS NOT NULL
AND m.w_svpt > 0 
AND m.l_svpt > 0
ORDER BY m.tournament_date
"""

matches_df = pd.read_sql_query(query, conn)
print(f"  Matches with complete data: {len(matches_df):,}")

# Calculate derived features for each match
print("\n[2/5] Engineering features...")

features_list = []
for idx, row in matches_df.iterrows():
    try:
        # Rankings-based features
        rank_diff = row['loser_rank'] - row['winner_rank']
        rank_ratio = np.log(row['loser_rank'] / row['winner_rank']) if row['winner_rank'] > 0 else 0
        
        # Points-based features
        pts_diff = (row['winner_rank_points'] or 0) - (row['loser_rank_points'] or 0)
        pts_ratio = np.log((row['winner_rank_points'] or 1) / (row['loser_rank_points'] or 1))
        
        # Age features
        age_diff = (row['winner_age'] or 25) - (row['loser_age'] or 25)
        
        # Height features (serves)
        ht_diff = (row['winner_ht'] or 185) - (row['loser_ht'] or 185)
        
        # Serve statistics for winner
        w_first_serve_pct = row['w_1stIn'] / row['w_svpt'] if row['w_svpt'] > 0 else 0.62
        w_first_win_pct = row['w_1stWon'] / row['w_1stIn'] if row['w_1stIn'] > 0 else 0.70
        w_second_win_pct = row['w_2ndWon'] / (row['w_svpt'] - row['w_1stIn']) if (row['w_svpt'] - row['w_1stIn']) > 0 else 0.50
        w_bp_save_pct = row['w_bpSaved'] / row['w_bpFaced'] if row['w_bpFaced'] > 0 else 0.65
        w_aces_per_game = row['w_ace'] / row['w_SvGms'] if row['w_SvGms'] > 0 else 0.5
        w_df_per_game = row['w_df'] / row['w_SvGms'] if row['w_SvGms'] > 0 else 0.2
        
        # Serve points won %
        w_serve_pts_won = (row['w_1stWon'] + row['w_2ndWon']) / row['w_svpt'] if row['w_svpt'] > 0 else 0.65
        
        # Serve statistics for loser
        l_first_serve_pct = row['l_1stIn'] / row['l_svpt'] if row['l_svpt'] > 0 else 0.62
        l_first_win_pct = row['l_1stWon'] / row['l_1stIn'] if row['l_1stIn'] > 0 else 0.70
        l_second_win_pct = row['l_2ndWon'] / (row['l_svpt'] - row['l_1stIn']) if (row['l_svpt'] - row['l_1stIn']) > 0 else 0.50
        l_bp_save_pct = row['l_bpSaved'] / row['l_bpFaced'] if row['l_bpFaced'] > 0 else 0.65
        l_aces_per_game = row['l_ace'] / row['l_SvGms'] if row['l_SvGms'] > 0 else 0.5
        l_df_per_game = row['l_df'] / row['l_SvGms'] if row['l_SvGms'] > 0 else 0.2
        
        # Serve points won %
        l_serve_pts_won = (row['l_1stWon'] + row['l_2ndWon']) / row['l_svpt'] if row['l_svpt'] > 0 else 0.65
        
        # Differential features (Winner - Loser)
        first_serve_pct_diff = w_first_serve_pct - l_first_serve_pct
        first_win_diff = w_first_win_pct - l_first_win_pct
        second_win_diff = w_second_win_pct - l_second_win_pct
        bp_save_diff = w_bp_save_pct - l_bp_save_pct
        ace_diff = w_aces_per_game - l_aces_per_game
        df_diff = w_df_per_game - l_df_per_game
        serve_pts_diff = w_serve_pts_won - l_serve_pts_won
        
        # Surface encoding
        surface = row['surface'] or 'Hard'
        is_clay = 1 if surface == 'Clay' else 0
        is_grass = 1 if surface == 'Grass' else 0
        is_hard = 1 if surface == 'Hard' else 0
        
        # Tournament level encoding
        level = row['tourney_level'] or 'A'
        is_grand_slam = 1 if level == 'G' else 0
        is_masters = 1 if level == 'M' else 0
        
        features_list.append({
            'match_id': row['match_id'],
            'tournament_date': row['tournament_date'],
            
            # Ranking features
            'rank_diff': rank_diff,
            'rank_ratio': rank_ratio,
            'pts_diff': pts_diff,
            'pts_ratio': pts_ratio,
            
            # Physical features
            'age_diff': age_diff,
            'ht_diff': ht_diff,
            
            # Serve features (differentials)
            'first_serve_pct_diff': first_serve_pct_diff,
            'first_win_diff': first_win_diff,
            'second_win_diff': second_win_diff,
            'bp_save_diff': bp_save_diff,
            'ace_diff': ace_diff,
            'df_diff': df_diff,
            'serve_pts_diff': serve_pts_diff,
            
            # Surface/tournament
            'is_clay': is_clay,
            'is_grass': is_grass,
            'is_hard': is_hard,
            'is_grand_slam': is_grand_slam,
            'is_masters': is_masters,
            
            # Target (winner is always player 1 in this data)
            'winner': 1
        })
        
    except Exception as e:
        continue
        
    if (idx + 1) % 10000 == 0:
        print(f"  Processed {idx + 1:,} matches...", end="\r")

features_df = pd.DataFrame(features_list)
print(f"  Created {len(features_df):,} feature rows")

# Now we need to create balanced data by randomly swapping player 1 and 2
print("\n[3/5] Creating balanced training data...")

# For each row, randomly decide if we swap players (simulate predicting either player)
np.random.seed(42)
swap_mask = np.random.random(len(features_df)) > 0.5

# Create swapped version
features_swapped = features_df.copy()
for col in ['rank_diff', 'pts_diff', 'age_diff', 'ht_diff',
            'first_serve_pct_diff', 'first_win_diff', 'second_win_diff',
            'bp_save_diff', 'ace_diff', 'df_diff', 'serve_pts_diff']:
    features_swapped.loc[swap_mask, col] = -features_df.loc[swap_mask, col]

for col in ['rank_ratio', 'pts_ratio']:
    features_swapped.loc[swap_mask, col] = -features_df.loc[swap_mask, col]

# After swapping, winner becomes 0 (because we swapped who player 1 is)
features_swapped.loc[swap_mask, 'winner'] = 0

print(f"  Class balance: {features_swapped['winner'].mean():.2%} player 1 wins")

# Split by time (train on older, test on newer)
features_swapped['tournament_date'] = pd.to_datetime(features_swapped['tournament_date'])
train_cutoff = features_swapped['tournament_date'].quantile(0.8)

train_data = features_swapped[features_swapped['tournament_date'] < train_cutoff]
test_data = features_swapped[features_swapped['tournament_date'] >= train_cutoff]

print(f"  Train: {len(train_data):,} matches (before {train_cutoff.date()})")
print(f"  Test: {len(test_data):,} matches (after {train_cutoff.date()})")

# Feature columns for model
feature_cols = ['rank_diff', 'rank_ratio', 'pts_ratio',
                'first_serve_pct_diff', 'first_win_diff', 'second_win_diff',
                'bp_save_diff', 'ace_diff', 'df_diff', 'serve_pts_diff',
                'is_clay', 'is_grass', 'is_grand_slam', 'is_masters']

X_train = train_data[feature_cols].fillna(0)
y_train = train_data['winner']
X_test = test_data[feature_cols].fillna(0)
y_test = test_data['winner']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression with hyperparameter tuning
print("\n[4/5] Training Logistic Regression with grid search...")

param_grid = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight': [None, 'balanced']
}

lr = LogisticRegression(max_iter=1000, random_state=42)
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"  Best params: {grid_search.best_params_}")
print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")

best_lr = grid_search.best_estimator_

# Calibrate for better probabilities
print("  Calibrating probabilities...")
calibrated_lr = CalibratedClassifierCV(best_lr, method='isotonic', cv=5)
calibrated_lr.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred = calibrated_lr.predict(X_test_scaled)
y_proba = calibrated_lr.predict_proba(X_test_scaled)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
brier = brier_score_loss(y_test, y_proba)
print(f"\n  Test Accuracy: {test_acc:.4f}")
print(f"  Brier Score: {brier:.4f}")

# Save LR model
lr_model = {
    'model': calibrated_lr,
    'scaler': scaler,
    'features': feature_cols,
    'best_params': grid_search.best_params_,
    'test_accuracy': test_acc,
    'brier_score': brier
}
with open('ml_models/logistic_regression_trained.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("  Saved to ml_models/logistic_regression_trained.pkl")

# Train Neural Network Ensemble
print("\n[5/5] Training Neural Network Ensemble...")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    class TennisNN(nn.Module):
        def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
            self.net = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.net(x)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)
    
    # Train ensemble with different configurations
    configs = [
        {'hidden_dims': [64, 32], 'dropout': 0.3, 'lr': 0.001},
        {'hidden_dims': [128, 64], 'dropout': 0.3, 'lr': 0.001},
        {'hidden_dims': [64, 32, 16], 'dropout': 0.2, 'lr': 0.001},
        {'hidden_dims': [128, 64, 32], 'dropout': 0.4, 'lr': 0.0005},
        {'hidden_dims': [256, 128, 64], 'dropout': 0.3, 'lr': 0.0005},
    ]
    
    n_models = 15
    ensemble_models = []
    
    print(f"  Training {n_models} ensemble models...")
    for i in range(n_models):
        config = configs[i % len(configs)]
        model = TennisNN(X_train_tensor.shape[1], config['hidden_dims'], config['dropout'])
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        
        # Bootstrap sample
        indices = np.random.choice(len(X_train_tensor), len(X_train_tensor), replace=True)
        X_boot = X_train_tensor[indices]
        y_boot = y_train_tensor[indices]
        
        dataset = TensorDataset(X_boot, y_boot)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        model.train()
        for epoch in range(100):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        ensemble_models.append(model)
        print(f"    Model {i+1}/{n_models} trained", end="\r")
    
    print(f"\n    Trained {n_models} models")
    
    # Evaluate ensemble
    preds = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            pred = model(X_test_tensor).numpy()
        preds.append(pred)
    
    ensemble_proba = np.mean(preds, axis=0).flatten()
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    
    nn_acc = accuracy_score(y_test, ensemble_pred)
    nn_brier = brier_score_loss(y_test, ensemble_proba)
    print(f"\n  NN Ensemble Test Accuracy: {nn_acc:.4f}")
    print(f"  NN Ensemble Brier Score: {nn_brier:.4f}")
    
    # Save NN ensemble
    nn_ensemble = {
        'models': ensemble_models,
        'scaler': scaler,
        'features': feature_cols,
        'n_models': n_models,
        'test_accuracy': nn_acc,
        'brier_score': nn_brier
    }
    with open('ml_models/neural_network_ensemble.pkl', 'wb') as f:
        pickle.dump(nn_ensemble, f)
    print("  Saved to ml_models/neural_network_ensemble.pkl")
    
except ImportError as e:
    print(f"  PyTorch not available: {e}")
    nn_acc = None
    nn_brier = None

conn.close()

# Final summary
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nData: {count:,} matches ({min_date} to {max_date})")
print(f"Training set: {len(train_data):,} matches")
print(f"Test set: {len(test_data):,} matches")
print(f"\nLogistic Regression:")
print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"  Brier Score: {brier:.4f}")
if nn_acc:
    print(f"\nNeural Network Ensemble ({n_models} models):")
    print(f"  Accuracy: {nn_acc:.4f} ({nn_acc*100:.1f}%)")
    print(f"  Brier Score: {nn_brier:.4f}")
print(f"\nModels saved to ml_models/")
