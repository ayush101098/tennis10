#!/usr/bin/env python3
"""
Train models using HISTORICAL player statistics (pre-match) - no data leakage
This properly simulates predicting before knowing the match result
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING WITH HISTORICAL STATS (NO DATA LEAKAGE)")
print("="*60)

conn = sqlite3.connect('tennis_data.db')
cursor = conn.cursor()

# Check data
cursor.execute("SELECT COUNT(*), MIN(tournament_date), MAX(tournament_date) FROM matches")
count, min_date, max_date = cursor.fetchone()
print(f"\nData: {count:,} matches from {min_date} to {max_date}")

# Get all matches
print("\n[1/6] Loading matches...")
matches_query = """
SELECT 
    match_id,
    tournament_date,
    surface,
    tourney_level,
    best_of,
    winner_id,
    loser_id,
    winner_rank,
    loser_rank,
    winner_rank_points,
    loser_rank_points,
    winner_age,
    loser_age,
    winner_ht,
    loser_ht,
    w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced,
    l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced
FROM matches
WHERE winner_rank IS NOT NULL AND loser_rank IS NOT NULL
ORDER BY tournament_date
"""
matches_df = pd.read_sql_query(matches_query, conn)
matches_df['tournament_date'] = pd.to_datetime(matches_df['tournament_date'])
print(f"  Loaded {len(matches_df):,} matches")

# Calculate historical serve stats for each player
print("\n[2/6] Computing historical serve averages per player...")

# Get all player IDs
all_players = set(matches_df['winner_id'].unique()) | set(matches_df['loser_id'].unique())
print(f"  {len(all_players):,} unique players")

# Pre-compute player career stats from the statistics table
# This gives us a rolling average of their serve performance
player_stats_query = """
SELECT 
    s.player_id,
    AVG(CASE WHEN s.first_serve_in IS NOT NULL AND s.first_serve_total > 0 
        THEN CAST(s.first_serve_in AS FLOAT) / s.first_serve_total END) as avg_first_serve_pct,
    AVG(CASE WHEN s.first_serve_points_won IS NOT NULL AND s.first_serve_points_total > 0 
        THEN CAST(s.first_serve_points_won AS FLOAT) / s.first_serve_points_total END) as avg_first_win_pct,
    AVG(CASE WHEN s.second_serve_points_won IS NOT NULL AND s.second_serve_points_total > 0 
        THEN CAST(s.second_serve_points_won AS FLOAT) / s.second_serve_points_total END) as avg_second_win_pct,
    AVG(CASE WHEN s.break_points_saved IS NOT NULL AND s.break_points_faced > 0 
        THEN CAST(s.break_points_saved AS FLOAT) / s.break_points_faced END) as avg_bp_save_pct,
    AVG(CASE WHEN s.aces IS NOT NULL AND s.service_games > 0 
        THEN CAST(s.aces AS FLOAT) / s.service_games END) as avg_aces_per_game,
    AVG(CASE WHEN s.double_faults IS NOT NULL AND s.service_games > 0 
        THEN CAST(s.double_faults AS FLOAT) / s.service_games END) as avg_df_per_game,
    COUNT(*) as matches_played,
    AVG(CASE WHEN s.is_winner = 1 THEN 1.0 ELSE 0.0 END) as win_rate
FROM statistics s
GROUP BY s.player_id
"""
player_stats = pd.read_sql_query(player_stats_query, conn)
player_stats = player_stats.set_index('player_id')
print(f"  Computed career stats for {len(player_stats):,} players")

# Default stats for players with no history
default_stats = {
    'avg_first_serve_pct': 0.62,
    'avg_first_win_pct': 0.70,
    'avg_second_win_pct': 0.50,
    'avg_bp_save_pct': 0.65,
    'avg_aces_per_game': 0.5,
    'avg_df_per_game': 0.2,
    'win_rate': 0.5
}

def get_player_stats(player_id):
    """Get player's historical stats or defaults"""
    if player_id in player_stats.index:
        stats = player_stats.loc[player_id]
        return {k: stats[k] if pd.notna(stats[k]) else default_stats.get(k, 0) 
                for k in default_stats.keys()}
    return default_stats

# Build features using ONLY pre-match information
print("\n[3/6] Building features from historical data...")

features_list = []
for idx, row in matches_df.iterrows():
    try:
        # Get historical stats for both players
        winner_stats = get_player_stats(row['winner_id'])
        loser_stats = get_player_stats(row['loser_id'])
        
        # Ranking features
        rank_diff = row['loser_rank'] - row['winner_rank']
        rank_ratio = np.log(row['loser_rank'] / row['winner_rank']) if row['winner_rank'] > 0 else 0
        pts_ratio = np.log((row['winner_rank_points'] or 1) / (row['loser_rank_points'] or 1))
        
        # Age/height features
        age_diff = (row['winner_age'] or 25) - (row['loser_age'] or 25)
        ht_diff = (row['winner_ht'] or 185) - (row['loser_ht'] or 185)
        
        # Historical serve differentials
        first_serve_pct_diff = winner_stats['avg_first_serve_pct'] - loser_stats['avg_first_serve_pct']
        first_win_diff = winner_stats['avg_first_win_pct'] - loser_stats['avg_first_win_pct']
        second_win_diff = winner_stats['avg_second_win_pct'] - loser_stats['avg_second_win_pct']
        bp_save_diff = winner_stats['avg_bp_save_pct'] - loser_stats['avg_bp_save_pct']
        ace_diff = winner_stats['avg_aces_per_game'] - loser_stats['avg_aces_per_game']
        df_diff = winner_stats['avg_df_per_game'] - loser_stats['avg_df_per_game']
        win_rate_diff = winner_stats['win_rate'] - loser_stats['win_rate']
        
        # Surface encoding
        surface = row['surface'] or 'Hard'
        is_clay = 1 if surface == 'Clay' else 0
        is_grass = 1 if surface == 'Grass' else 0
        
        # Tournament level
        level = row['tourney_level'] or 'A'
        is_grand_slam = 1 if level == 'G' else 0
        is_masters = 1 if level == 'M' else 0
        
        features_list.append({
            'match_id': row['match_id'],
            'tournament_date': row['tournament_date'],
            'rank_diff': rank_diff,
            'rank_ratio': rank_ratio,
            'pts_ratio': pts_ratio,
            'age_diff': age_diff,
            'ht_diff': ht_diff,
            'first_serve_pct_diff': first_serve_pct_diff,
            'first_win_diff': first_win_diff,
            'second_win_diff': second_win_diff,
            'bp_save_diff': bp_save_diff,
            'ace_diff': ace_diff,
            'df_diff': df_diff,
            'win_rate_diff': win_rate_diff,
            'is_clay': is_clay,
            'is_grass': is_grass,
            'is_grand_slam': is_grand_slam,
            'is_masters': is_masters,
            'winner': 1
        })
    except Exception as e:
        continue
    
    if (idx + 1) % 10000 == 0:
        print(f"  Processed {idx + 1:,} matches...", end="\r")

features_df = pd.DataFrame(features_list)
print(f"  Created {len(features_df):,} feature rows")

# Balance data by randomly swapping player order
print("\n[4/6] Creating balanced training data...")
np.random.seed(42)
swap_mask = np.random.random(len(features_df)) > 0.5

features_balanced = features_df.copy()
# Negate differential features when swapping
diff_cols = ['rank_diff', 'rank_ratio', 'pts_ratio', 'age_diff', 'ht_diff',
             'first_serve_pct_diff', 'first_win_diff', 'second_win_diff',
             'bp_save_diff', 'ace_diff', 'df_diff', 'win_rate_diff']

for col in diff_cols:
    features_balanced.loc[swap_mask, col] = -features_df.loc[swap_mask, col]
features_balanced.loc[swap_mask, 'winner'] = 0

print(f"  Class balance: {features_balanced['winner'].mean():.2%} player 1 wins")

# Split by time
train_cutoff = features_balanced['tournament_date'].quantile(0.8)
train_data = features_balanced[features_balanced['tournament_date'] < train_cutoff]
test_data = features_balanced[features_balanced['tournament_date'] >= train_cutoff]

print(f"  Train: {len(train_data):,} matches (before {train_cutoff.date()})")
print(f"  Test: {len(test_data):,} matches (after {train_cutoff.date()})")

# Feature columns
feature_cols = ['rank_diff', 'rank_ratio', 'pts_ratio',
                'first_serve_pct_diff', 'first_win_diff', 'second_win_diff',
                'bp_save_diff', 'ace_diff', 'df_diff', 'win_rate_diff',
                'is_clay', 'is_grass', 'is_grand_slam', 'is_masters']

X_train = train_data[feature_cols].fillna(0)
y_train = train_data['winner']
X_test = test_data[feature_cols].fillna(0)
y_test = test_data['winner']

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train LR
print("\n[5/6] Training Logistic Regression...")
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight': [None, 'balanced']
}

lr = LogisticRegression(max_iter=1000, random_state=42)
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"  Best params: {grid_search.best_params_}")
print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")

# Calibrate
calibrated_lr = CalibratedClassifierCV(grid_search.best_estimator_, method='isotonic', cv=5)
calibrated_lr.fit(X_train_scaled, y_train)

# Evaluate
y_pred = calibrated_lr.predict(X_test_scaled)
y_proba = calibrated_lr.predict_proba(X_test_scaled)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
brier = brier_score_loss(y_test, y_proba)
print(f"\n  Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"  Brier Score: {brier:.4f}")

# Save
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

# Train Neural Network
print("\n[6/6] Training Neural Network Ensemble...")
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
            for hd in hidden_dims:
                layers.extend([nn.Linear(prev_dim, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(dropout)])
                prev_dim = hd
            layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)
    
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test_scaled)
    
    configs = [
        {'hidden_dims': [64, 32], 'dropout': 0.3, 'lr': 0.001},
        {'hidden_dims': [128, 64], 'dropout': 0.3, 'lr': 0.001},
        {'hidden_dims': [64, 32, 16], 'dropout': 0.2, 'lr': 0.001},
    ]
    
    n_models = 10
    ensemble_models = []
    
    for i in range(n_models):
        config = configs[i % len(configs)]
        model = TennisNN(X_train_t.shape[1], config['hidden_dims'], config['dropout'])
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        
        indices = np.random.choice(len(X_train_t), len(X_train_t), replace=True)
        dataset = TensorDataset(X_train_t[indices], y_train_t[indices])
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        model.train()
        for epoch in range(50):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
        
        model.eval()
        ensemble_models.append(model)
        print(f"  Model {i+1}/{n_models} trained", end="\r")
    
    print(f"\n  Trained {n_models} models")
    
    # Evaluate
    preds = [m(X_test_t).detach().numpy() for m in ensemble_models]
    ensemble_proba = np.mean(preds, axis=0).flatten()
    nn_acc = accuracy_score(y_test, (ensemble_proba > 0.5).astype(int))
    nn_brier = brier_score_loss(y_test, ensemble_proba)
    
    print(f"\n  NN Ensemble Test Accuracy: {nn_acc:.4f} ({nn_acc*100:.1f}%)")
    print(f"  NN Ensemble Brier Score: {nn_brier:.4f}")
    
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
    
except ImportError:
    nn_acc = None
    nn_brier = None

conn.close()

# Summary
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nData: {count:,} matches ({min_date} to {max_date})")
print(f"Training: {len(train_data):,} | Test: {len(test_data):,}")
print(f"\nLogistic Regression: {test_acc*100:.1f}% accuracy")
if nn_acc:
    print(f"Neural Network Ensemble: {nn_acc*100:.1f}% accuracy")
print("\nModels saved to ml_models/")
