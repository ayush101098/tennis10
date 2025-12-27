"""
Unified Model Training and Evaluation Script

This script:
1. Extracts features from the database using TennisFeatureExtractor
2. Trains Logistic Regression, Neural Network Ensemble, and evaluates Markov model
3. Compares all models on test data (2023-2024)
4. Generates performance reports and saves trained models
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')

# Import custom modules
from features import TennisFeatureExtractor
from hierarchical_model import HierarchicalTennisModel
from ml_models.logistic_regression import SymmetricLogisticRegression, forward_feature_selection, calculate_kelly_roi


# ============================================
# Neural Network Architecture
# ============================================
class SymmetricNeuralNetwork(nn.Module):
    """Neural network without bias terms for symmetric predictions"""
    
    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()


def train_single_nn(X_train, y_train, X_val, y_val, hidden_dim=100, epochs=200, lr=0.01, weight_decay=0.01):
    """Train a single neural network with early stopping"""
    model = SymmetricNeuralNetwork(X_train.shape[1], hidden_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    patience = 20
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    model.load_state_dict(best_weights)
    return model


def train_nn_ensemble(X_train, y_train, X_val, y_val, n_models=20):
    """Train bagging ensemble of neural networks"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    models = []
    for i in range(n_models):
        # Bootstrap sample
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        model = train_single_nn(X_boot, y_boot, X_val, y_val)
        models.append(model)
        print(f"  Trained model {i+1}/{n_models}")
    
    return models


def ensemble_predict(models, X_tensor):
    """Get average predictions from ensemble"""
    all_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds = model(X_tensor).numpy()
            all_preds.append(preds)
    return np.mean(all_preds, axis=0)


def augment_with_reverse(df, feature_cols):
    """Create symmetric training data by adding reverse perspective"""
    df_original = df.copy()
    df_original['winner'] = 1
    
    df_reverse = df.copy()
    df_reverse['winner'] = 0
    
    # Flip signs of difference features
    for col in df_reverse.columns:
        if col.endswith('_DIFF') or col in ['SERVEADV', 'COMPLETE_DIFF', 'DIRECT_H2H', 'FATIGUE_DIFF', 'RETIRED_DIFF']:
            df_reverse[col] = -df_reverse[col]
    
    return pd.concat([df_original, df_reverse], ignore_index=True)


def evaluate_model(predictions, actuals, model_name):
    """Calculate evaluation metrics for a model"""
    accuracy = accuracy_score(actuals, (predictions > 0.5).astype(int))
    # Handle case where all labels are the same
    ll = log_loss(actuals, predictions, labels=[0, 1])
    brier = brier_score_loss(actuals, predictions)
    roi = calculate_kelly_roi(predictions, actuals)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'log_loss': ll,
        'brier_score': brier,
        'roi': roi
    }


def main():
    """Main training and evaluation pipeline"""
    print("=" * 70)
    print("TENNIS PREDICTION MODEL TRAINING & EVALUATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ============================================
    # 1. EXTRACT FEATURES
    # ============================================
    print("\n" + "=" * 70)
    print("1. EXTRACTING FEATURES")
    print("=" * 70)
    
    conn = sqlite3.connect('tennis_data.db')
    feature_extractor = TennisFeatureExtractor('tennis_data.db')
    
    # Get all matches
    query = """
    SELECT match_id, tournament_date, surface, winner_id, loser_id
    FROM matches
    WHERE tournament_date >= '2020-01-01'
      AND surface IS NOT NULL
    ORDER BY tournament_date
    """
    matches = pd.read_sql_query(query, conn)
    print(f"Total matches to process: {len(matches):,}")
    
    # Extract features
    df_features = feature_extractor.extract_features_batch(
        match_ids=matches['match_id'].tolist(),
        uncertainty_threshold=0.7
    )
    print(f"Features extracted: {len(df_features):,} matches")
    
    # Define feature columns (exclude non-model features)
    feature_cols = [col for col in df_features.columns 
                   if col.endswith('_DIFF') or col in ['SERVEADV', 'COMPLETE_DIFF', 'DIRECT_H2H']]
    exclude_cols = ['RANK_DIFF', 'POINTS_DIFF', 'UNCERTAINTY']
    available_features = [col for col in feature_cols if col not in exclude_cols]
    
    print(f"Available features: {len(available_features)}")
    
    # ============================================
    # 2. SPLIT DATA
    # ============================================
    print("\n" + "=" * 70)
    print("2. SPLITTING DATA")
    print("=" * 70)
    
    df_features['match_date'] = pd.to_datetime(df_features['match_date'])
    
    train_mask = df_features['match_date'].dt.year.isin([2020, 2021])
    val_mask = df_features['match_date'].dt.year == 2022
    test_mask = df_features['match_date'].dt.year.isin([2023, 2024])
    
    train_df = df_features[train_mask].copy()
    val_df = df_features[val_mask].copy()
    test_df = df_features[test_mask].copy()
    
    print(f"Train: {len(train_df):,} (2020-2021)")
    print(f"Validation: {len(val_df):,} (2022)")
    print(f"Test: {len(test_df):,} (2023-2024)")
    
    # Augment for training
    train_aug = augment_with_reverse(train_df, available_features)
    val_aug = augment_with_reverse(val_df, available_features)
    test_aug = augment_with_reverse(test_df, available_features)
    
    print(f"After augmentation - Train: {len(train_aug):,}, Val: {len(val_aug):,}, Test: {len(test_aug):,}")
    
    # ============================================
    # 3. TRAIN LOGISTIC REGRESSION
    # ============================================
    print("\n" + "=" * 70)
    print("3. TRAINING LOGISTIC REGRESSION")
    print("=" * 70)
    
    # Forward feature selection
    selected_features = forward_feature_selection(
        train_aug, val_aug,
        available_features=available_features,
        max_features=12,
        C=1.0,
        optimize_metric='combined'
    )
    
    print(f"\nSelected {len(selected_features)} features: {selected_features}")
    
    # Hyperparameter tuning
    print("\nTuning regularization (C)...")
    best_C = 1.0
    best_ll = float('inf')
    
    for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
        model = SymmetricLogisticRegression(C=C)
        model.fit(train_aug, selected_features)
        val_probs = model.predict_proba(val_aug)
        ll = log_loss(val_aug['winner'].values, val_probs)
        print(f"  C={C}: log_loss={ll:.4f}")
        if ll < best_ll:
            best_ll = ll
            best_C = C
    
    print(f"\nBest C: {best_C}")
    
    # Train final model
    lr_model = SymmetricLogisticRegression(C=best_C)
    lr_model.fit(train_aug, selected_features)
    
    print("\nFeature weights:")
    for feat, weight in sorted(lr_model.get_feature_weights().items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:30s}: {weight:+.4f}")
    
    # ============================================
    # 4. TRAIN NEURAL NETWORK ENSEMBLE
    # ============================================
    print("\n" + "=" * 70)
    print("4. TRAINING NEURAL NETWORK ENSEMBLE")
    print("=" * 70)
    
    # Prepare data
    X_train = train_aug[available_features].values
    X_val = val_aug[available_features].values
    X_test = test_aug[available_features].values
    y_train = train_aug['winner'].values
    y_val = val_aug['winner'].values
    y_test = test_aug['winner'].values
    
    # Scale features
    nn_scaler = StandardScaler()
    X_train_scaled = nn_scaler.fit_transform(X_train)
    X_val_scaled = nn_scaler.transform(X_val)
    X_test_scaled = nn_scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    print("Training 20-model ensemble...")
    nn_models = train_nn_ensemble(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, n_models=20)
    
    # ============================================
    # 5. EVALUATE ALL MODELS
    # ============================================
    print("\n" + "=" * 70)
    print("5. EVALUATING ALL MODELS ON TEST SET")
    print("=" * 70)
    
    results = []
    
    # Logistic Regression
    lr_probs = lr_model.predict_proba(test_aug)
    lr_results = evaluate_model(lr_probs, y_test, 'Logistic Regression')
    results.append(lr_results)
    print(f"\nLogistic Regression:")
    print(f"  Accuracy: {lr_results['accuracy']:.4f}")
    print(f"  Log Loss: {lr_results['log_loss']:.4f}")
    print(f"  ROI: {lr_results['roi']*100:.2f}%")
    
    # Neural Network Ensemble
    nn_probs = ensemble_predict(nn_models, X_test_tensor)
    nn_results = evaluate_model(nn_probs, y_test, 'Neural Network')
    results.append(nn_results)
    print(f"\nNeural Network Ensemble:")
    print(f"  Accuracy: {nn_results['accuracy']:.4f}")
    print(f"  Log Loss: {nn_results['log_loss']:.4f}")
    print(f"  ROI: {nn_results['roi']*100:.2f}%")
    
    # Markov Model (evaluate on non-augmented test set)
    print("\nEvaluating Markov Model...")
    markov_model = HierarchicalTennisModel('tennis_data.db')
    markov_probs = []
    markov_actuals = []
    
    for idx, row in test_df.iterrows():
        try:
            result = markov_model.predict_match(
                int(row['player1_id']),
                int(row['player2_id']),
                row['surface'],
                num_sets=3,
                match_date=row['match_date'].strftime('%Y-%m-%d')
            )
            markov_probs.append(result['p_player1_win'])
            markov_actuals.append(1)  # Player 1 is always the winner
        except Exception as e:
            continue
    
    if len(markov_probs) > 0:
        markov_results = evaluate_model(np.array(markov_probs), np.array(markov_actuals), 'Markov Model')
        results.append(markov_results)
        print(f"\nMarkov Model:")
        print(f"  Accuracy: {markov_results['accuracy']:.4f}")
        print(f"  Log Loss: {markov_results['log_loss']:.4f}")
        print(f"  ROI: {markov_results['roi']*100:.2f}%")
    
    # Meta-Ensemble (weighted average)
    print("\nCreating Meta-Ensemble...")
    weights = {'lr': 0.35, 'nn': 0.45, 'markov': 0.20}
    
    # For matches where we have all predictions
    meta_probs = weights['lr'] * lr_probs + weights['nn'] * nn_probs
    meta_results = evaluate_model(meta_probs, y_test, 'Meta-Ensemble (LR+NN)')
    results.append(meta_results)
    print(f"\nMeta-Ensemble:")
    print(f"  Accuracy: {meta_results['accuracy']:.4f}")
    print(f"  Log Loss: {meta_results['log_loss']:.4f}")
    print(f"  ROI: {meta_results['roi']*100:.2f}%")
    
    # ============================================
    # 6. SAVE MODELS
    # ============================================
    print("\n" + "=" * 70)
    print("6. SAVING MODELS")
    print("=" * 70)
    
    os.makedirs('ml_models', exist_ok=True)
    
    # Save Logistic Regression
    lr_data = {
        'model': lr_model,
        'selected_features': selected_features,
        'best_C': best_C,
        'metrics': lr_results,
        'training_date': datetime.now().isoformat()
    }
    with open('ml_models/logistic_regression_trained.pkl', 'wb') as f:
        pickle.dump(lr_data, f)
    print("✅ Saved: ml_models/logistic_regression_trained.pkl")
    
    # Save Neural Network Ensemble
    nn_data = {
        'models': [model.state_dict() for model in nn_models],
        'scaler': nn_scaler,
        'features': available_features,
        'hidden_dim': 100,
        'n_models': len(nn_models),
        'metrics': nn_results,
        'training_date': datetime.now().isoformat()
    }
    with open('ml_models/neural_network_ensemble.pkl', 'wb') as f:
        pickle.dump(nn_data, f)
    print("✅ Saved: ml_models/neural_network_ensemble.pkl")
    
    # ============================================
    # 7. SUMMARY
    # ============================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('model_evaluation_results.csv', index=False)
    print("\n✅ Results saved to model_evaluation_results.csv")
    
    print("\n" + "=" * 70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Cleanup
    feature_extractor.close()
    conn.close()
    
    return results_df


if __name__ == "__main__":
    main()
