#!/usr/bin/env python3
"""
🧠 ADVANCED MODEL TRAINING WITH SPECIAL PARAMETERS
===================================================
Train ML models using comprehensive ATP/WTA data (2000-2026) with:
- Traditional features (serve %, return %, ranking)
- Special parameters (momentum, clutch, surface mastery)
- Temporal features (recent form, career progression)
- Contextual features (tournament level, opponent quality)

Models trained:
1. Logistic Regression (baseline + interpretability)
2. Gradient Boosting (XGBoost - best performance)
3. Neural Network (deep patterns)
4. Ensemble (weighted combination)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedTennisTrainer:
    """Train advanced ML models with special parameters"""
    
    def __init__(self, db_path='tennis_betting.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.scaler = StandardScaler()
        
    def load_training_data(self, min_date='2010-01-01'):
        """
        Load matches with comprehensive features
        Only use data from min_date onwards for training (more reliable stats)
        """
        print("="*70)
        print("📥 LOADING TRAINING DATA")
        print("="*70)
        
        query = """
        SELECT 
            m.match_id,
            m.tourney_date,
            m.surface,
            m.tourney_level,
            m.round,
            m.winner_id,
            m.loser_id,
            m.winner_rank,
            m.loser_rank,
            m.winner_rank_points,
            m.loser_rank_points,
            m.winner_age,
            m.loser_age,
            m.w_svpt,
            m.w_1stIn,
            m.w_1stWon,
            m.w_2ndWon,
            m.w_bpSaved,
            m.w_bpFaced,
            m.l_svpt,
            m.l_1stIn,
            m.l_1stWon,
            m.l_2ndWon,
            m.l_bpSaved,
            m.l_bpFaced,
            
            -- Winner special parameters
            sp1.career_win_rate as w_career_wr,
            sp1.momentum_score as w_momentum,
            sp1.best_surface as w_best_surface,
            sp1.surface_mastery as w_surface_mastery,
            sp1.clutch_performance as w_clutch,
            sp1.bp_defense_rate as w_bp_defense,
            sp1.first_serve_win_pct as w_1st_srv_win,
            sp1.consistency_rating as w_consistency,
            sp1.peak_rating as w_peak_rating,
            
            -- Loser special parameters
            sp2.career_win_rate as l_career_wr,
            sp2.momentum_score as l_momentum,
            sp2.best_surface as l_best_surface,
            sp2.surface_mastery as l_surface_mastery,
            sp2.clutch_performance as l_clutch,
            sp2.bp_defense_rate as l_bp_defense,
            sp2.first_serve_win_pct as l_1st_srv_win,
            sp2.consistency_rating as l_consistency,
            sp2.peak_rating as l_peak_rating
            
        FROM matches m
        LEFT JOIN special_parameters sp1 ON m.winner_id = sp1.player_id
        LEFT JOIN special_parameters sp2 ON m.loser_id = sp2.player_id
        WHERE m.tourney_date >= ?
            AND m.w_svpt IS NOT NULL
            AND m.l_svpt IS NOT NULL
            AND sp1.player_id IS NOT NULL
            AND sp2.player_id IS NOT NULL
        ORDER BY m.tourney_date
        """
        
        df = pd.read_sql_query(query, self.conn, params=[min_date])
        print(f"\n✅ Loaded {len(df):,} matches from {min_date}")
        print(f"📅 Date range: {df['tourney_date'].min()} to {df['tourney_date'].max()}")
        
        return df
    
    def engineer_features(self, df):
        """
        Create comprehensive feature set combining:
        - Match statistics
        - Special parameters
        - Derived features
        """
        print("\n🔧 ENGINEERING FEATURES")
        print("="*70)
        
        features_df = df.copy()
        
        # 1. SERVE & RETURN PERCENTAGES
        features_df['w_serve_win_pct'] = (
            (features_df['w_1stWon'] + features_df['w_2ndWon']) / features_df['w_svpt']
        ).fillna(0.65)
        
        features_df['l_serve_win_pct'] = (
            (features_df['l_1stWon'] + features_df['l_2ndWon']) / features_df['l_svpt']
        ).fillna(0.65)
        
        # Return win % is opponent's serve loss %
        features_df['w_return_win_pct'] = 1 - features_df['l_serve_win_pct']
        features_df['l_return_win_pct'] = 1 - features_df['w_serve_win_pct']
        
        # 2. RANKING FEATURES
        features_df['rank_diff'] = features_df['loser_rank'] - features_df['winner_rank']
        features_df['rank_ratio'] = np.log1p(features_df['winner_rank']) / np.log1p(features_df['loser_rank'])
        
        # 3. RANKING POINTS DIFFERENTIAL
        features_df['rank_points_diff'] = (
            features_df['winner_rank_points'] - features_df['loser_rank_points']
        ).fillna(0)
        
        # 4. AGE FEATURES
        features_df['age_diff'] = (features_df['winner_age'] - features_df['loser_age']).fillna(0)
        features_df['w_age_exp'] = (features_df['winner_age'] - 25).clip(-5, 10) / 10  # Peak around 25
        features_df['l_age_exp'] = (features_df['loser_age'] - 25).clip(-5, 10) / 10
        
        # 5. SPECIAL PARAMETER DIFFERENTIALS
        features_df['momentum_diff'] = features_df['w_momentum'] - features_df['l_momentum']
        features_df['clutch_diff'] = features_df['w_clutch'] - features_df['l_clutch']
        features_df['consistency_diff'] = features_df['w_consistency'] - features_df['l_consistency']
        features_df['peak_rating_diff'] = features_df['w_peak_rating'] - features_df['l_peak_rating']
        
        # 6. SURFACE ADVANTAGE
        features_df['w_surface_adv'] = (
            features_df['w_best_surface'] == features_df['surface']
        ).astype(int)
        features_df['l_surface_adv'] = (
            features_df['l_best_surface'] == features_df['surface']
        ).astype(int)
        features_df['surface_adv_diff'] = features_df['w_surface_adv'] - features_df['l_surface_adv']
        
        # 7. TOURNAMENT IMPORTANCE (encoding)
        importance_map = {'G': 4, 'F': 3, 'M': 3, 'A': 2, 'D': 1, 'C': 1}
        features_df['tourney_importance'] = features_df['tourney_level'].map(importance_map).fillna(1)
        
        # 8. BREAK POINT FEATURES
        features_df['w_bp_save_rate'] = (
            features_df['w_bpSaved'] / features_df['w_bpFaced']
        ).fillna(0.5)
        features_df['l_bp_save_rate'] = (
            features_df['l_bpSaved'] / features_df['l_bpFaced']
        ).fillna(0.5)
        
        # 9. COMBINED QUALITY SCORE
        features_df['w_quality'] = (
            features_df['w_career_wr'] * 0.3 +
            features_df['w_momentum'] * 0.3 +
            features_df['w_peak_rating'] * 0.2 +
            features_df['w_clutch'] * 0.2
        )
        features_df['l_quality'] = (
            features_df['l_career_wr'] * 0.3 +
            features_df['l_momentum'] * 0.3 +
            features_df['l_peak_rating'] * 0.2 +
            features_df['l_clutch'] * 0.2
        )
        features_df['quality_diff'] = features_df['w_quality'] - features_df['l_quality']
        
        # 10. SURFACE-SPECIFIC ENCODING
        surface_dummies = pd.get_dummies(features_df['surface'], prefix='surface')
        features_df = pd.concat([features_df, surface_dummies], axis=1)
        
        print(f"✅ Created {len(features_df.columns)} total columns")
        
        return features_df
    
    def prepare_training_set(self, features_df):
        """Prepare X and y for training"""
        print("\n📊 PREPARING TRAINING SET")
        print("="*70)
        
        # Feature columns to use
        feature_cols = [
            # Serve & Return
            'w_serve_win_pct', 'l_serve_win_pct',
            'w_return_win_pct', 'l_return_win_pct',
            
            # Rankings
            'rank_diff', 'rank_ratio', 'rank_points_diff',
            
            # Age
            'age_diff', 'w_age_exp', 'l_age_exp',
            
            # Special parameters (raw)
            'w_momentum', 'l_momentum',
            'w_clutch', 'l_clutch',
            'w_consistency', 'l_consistency',
            'w_peak_rating', 'l_peak_rating',
            'w_career_wr', 'l_career_wr',
            
            # Special parameters (differentials)
            'momentum_diff', 'clutch_diff', 'consistency_diff',
            'peak_rating_diff', 'quality_diff',
            
            # Surface
            'surface_adv_diff', 'w_surface_mastery', 'l_surface_mastery',
            
            # Tournament
            'tourney_importance',
            
            # Break points
            'w_bp_save_rate', 'l_bp_save_rate'
        ]
        
        # Add surface dummies
        surface_cols = [col for col in features_df.columns if col.startswith('surface_')]
        feature_cols.extend(surface_cols)
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in features_df.columns]
        
        X = features_df[available_cols].fillna(0)
        
        # Create target: simulate each match twice (winner and loser perspective)
        # This doubles training data and balances classes
        X_doubled = pd.concat([X, X], ignore_index=True)
        
        # First half: winner perspective (target=1)
        # Second half: loser perspective (target=0, features swapped)
        y = np.concatenate([
            np.ones(len(X)),   # Winner won
            np.zeros(len(X))   # Loser lost
        ])
        
        # For second half, swap winner/loser features
        second_half_start = len(X)
        for col in available_cols:
            if col.startswith('w_'):
                loser_col = 'l_' + col[2:]
                if loser_col in available_cols:
                    X_doubled.iloc[second_half_start:, X_doubled.columns.get_loc(col)] = X[loser_col].values
                    X_doubled.iloc[second_half_start:, X_doubled.columns.get_loc(loser_col)] = X[col].values
            elif col.endswith('_diff'):
                # Flip differential features
                X_doubled.iloc[second_half_start:, X_doubled.columns.get_loc(col)] = -X[col].values
        
        print(f"✅ Training set shape: {X_doubled.shape}")
        print(f"   Features: {len(available_cols)}")
        print(f"   Samples: {len(X_doubled):,} (doubled from {len(X):,})")
        print(f"   Class balance: {y.mean():.1%} positive")
        
        return X_doubled, y, available_cols
    
    def train_models(self, X, y, feature_names):
        """Train multiple models"""
        print("\n🎓 TRAINING MODELS")
        print("="*70)
        
        # Split data: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {}
        results = {}
        
        # 1. LOGISTIC REGRESSION
        print("\n1️⃣  Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        lr.fit(X_train_scaled, y_train)
        
        y_pred_lr = lr.predict(X_test_scaled)
        y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
        
        models['logistic_regression'] = lr
        results['logistic_regression'] = {
            'accuracy': accuracy_score(y_test, y_pred_lr),
            'log_loss': log_loss(y_test, y_prob_lr),
            'roc_auc': roc_auc_score(y_test, y_prob_lr)
        }
        
        print(f"   Accuracy: {results['logistic_regression']['accuracy']:.4f}")
        print(f"   Log Loss: {results['logistic_regression']['log_loss']:.4f}")
        print(f"   ROC AUC: {results['logistic_regression']['roc_auc']:.4f}")
        
        # Feature importance for LR
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': lr.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\n   Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"     {row['feature']}: {row['coefficient']:.4f}")
        
        # 2. RANDOM FOREST
        print("\n2️⃣  Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        rf_model.fit(X_train, y_train)
        
        y_pred_rf = rf_model.predict(X_test)
        y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
        
        models['random_forest'] = rf_model
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'log_loss': log_loss(y_test, y_prob_rf),
            'roc_auc': roc_auc_score(y_test, y_prob_rf)
        }
        
        print(f"   Accuracy: {results['random_forest']['accuracy']:.4f}")
        print(f"   Log Loss: {results['random_forest']['log_loss']:.4f}")
        print(f"   ROC AUC: {results['random_forest']['roc_auc']:.4f}")
        
        # Random Forest feature importance
        rf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n   Top 10 Most Important Features:")
        for idx, row in rf_importance.head(10).iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")
        
        # 3. ENSEMBLE (Just LR + RF for speed)
        print("\n3️⃣  Creating Ensemble (LR + RF)...")
        y_prob_ensemble = 0.4 * y_prob_lr + 0.6 * y_prob_rf
        y_pred_ensemble = (y_prob_ensemble >= 0.5).astype(int)
        
        results['ensemble'] = {
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'log_loss': log_loss(y_test, y_prob_ensemble),
            'roc_auc': roc_auc_score(y_test, y_prob_ensemble)
        }
        
        print(f"   Accuracy: {results['ensemble']['accuracy']:.4f}")
        print(f"   Log Loss: {results['ensemble']['log_loss']:.4f}")
        print(f"   ROC AUC: {results['ensemble']['roc_auc']:.4f}")
        
        # Save models
        print("\n💾 Saving models...")
        joblib.dump(lr, 'ml_models/logistic_regression_advanced.pkl')
        joblib.dump(rf_model, 'ml_models/random_forest_advanced.pkl')
        joblib.dump(self.scaler, 'ml_models/scaler_advanced.pkl')
        joblib.dump(feature_names, 'ml_models/feature_names_advanced.pkl')
        
        print("✅ Models saved to ml_models/")
        
        return models, results, feature_importance, rf_importance
    
    def display_summary(self, results):
        """Display training summary"""
        print("\n" + "="*70)
        print("📊 TRAINING RESULTS SUMMARY")
        print("="*70)
        
        summary_df = pd.DataFrame(results).T
        summary_df = summary_df.sort_values('roc_auc', ascending=False)
        
        print("\n🏆 Model Performance (sorted by ROC AUC):")
        print(summary_df.to_string())
        
        best_model = summary_df.index[0]
        print(f"\n✨ Best Model: {best_model.upper()}")
        print(f"   ROC AUC: {summary_df.loc[best_model, 'roc_auc']:.4f}")
        print(f"   Accuracy: {summary_df.loc[best_model, 'accuracy']:.4f}")
        print(f"   Log Loss: {summary_df.loc[best_model, 'log_loss']:.4f}")
    
    def run(self):
        """Execute full training pipeline"""
        start_time = datetime.now()
        
        print("\n" + "="*70)
        print("🚀 ADVANCED TENNIS ML TRAINING PIPELINE")
        print("="*70)
        
        # 1. Load data
        df = self.load_training_data(min_date='2010-01-01')
        
        # 2. Engineer features
        features_df = self.engineer_features(df)
        
        # 3. Prepare training set
        X, y, feature_names = self.prepare_training_set(features_df)
        
        # 4. Train models
        models, results, lr_importance, rf_importance = self.train_models(X, y, feature_names)
        
        # 5. Display summary
        self.display_summary(results)
        
        elapsed = datetime.now() - start_time
        print("\n" + "="*70)
        print(f"✅ TRAINING COMPLETE! (Elapsed: {elapsed})")
        print("="*70)
        
        print("\n📁 Output files:")
        print("   - ml_models/logistic_regression_advanced.pkl")
        print("   - ml_models/random_forest_advanced.pkl")
        print("   - ml_models/scaler_advanced.pkl")
        print("   - ml_models/feature_names_advanced.pkl")
        
        print("\n🎯 Next steps:")
        print("   1. Test models on live matches")
        print("   2. Integrate with dashboard")
        print("   3. Monitor performance and retrain monthly")
        
        self.conn.close()


if __name__ == '__main__':
    trainer = AdvancedTennisTrainer(db_path='tennis_betting.db')
    trainer.run()
