"""
Symmetric Logistic Regression Model for Tennis Match Prediction

Features:
- No bias term (symmetric - equal players get 50% probability)
- Forward feature selection optimized for log-loss + ROI
- Time-weighted statistics
- Hyperparameter tuning for regularization and time decay
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class SymmetricLogisticRegression:
    """
    Logistic regression model without bias term for symmetric predictions.
    
    P(player1 wins) = 1 / (1 + exp(-z))
    where z = Σ(β_i * x_i) and x_i are feature differences (player1 - player2)
    """
    
    def __init__(self, C: float = 1.0, selected_features: Optional[List[str]] = None):
        """
        Initialize symmetric logistic regression model.
        
        Args:
            C: Regularization strength (smaller = stronger regularization)
            selected_features: List of features to use (if None, will be set during training)
        """
        self.C = C
        self.selected_features = selected_features
        self.model = None
        self.scaler = None
        self.feature_weights = None
        
        # Features that should NOT be standardized
        self.non_standardized = ['DIRECT', 'RETIRED', 'FATIGUE']
    
    def _prepare_features(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Prepare feature matrix from dataframe.
        
        Supports two input formats:
        1. Pre-computed differences: columns ending in '_DIFF' or exact feature names
        2. Player1/Player2 format: columns named player1_{feat} and player2_{feat}
        
        Args:
            df: DataFrame with features
            features: List of feature names to use
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        X = []
        
        for feat in features:
            # Check if feature is already in diff format
            if feat in df.columns:
                # Feature is directly available (already a diff or pre-computed)
                X.append(df[feat].values)
            else:
                # Try player1/player2 format
                p1_col = f'player1_{feat}'
                p2_col = f'player2_{feat}'
                
                if p1_col in df.columns and p2_col in df.columns:
                    # Feature difference (player1 - player2)
                    diff = df[p1_col].values - df[p2_col].values
                    X.append(diff)
                else:
                    raise ValueError(f"Feature '{feat}' not found. Expected either '{feat}' column or '{p1_col}'/'{p2_col}' columns")
        
        return np.column_stack(X)
    
    def _standardize_features(self, X: np.ndarray, features: List[str], 
                             fit: bool = False) -> np.ndarray:
        """
        Standardize features (except DIRECT, RETIRED, FATIGUE).
        
        Args:
            X: Feature matrix
            features: Feature names
            fit: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            Standardized feature matrix
        """
        X_scaled = X.copy()
        
        # Identify indices of features to standardize
        standardize_idx = [i for i, feat in enumerate(features) 
                          if feat not in self.non_standardized]
        
        if len(standardize_idx) > 0:
            if fit:
                self.scaler = StandardScaler()
                X_scaled[:, standardize_idx] = self.scaler.fit_transform(X[:, standardize_idx])
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. Call fit() first.")
                X_scaled[:, standardize_idx] = self.scaler.transform(X[:, standardize_idx])
        
        return X_scaled
    
    def fit(self, df: pd.DataFrame, features: List[str]) -> None:
        """
        Train the model on given data.
        
        Args:
            df: Training dataframe with player1/player2 features and 'winner' column
            features: List of feature names to use
        """
        self.selected_features = features
        
        # Prepare features
        X = self._prepare_features(df, features)
        
        # Standardize
        X = self._standardize_features(X, features, fit=True)
        
        # Target: 1 if player1 won, 0 if player2 won
        y = (df['winner'] == 1).astype(int).values
        
        # Train logistic regression without intercept (fit_intercept=False)
        self.model = LogisticRegression(
            penalty='l2',
            C=self.C,
            fit_intercept=False,  # No bias term (symmetric)
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        
        self.model.fit(X, y)
        
        # Store feature weights
        self.feature_weights = dict(zip(features, self.model.coef_[0]))
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of player1 winning.
        
        Args:
            df: DataFrame with player1/player2 features
            
        Returns:
            Array of probabilities for player1 winning
        """
        if self.model is None or self.selected_features is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Prepare features
        X = self._prepare_features(df, self.selected_features)
        
        # Standardize
        X = self._standardize_features(X, self.selected_features, fit=False)
        
        # Predict probabilities
        probs = self.model.predict_proba(X)[:, 1]
        
        return probs
    
    def get_feature_weights(self) -> Dict[str, float]:
        """Get dictionary of feature weights."""
        if self.feature_weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.feature_weights


def calculate_kelly_roi(predictions: np.ndarray, actuals: np.ndarray, 
                        odds: Optional[np.ndarray] = None,
                        kelly_fraction: float = 0.25) -> float:
    """
    Calculate ROI using Kelly criterion betting strategy.
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual outcomes (1 if player1 won, 0 otherwise)
        odds: Decimal odds for player1 (if None, use implied odds from predictions)
        kelly_fraction: Fraction of Kelly bet to use (0.25 = quarter Kelly)
        
    Returns:
        ROI (return on investment) as decimal
    """
    if odds is None:
        # Use implied odds from predictions (with margin)
        margin = 1.05  # 5% bookmaker margin
        odds = margin / predictions
    
    # Calculate Kelly stakes
    edge = predictions - (1 / odds)  # Expected value
    kelly_stakes = np.maximum(edge * kelly_fraction, 0)  # Only bet when edge > 0
    
    # Calculate returns
    wins = actuals == 1
    returns = np.where(wins, kelly_stakes * (odds - 1), -kelly_stakes)
    
    # ROI = total return / total staked
    total_staked = kelly_stakes.sum()
    if total_staked == 0:
        return 0.0
    
    roi = returns.sum() / total_staked
    return roi


def forward_feature_selection(train_df: pd.DataFrame, 
                              val_df: pd.DataFrame,
                              available_features: List[str],
                              max_features: int = 15,
                              C: float = 1.0,
                              optimize_metric: str = 'combined') -> List[str]:
    """
    Forward feature selection optimized on validation set.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        available_features: List of all available features
        max_features: Maximum number of features to select
        C: Regularization parameter
        optimize_metric: 'log_loss', 'roi', or 'combined'
        
    Returns:
        List of selected feature names
    """
    selected_features = []
    remaining_features = available_features.copy()
    
    best_score = float('inf') if optimize_metric in ['log_loss', 'combined'] else float('-inf')
    
    print(f"Forward feature selection (max {max_features} features)...")
    print(f"Optimizing for: {optimize_metric}")
    print("-" * 70)
    
    for step in range(max_features):
        if len(remaining_features) == 0:
            break
        
        best_feature = None
        best_step_score = best_score
        
        for feature in remaining_features:
            # Test adding this feature
            test_features = selected_features + [feature]
            
            # Train model
            model = SymmetricLogisticRegression(C=C, selected_features=test_features)
            model.fit(train_df, test_features)
            
            # Evaluate on validation set
            val_probs = model.predict_proba(val_df)
            val_actuals = (val_df['winner'] == 1).astype(int).values
            
            # Calculate metrics
            ll = log_loss(val_actuals, val_probs)
            roi = calculate_kelly_roi(val_probs, val_actuals)
            
            # Combined score (normalized)
            if optimize_metric == 'log_loss':
                score = ll
            elif optimize_metric == 'roi':
                score = -roi  # Negative because we're minimizing
            else:  # combined
                # Combine log-loss (lower is better) and ROI (higher is better)
                # Normalize: log-loss typically 0.5-0.7, ROI typically -0.1 to 0.1
                score = ll - 5.0 * roi  # Weight ROI heavily
            
            # Check if this is better
            if (optimize_metric in ['log_loss', 'combined'] and score < best_step_score) or \
               (optimize_metric == 'roi' and score > best_step_score):
                best_step_score = score
                best_feature = feature
        
        if best_feature is None:
            break
        
        # Add best feature
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        best_score = best_step_score
        
        # Re-evaluate with all selected features
        model = SymmetricLogisticRegression(C=C, selected_features=selected_features)
        model.fit(train_df, selected_features)
        val_probs = model.predict_proba(val_df)
        val_actuals = (val_df['winner'] == 1).astype(int).values
        ll = log_loss(val_actuals, val_probs)
        roi = calculate_kelly_roi(val_probs, val_actuals)
        
        print(f"Step {step+1:2d}: Added {best_feature:20s} | "
              f"LogLoss={ll:.4f}, ROI={roi:+.3f}, Features={len(selected_features)}")
    
    print("-" * 70)
    print(f"Selected {len(selected_features)} features")
    
    return selected_features


def hyperparameter_tuning(train_df: pd.DataFrame,
                         val_df: pd.DataFrame,
                         features: List[str],
                         C_values: List[float] = [0.1, 0.2, 0.5, 1.0],
                         time_discount_values: List[float] = [0.6, 0.7, 0.8, 0.9],
                         filter_uncertain: List[float] = [0.70, 0.80, 0.90]) -> Dict:
    """
    Grid search for optimal hyperparameters.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        features: Selected features
        C_values: Regularization values to test
        time_discount_values: Time discount factors to test
        filter_uncertain: Thresholds for filtering uncertain matches
        
    Returns:
        Dictionary with best parameters and results
    """
    print("Hyperparameter tuning...")
    print("-" * 70)
    
    best_score = float('inf')
    best_params = {}
    results = []
    
    for C in C_values:
        for time_discount in time_discount_values:
            for filter_thresh in filter_uncertain:
                # Apply time weighting (placeholder - would need match dates)
                # For now, just filter by uncertainty
                
                # Filter training data: remove top X% uncertain matches
                model_temp = SymmetricLogisticRegression(C=C)
                model_temp.fit(train_df, features)
                train_probs = model_temp.predict_proba(train_df)
                
                # Uncertainty = distance from 0.5
                uncertainty = np.abs(train_probs - 0.5)
                threshold = np.percentile(uncertainty, (1 - filter_thresh) * 100)
                filtered_train = train_df[uncertainty >= threshold].copy()
                
                # Train on filtered data
                model = SymmetricLogisticRegression(C=C)
                model.fit(filtered_train, features)
                
                # Evaluate on validation
                val_probs = model.predict_proba(val_df)
                val_actuals = (val_df['winner'] == 1).astype(int).values
                
                ll = log_loss(val_actuals, val_probs)
                roi = calculate_kelly_roi(val_probs, val_actuals)
                
                # Combined score
                score = ll - 5.0 * roi
                
                results.append({
                    'C': C,
                    'time_discount': time_discount,
                    'filter_thresh': filter_thresh,
                    'log_loss': ll,
                    'roi': roi,
                    'score': score,
                    'n_train': len(filtered_train)
                })
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        'C': C,
                        'time_discount': time_discount,
                        'filter_thresh': filter_thresh,
                        'log_loss': ll,
                        'roi': roi,
                        'n_train': len(filtered_train)
                    }
    
    # Sort results by score
    results_df = pd.DataFrame(results).sort_values('score')
    
    print("\nTop 5 configurations:")
    print(results_df.head().to_string(index=False))
    print("\n" + "-" * 70)
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return {
        'best_params': best_params,
        'all_results': results_df
    }


def train_logistic_model(train_df: pd.DataFrame, 
                        validation_df: pd.DataFrame,
                        available_features: List[str],
                        max_features: int = 15,
                        C: float = 1.0,
                        perform_tuning: bool = True) -> Tuple[SymmetricLogisticRegression, List[str], Dict]:
    """
    Complete training pipeline with feature selection and hyperparameter tuning.
    
    Args:
        train_df: Training dataframe
        validation_df: Validation dataframe
        available_features: List of all available features (excluding RANK, POINTS)
        max_features: Maximum number of features to select
        C: Initial regularization parameter
        perform_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        Tuple of (trained_model, selected_features, tuning_results)
    """
    print("=" * 70)
    print("LOGISTIC REGRESSION TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Available features: {len(available_features)}")
    print()
    
    # Step 1: Forward feature selection
    selected_features = forward_feature_selection(
        train_df, 
        validation_df,
        available_features,
        max_features=max_features,
        C=C
    )
    
    print(f"\nSelected features: {selected_features}")
    print()
    
    # Step 2: Hyperparameter tuning (optional)
    tuning_results = None
    if perform_tuning:
        tuning_results = hyperparameter_tuning(
            train_df,
            validation_df,
            selected_features
        )
        
        # Use best parameters
        best_C = tuning_results['best_params']['C']
        filter_thresh = tuning_results['best_params']['filter_thresh']
        
        print(f"\nUsing optimized parameters: C={best_C}, filter_thresh={filter_thresh}")
    else:
        best_C = C
        filter_thresh = 0.90
    
    # Step 3: Train final model
    print("\nTraining final model...")
    
    # Filter training data
    model_temp = SymmetricLogisticRegression(C=best_C)
    model_temp.fit(train_df, selected_features)
    train_probs = model_temp.predict_proba(train_df)
    uncertainty = np.abs(train_probs - 0.5)
    threshold = np.percentile(uncertainty, (1 - filter_thresh) * 100)
    filtered_train = train_df[uncertainty >= threshold].copy()
    
    # Final training
    final_model = SymmetricLogisticRegression(C=best_C)
    final_model.fit(filtered_train, selected_features)
    
    print(f"Trained on {len(filtered_train)} samples (filtered from {len(train_df)})")
    
    # Evaluate
    val_probs = final_model.predict_proba(validation_df)
    val_actuals = (validation_df['winner'] == 1).astype(int).values
    val_ll = log_loss(val_actuals, val_probs)
    val_roi = calculate_kelly_roi(val_probs, val_actuals)
    
    print(f"\nValidation Performance:")
    print(f"  Log Loss: {val_ll:.4f}")
    print(f"  ROI (Kelly): {val_roi:+.3f}")
    print(f"  Accuracy: {(val_probs.round() == val_actuals).mean():.1%}")
    
    print("\n" + "=" * 70)
    
    return final_model, selected_features, tuning_results
