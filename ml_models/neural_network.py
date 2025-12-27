"""
Neural Network Model for Tennis Match Prediction with Bagging

Architecture:
- Input layer: 20 features (excluding RANK, POINTS)
- Hidden layer: 100 neurons, tanh activation
- Output layer: 1 neuron, sigmoid activation
- NO bias neurons (maintains symmetry)

Training:
- SGD with momentum=0.55
- Learning rate=0.0004
- Weight decay=0.002 (L2 regularization)
- Online learning (batch_size=1)
- Early stopping (patience=10)

Bagging:
- Train 20 neural networks on bootstrap samples
- Final prediction: Average of all networks
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SymmetricNeuralNetwork(nn.Module):
    """
    Neural network without bias terms for symmetric predictions.
    
    Architecture:
    - Input: n_features
    - Hidden: 100 neurons with tanh activation
    - Output: 1 neuron with sigmoid activation
    - No bias terms in any layer
    """
    
    def __init__(self, n_features: int = 20):
        super(SymmetricNeuralNetwork, self).__init__()
        
        # Hidden layer: input -> 100 neurons (no bias)
        self.hidden = nn.Linear(n_features, 100, bias=False)
        
        # Output layer: 100 -> 1 (no bias)
        self.output = nn.Linear(100, 1, bias=False)
        
        # Initialize weights (Xavier initialization works well with tanh)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Hidden layer with tanh activation
        h = torch.tanh(self.hidden(x))
        
        # Output layer with sigmoid activation
        out = torch.sigmoid(self.output(h))
        
        return out


class NeuralNetworkTrainer:
    """
    Trainer for symmetric neural network with early stopping.
    """
    
    def __init__(self, 
                 n_features: int = 20,
                 learning_rate: float = 0.0004,
                 momentum: float = 0.55,
                 weight_decay: float = 0.002,
                 patience: int = 10,
                 verbose: bool = True):
        """
        Initialize trainer.
        
        Args:
            n_features: Number of input features
            learning_rate: Learning rate for SGD
            momentum: Momentum for SGD
            weight_decay: L2 regularization strength
            patience: Early stopping patience (epochs)
            verbose: Print training progress
        """
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.patience = patience
        self.verbose = verbose
        
        self.model = None
        self.scaler = None
        self.optimizer = None
        self.criterion = None
        self.history = {'train_loss': [], 'val_loss': []}
        
        # Features that should NOT be standardized
        self.non_standardized = ['DIRECT', 'RETIRED', 'FATIGUE']
    
    def _prepare_features(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """Prepare feature matrix from dataframe."""
        X = []
        
        for feat in features:
            p1_col = f'player1_{feat}'
            p2_col = f'player2_{feat}'
            
            if p1_col not in df.columns or p2_col not in df.columns:
                raise ValueError(f"Missing feature columns: {p1_col} or {p2_col}")
            
            # Feature difference (player1 - player2)
            diff = df[p1_col].values - df[p2_col].values
            X.append(diff)
        
        return np.column_stack(X)
    
    def _standardize_features(self, X: np.ndarray, features: List[str], 
                             fit: bool = False) -> np.ndarray:
        """Standardize features (except DIRECT, RETIRED, FATIGUE)."""
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
    
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
            features: List[str], max_epochs: int = 100) -> None:
        """
        Train the neural network with early stopping.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            features: List of feature names
            max_epochs: Maximum number of training epochs
        """
        # Prepare data
        X_train = self._prepare_features(train_df, features)
        X_train = self._standardize_features(X_train, features, fit=True)
        y_train = (train_df['winner'] == 1).astype(float).values
        
        X_val = self._prepare_features(val_df, features)
        X_val = self._standardize_features(X_val, features, fit=False)
        y_val = (val_df['winner'] == 1).astype(float).values
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
        
        # Initialize model
        self.model = SymmetricNeuralNetwork(n_features=len(features))
        
        # Loss function (Binary Cross Entropy)
        self.criterion = nn.BCELoss()
        
        # Optimizer (SGD with momentum and weight decay)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training phase (online learning - one sample at a time)
            self.model.train()
            train_losses = []
            
            # Shuffle training data
            indices = torch.randperm(len(X_train_tensor))
            
            for idx in indices:
                # Get single sample
                x_sample = X_train_tensor[idx:idx+1]
                y_sample = y_train_tensor[idx:idx+1]
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(x_sample)
                loss = self.criterion(output, y_sample)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val_tensor)
                val_loss = self.criterion(val_output, y_val_tensor).item()
            
            # Store history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{max_epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, "
                      f"Patience={patience_counter}/{self.patience}")
            
            # Early stopping
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                # Restore best model
                self.model.load_state_dict(best_model_state)
                break
        
        if self.verbose:
            print(f"Training completed. Best val loss: {best_val_loss:.4f}")
    
    def predict(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Predict probabilities for new data.
        
        Args:
            df: Dataframe with features
            features: List of feature names
            
        Returns:
            Array of predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Prepare data
        X = self._prepare_features(df, features)
        X = self._standardize_features(X, features, fit=False)
        X_tensor = torch.FloatTensor(X)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy().flatten()
        
        return predictions
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.history


def train_nn_ensemble(train_df: pd.DataFrame,
                     validation_df: pd.DataFrame,
                     features: List[str],
                     n_bags: int = 20,
                     learning_rate: float = 0.0004,
                     momentum: float = 0.55,
                     weight_decay: float = 0.002,
                     patience: int = 10,
                     max_epochs: int = 100,
                     verbose: bool = True) -> Tuple[List[NeuralNetworkTrainer], Dict]:
    """
    Train ensemble of neural networks using bagging.
    
    Args:
        train_df: Training dataframe
        validation_df: Validation dataframe
        features: List of feature names
        n_bags: Number of models in ensemble
        learning_rate: Learning rate for SGD
        momentum: Momentum for SGD
        weight_decay: L2 regularization
        patience: Early stopping patience
        max_epochs: Maximum epochs per model
        verbose: Print progress
        
    Returns:
        Tuple of (list of trained models, ensemble statistics)
    """
    print("=" * 70)
    print("NEURAL NETWORK ENSEMBLE TRAINING (BAGGING)")
    print("=" * 70)
    print(f"Number of models: {n_bags}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Features: {len(features)}")
    print(f"Learning rate: {learning_rate}")
    print(f"Momentum: {momentum}")
    print(f"Weight decay: {weight_decay}")
    print(f"Patience: {patience}")
    print("-" * 70)
    
    models = []
    ensemble_stats = {
        'train_losses': [],
        'val_losses': [],
        'final_epochs': []
    }
    
    for i in range(n_bags):
        if verbose:
            print(f"\nTraining model {i+1}/{n_bags}...")
        
        # Bootstrap sample (sample with replacement)
        bootstrap_sample = train_df.sample(n=len(train_df), replace=True)
        
        # Train individual model
        trainer = NeuralNetworkTrainer(
            n_features=len(features),
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            patience=patience,
            verbose=verbose
        )
        
        trainer.fit(bootstrap_sample, validation_df, features, max_epochs=max_epochs)
        
        # Store model
        models.append(trainer)
        
        # Store statistics
        history = trainer.get_history()
        ensemble_stats['train_losses'].append(history['train_loss'])
        ensemble_stats['val_losses'].append(history['val_loss'])
        ensemble_stats['final_epochs'].append(len(history['train_loss']))
    
    print("\n" + "=" * 70)
    print("ENSEMBLE TRAINING COMPLETED")
    print("=" * 70)
    print(f"Average epochs: {np.mean(ensemble_stats['final_epochs']):.1f}")
    print(f"Average final train loss: {np.mean([losses[-1] for losses in ensemble_stats['train_losses']]):.4f}")
    print(f"Average final val loss: {np.mean([losses[-1] for losses in ensemble_stats['val_losses']]):.4f}")
    
    return models, ensemble_stats


def predict_ensemble(models: List[NeuralNetworkTrainer],
                    df: pd.DataFrame,
                    features: List[str]) -> np.ndarray:
    """
    Predict using ensemble of models (average predictions).
    
    Args:
        models: List of trained models
        df: Dataframe to predict on
        features: List of feature names
        
    Returns:
        Array of ensemble predictions
    """
    predictions = []
    
    for model in models:
        pred = model.predict(df, features)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    
    return ensemble_pred


def calculate_permutation_importance(models: List[NeuralNetworkTrainer],
                                    val_df: pd.DataFrame,
                                    features: List[str],
                                    n_repeats: int = 5) -> pd.DataFrame:
    """
    Calculate feature importance via permutation.
    
    Args:
        models: Ensemble of trained models
        val_df: Validation dataframe
        features: List of feature names
        n_repeats: Number of permutation repeats
        
    Returns:
        DataFrame with feature importances
    """
    from sklearn.metrics import log_loss
    
    print("Calculating permutation feature importance...")
    
    # Baseline predictions
    baseline_pred = predict_ensemble(models, val_df, features)
    y_true = (val_df['winner'] == 1).astype(int).values
    baseline_loss = log_loss(y_true, baseline_pred)
    
    importances = []
    
    for feature in features:
        feature_losses = []
        
        for _ in range(n_repeats):
            # Create copy of validation data
            val_permuted = val_df.copy()
            
            # Permute this feature for both players
            p1_col = f'player1_{feature}'
            p2_col = f'player2_{feature}'
            
            val_permuted[p1_col] = np.random.permutation(val_permuted[p1_col].values)
            val_permuted[p2_col] = np.random.permutation(val_permuted[p2_col].values)
            
            # Predict with permuted data
            permuted_pred = predict_ensemble(models, val_permuted, features)
            permuted_loss = log_loss(y_true, permuted_pred)
            
            # Importance = increase in loss
            feature_losses.append(permuted_loss - baseline_loss)
        
        importances.append({
            'feature': feature,
            'importance': np.mean(feature_losses),
            'std': np.std(feature_losses)
        })
    
    importance_df = pd.DataFrame(importances)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df
