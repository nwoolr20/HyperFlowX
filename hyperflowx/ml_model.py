"""AI-powered machine learning with adaptive model selection.

This module provides intelligent model selection between traditional ML algorithms
(XGBoost) and deep learning (PyTorch neural networks) based on data characteristics.
"""

import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Union, Any, Tuple, cast


# 🚀 Simple Neural Network
class NeuralNet(nn.Module):
    """Simple feedforward neural network for regression tasks.
    
    Architecture:
        - Input layer (variable size)
        - Hidden layer 1: 128 neurons with ReLU activation
        - Hidden layer 2: 64 neurons with ReLU activation  
        - Output layer: 1 neuron (regression)
    """
    def __init__(self, input_size: int):
        """Initialize neural network.
        
        Args:
            input_size: Number of input features
        """
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 🚀 Adaptive AI Model Selection
def train_hyperflowx(
    X: Union[np.ndarray, list], 
    y: Union[np.ndarray, list], 
    use_cuda: bool = True
) -> Any:
    """Selects the best ML model dynamically based on data size.
    
    Algorithm Selection Logic:
    - Small datasets (≤50 features): XGBoost (faster training, good performance)
    - Large datasets (>50 features): Neural Network (better capacity for complex patterns)
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        use_cuda: Whether to use GPU acceleration for neural networks
        
    Returns:
        Trained model (either XGBRegressor or PyTorch model)
        
    Note:
        Model selection is based on feature count. For production use,
        consider cross-validation and performance-based selection.
    """
    # Convert inputs to numpy arrays for consistent handling
    X = cast(np.ndarray, np.asarray(X))
    y = cast(np.ndarray, np.asarray(y))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔹 If dataset is small, use **XGBoost**
    if X.shape[1] <= 50:
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
        xgb_model.fit(X_train, y_train)
        return xgb_model

    # 🔹 Otherwise, use a **Neural Network**
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    nn_model = NeuralNet(X.shape[1]).to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(100):  # 🔹 Small fixed training loop
        optimizer.zero_grad()
        predictions = nn_model(X_train)
        loss = loss_fn(predictions, y_train)
        loss.backward()
        optimizer.step()

    return nn_model
