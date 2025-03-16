import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 🚀 Simple Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 🚀 Adaptive AI Model Selection
def train_hyperflowx(X, y, use_cuda=True):
    """Selects the best ML model dynamically based on data size."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔹 If dataset is small, use **XGBoost**
    if X.shape[1] <= 50:
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
        model.fit(X_train, y_train)
        return model

    # 🔹 Otherwise, use a **Neural Network**
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    model = NeuralNet(X.shape[1]).to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(100):  # 🔹 Small fixed training loop
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)
        loss.backward()
        optimizer.step()

    return model
