# pytorch_mlp_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from src.common.logger import logger
from src.model_training.models.base_model import BaseModel

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class PyTorchMLPModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, model_name='pytorch_mlp')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.epochs = self.config.get('models', {}).get('pytorch_mlp', {}).get('epochs', 10)
        self.batch_size = self.config.get('models', {}).get('pytorch_mlp', {}).get('batch_size', 32)
        self.lr = self.config.get('models', {}).get('pytorch_mlp', {}).get('learning_rate', 0.001)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        input_dim = X_train.shape[1]
        self.model = MLP(input_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values.reshape(-1,1), dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(dataloader.dataset)
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")

        self.is_trained = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError(f"Model {self.name} is not trained yet")
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy()
        return outputs.ravel()

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        from sklearn.metrics import accuracy_score
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        return accuracy_score(y, y_pred)