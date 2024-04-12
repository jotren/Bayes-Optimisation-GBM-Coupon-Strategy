import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import joblib

class NeuralNetwork(nn.Module):
    def __init__(self, num_features):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class NeuralNetworkTrainer:
    def __init__(self, num_features):
        self.model = NeuralNetwork(num_features=num_features)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, epochs=25, batch_size=32):
        X_train_scaled = torch.tensor(self.scaler.fit_transform(X_train), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        for epoch in range(epochs):
            self.model.train()
            for i in range(0, len(X_train_scaled), batch_size):
                indices = torch.randperm(X_train_scaled.size(0))[i:i+batch_size]
                batch_x, batch_y = X_train_scaled[indices], y_train_tensor[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def predict(self, X_test):
        X_test_scaled = torch.tensor(self.scaler.transform(X_test), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            probabilities = self.model(X_test_scaled).squeeze().numpy()
        probabilities = np.nan_to_num(probabilities, nan=0.0)
        return probabilities

    def save_model(self, directory="models", scaler_path="models/scalers/scaler.joblib"):
        filename = f"../{directory}/trained-models/{self.__class__.__name__}_model"

        os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), filename)
        joblib.dump(self.scaler, scaler_path)
        print("Model and scaler saved.")    
    
    def load_model(self, directory="models",scaler_path="models/scalers/scaler.joblib"):
        model_path = f"../{directory}/trained-models/{self.__class__.__name__}_model"
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode
        self.scaler = joblib.load(scaler_path)
        print("Model and scaler loaded.")
    
    def predict_df(self, df, feature_names):
        """Takes a DataFrame and returns the prediction for each row."""
        # Ensure df is a DataFrame with the right columns (features)
        df = df[feature_names]
        # Preprocess the DataFrame
        scaled_features = self.scaler.transform(df)
        # Convert to tensor
        inputs = torch.tensor(scaled_features, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(inputs).numpy().flatten()  # Predict and convert to numpy array
        return predictions