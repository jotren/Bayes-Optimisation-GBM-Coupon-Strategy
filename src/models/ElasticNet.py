from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import os
import joblib

class ElasticNetTrainer:
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("Elastic Net model trained.")

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        return predictions

    def save_model(self, directory="models"):
        os.makedirs(directory, exist_ok=True)
        filename = f"../{directory}/trained-models/{self.__class__.__name__}_model.joblib"
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
