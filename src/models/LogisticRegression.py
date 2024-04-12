import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import joblib

class LogisticRegressionTrainer:
    def __init__(self):
        self.model = LogisticRegression(solver='liblinear')
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("Model trained.")

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]  # Get probabilities for the positive class
        return probabilities

    def save_model(self, directory="models"):
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        # Generate the filename using the class name
        filename = f"../{directory}/trained-models/{self.__class__.__name__}_model.joblib"
        # Save the model
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")