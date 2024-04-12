from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import os
import joblib

class BayesianModelTrainer:
    def __init__(self):
        self.model = GaussianNB()
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("Bayesian model trained.")

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        return probabilities

    def save_model(self, directory="models"):
        os.makedirs(directory, exist_ok=True)
        filename = f"../{directory}/trained-models/{self.__class__.__name__}_model.joblib"
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
