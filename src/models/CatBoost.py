from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import os
import joblib

class CatBoostTrainer:
    def __init__(self, iterations=100, learning_rate=0.1, depth=6):
        self.model = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=0)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("CatBoost model trained.")

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        return probabilities

    def save_model(self, directory="models"):
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        # Generate the filename using the class name
        filename = f"../{directory}/trained-models/{self.__class__.__name__}_model.joblib"
        # Save the model
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")