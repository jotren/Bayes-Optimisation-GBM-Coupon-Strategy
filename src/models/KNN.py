from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os
import joblib

class KNNTrainer:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("KNN model trained.")

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        return probabilities

    def save_model(self, directory="models"):
        os.makedirs(directory, exist_ok=True)
        filename = f"../{directory}/trained-models/{self.__class__.__name__}_model.joblib"
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
