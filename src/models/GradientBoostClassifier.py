from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import os
import joblib
import pandas as pd

class GBMTrainer:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                                max_depth=max_depth, min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf)
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

    def load_model(self, directory="models"):
        # Load the model and scaler
        self.model = joblib.load( f"../{directory}/trained-models/{self.__class__.__name__}_model.joblib")
        self.scaler = joblib.load( f"../{directory}/scalers/scaler.joblib")
        print("Model and scaler loaded.")

    def predict_df(self, df, feature_names):
        """Predicts probabilities for the positive class for a DataFrame of inputs."""
        # Ensure df is a DataFrame with the right columns (features)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        df_features = df[feature_names]
        # Preprocess the DataFrame
        scaled_features = self.scaler.transform(df_features)
        # Predict and return probabilities
        probabilities = self.model.predict_proba(scaled_features)[:, 1]
        return probabilities