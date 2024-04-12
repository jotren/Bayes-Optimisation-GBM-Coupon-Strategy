from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score
import mlflow

def evaluate_model(y_test, probabilities, threshold=0.5):
    predicted_labels = (probabilities > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels).ravel()
    metrics = {
        'precision': precision_score(y_test, predicted_labels),
        'recall': recall_score(y_test, predicted_labels),
        'roc_auc': roc_auc_score(y_test, probabilities),
        'f1': f1_score(y_test, predicted_labels),
        'TP': tp, 
        'TN': tn, 
        'FP': fp, 
        'FN': fn
    }
    return metrics

def log_metrics(metrics):
    # Assuming mlflow is configured and imported
    mlflow.log_metrics(metrics)
