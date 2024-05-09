# Coupon Strategy Optimisation

## Introduction

The goal of this project was to optimise the coupon strategy for a customer base to maximise total spend. This involved three key analyses:

1. **Calculate Customer Sensitivity**: Modeled how spend changes with coupon offers using RANSAC.
2. **Redemption Probability**: Employed various machine learning models to predict the likelihood of an offer being redeemed.
3. **Optimise Strategy**: Applied Bayes-Optimisation and Linear Algebra to maximise spend based on coupon offer amounts.

## Machine Learning Models

The machine learning models explored included:

- **XGBoost**: Robust performance.
- **Neural Network**: Prone to overfitting due to small data size and limited features.
- **Gradient Boost Classifier**: Best performance, chosen for the final model.
- **Logistic Regression**: Robust but with poor precision.
- **CatBoost**: Underperformed.
- **Light Gradient Boosting (LightGBM)**: Good performance but with low precision and high recall.
- **K-Nearest Neighbors (KNN)**: Underperformed.
- **Elastic Net**: Underperformed.

The Gradient Boost Machine (GBM) was selected due to its high F1-score (max 0.75) and precision, crucial for ensuring coupons are redeemed.

## Feature Selection with Optuna

Optuna was used to fine-tune the GBM, focusing on features that significantly impacted model accuracy:

```python
features = [
    'TotalOfferAmtRedeemed',
    'TotalCouponRedeem', 
    'OfferAmt', 
    'AverageOfferAmtPerVisit',
    'AverageSpendPerVisit',
    'TotalOfferAmtReceived',
    'n-3_spend',
    'n-4_spend',
    'MinOfferAmt', 
    'Year',
    'Month',
    'weekNum'
]
```
Notably, CouponUsageRate and PatronID were found to be poor indicators of redemption probability.

## Model Performance Tracking with MLFlow

MLFlow was utilised to log and monitor various metrics effectively:

```python
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
```

## Optimised Strategy Models

The following approaches were tested for optimising the coupon strategy:

- __SKOPT__: Demonstrated the best performance; its ability to set initial points was crucial.
- __Bayes-Optimisation__: Good performance but slow in finding maxima.
- __Linprog__: Fast but limited to mapping linear relationships.

SKOPT was chosen for its efficiency in handling complex non-linear systems through an incremental strategy that proved highly effective:

- __Incremental Strategy__: Methodically increased the offer amounts while adhering to budget constraints, consistently steering towards higher spends.
- __Immediate Feedback__: Utilised immediate results to refine the model, enhancing the speed and accuracy of finding optimal solutions.

## Key Takeaways

- Iterative optimisation can significantly enhance speed and efficiency.
- Neural networks may overfit with limited data.
- GBM and LightGBM excel in handling small datasets and are resilient against overfitting.
