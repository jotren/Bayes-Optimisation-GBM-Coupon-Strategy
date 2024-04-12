# Optimisation 

## Introduction

Challenge was to optimise the coupon strategy for a customer base to maximse the total spend. Completed three peices of analysis on this data: 

1) __Calculate Customer Sensitivity__: Used RANSAC to model how spend changes with coupon offers
2) __Redemption Probability__: Used various ML models to predict the probability of redeeming an offer
3) __Optimise Strategy__: Used Bayes-Optimisation/Linear Algebra to optimise Spend based on coupon offer amount

### Machine Learning Models

- XGBoost: Robust performance.
- Neural Network: Prone to overfitting, data was relatively small and had few features.
- Gradient Boost Classifier: Best performance, used in final model. 
- Logistic Regression: Robust performance, poor precision
- CatBoost: Poor performance
- Light Gradient Boostr: Good performance, low precision high recall
- KNN: Poor performance
- Elastic Net: Poor performance

Used GBM as this provided the highest f1-score as well as a very high precision. Precision was important as don't want to hand out offers to people who won't redeem. Max f1-scre: 0.75

Used Optuna on my GBM to maximise the f1-score. These were the features that provided the highest level of accuracy:

```Typescript
features = [
            # 'CouponUsageRate',
            'TotalOfferAmtRedeemed',
            # 'TotalVisits', 
            # 'TotalSpendAmt',
            'TotalCouponRedeem', 
            'OfferAmt', 
            'AverageOfferAmtPerVisit',
            'AverageSpendPerVisit',
            'TotalOfferAmtReceived',
            'n-1_spend',
            # 'n-2_spend',
            'n-3_spend',
            'n-4_spend',
            # 'n-5_spend',
            # 'MaxSpend',
            # 'sensitivity_gradient', 
            'MinOfferAmt', 
            # 'MaxOfferAmt',
            # 'MedianSpendPerOfferAmt',
            'Year',
            # 'PatronID',
            'Month',
            'weekNum'
               ]
```

Interestingly, the CouponUsageRate was not a good indicator of redeemption. This could be related to the fact that new customers have a lower usage rate, but still likely to redeem. Also PatronID didn't seem to effect the outcome. Tried an encoding style model but looking at individual PatronIDs, but there was not enough data. With more data I think this could be effective.

### MlFlow

In order to track how the models are performing, we are using MlFlow. Logged varoius metrics and was able to keep track of the results through mlFlow. The metrics used where: 

```typescript
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

### Optimised Stragey Models

- SKOPT: Best performance, much faster and could set intial points which was crucial in calculating an outcome.
- Bayes-Optimisation: Good performance, just took to long to find the maxima.
- Linprog: Fast, but was only able to map linear relationships. In the end, needed a function that could model complex non-linear systems.

Eventually used SKOPT, my intial approach was to simply provide the optimiser with the bounds, and randomly intialise the data. It proved very difficult to contrust a penalty that managed to keep the offers in budget. Instead, my approach was to start with the minimum amount ever offered to a customer, and slowly increment the allocated offer amount. This convergence approach proved highly effective and allowed the optimise to find the maxima a lot faster. I think for two reasons: 

1) __Incrementing Strategy__: By consistently incrementing the offers and checking against the budget, you are continually steering the solution towards higher spends, which aligns directly with maximizing your objective.

2) __Immediate Feedback__-: Each round's results are immediately used to update the model about which regions of the parameter space are most promising. This tight feedback loop can accelerate finding an optimal solution when combined with a well-behaving objective function and model predictions that provide meaningful gradients.

### Takeaways 

1) Optimising using iterative approach can be a lot faster
2) Neural networks are very prone to overfit with limited data
3) GBM and LightGBM are excellent with small datasets and robust with respect to overfitting