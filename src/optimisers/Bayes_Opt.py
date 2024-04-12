import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction

global corrected_offers
corrected_offers = {}

def adjust_offer_amount(value, min_val, max_val):
    """Round the value within specified min and max bounds."""
    return max(min(value, max_val), min_val)

def runMachineLearningPrediction(row_for_prediction, model, feature_names):
    """Preprocess and predict the probability for a given input row using the trainer instance."""
    row_df = row_for_prediction.to_frame().T if isinstance(row_for_prediction, pd.Series) else row_for_prediction
    predicted_probabilities = model.predict_df(row_df, feature_names)
    return predicted_probabilities[0] if len(predicted_probabilities) == 1 else predicted_probabilities

def calculate_expected_spend(prob, median_spend_per_offer_amt, offer_amt):
    """Calculate expected spend based on probability, median spend per offer, and offer amount."""
    return median_spend_per_offer_amt * prob * offer_amt

def optimize_offers(data, model, feature_names, budget, rounds, init_points, budget_penalty_weight, penalty_exponent):
    global corrected_offers

    def objective(**params):
        global corrected_offers
        total_spend = 0
        corrected_params = {}
                
        for i, (key, offer_amt_param) in enumerate(params.items()):
            row = data.iloc[i]
            offer_amt_corrected = offer_amt_param
            corrected_params[key] = offer_amt_corrected
            
            row_for_prediction = row.copy()
            row_for_prediction['OfferAmt'] = offer_amt_corrected
            prob = runMachineLearningPrediction(row_for_prediction, model, feature_names)
            
            if prob >= 0.5:
                expected_spend = calculate_expected_spend(prob, row['MedianSpendPerOfferAmt'], offer_amt_corrected)
                total_spend += expected_spend

        corrected_offers = corrected_params.copy()

        budget_penalty = (budget - total_spend)**penalty_exponent if total_spend < budget else 0
        return -(total_spend - budget_penalty_weight * budget_penalty)

    # Adjust initial pbounds if needed based on historical data or other heuristics
    pbounds = {f'offer_amt_{i}': (row['MinOfferAmt'], row['MaxOfferAmt']) 
               for i, row in data.iterrows()}

    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42, verbose=0)
    optimizer.maximize(init_points=init_points, n_iter=rounds)

    # Apply the optimized offers
    for i, (key, offer_amt) in enumerate(corrected_offers.items()):
        data.at[int(key.split('_')[-1]), 'BayesOptOfferAmt'] = offer_amt
        row_for_prediction = data.iloc[int(key.split('_')[-1])].copy()
        row_for_prediction['OfferAmt'] = offer_amt
        prob = runMachineLearningPrediction(row_for_prediction, model, feature_names)
        data.at[int(key.split('_')[-1]), 'NNRedeemPrediction'] = prob
        data.at[int(key.split('_')[-1]), 'BayesOptExpSpend'] = calculate_expected_spend(prob, data.iloc[int(key.split('_')[-1])]['MedianSpendPerOfferAmt'], offer_amt)

    return data
