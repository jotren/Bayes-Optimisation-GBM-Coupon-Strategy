import pandas as pd
import numpy as np
from skopt import Optimizer
from skopt.space import Real
from skopt.utils import use_named_args

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

def optimize_offers_skopt(data, model, feature_names, budget, rounds, increment_amount):
    global corrected_offers
    corrected_offers = {}
    
    # Define the space for optimization
    space = [Real(0, row['MaxOfferAmt'], name=f'offer_amt_{i}') for i, row in data.iterrows()]

    def objective(params_list):
        total_expected_spend = 0
        for i, offer_amt_param in enumerate(params_list):
            row = data.iloc[i]
            prob = runMachineLearningPrediction(row, model, feature_names)
            
            offer_amt_corrected = adjust_offer_amount(offer_amt_param, 0, row['MaxOfferAmt']) if prob >= 0.5 else 0
            expected_spend = calculate_expected_spend(prob, row['MedianSpendPerOfferAmt'], offer_amt_corrected)
            total_expected_spend += expected_spend
            corrected_offers[i] = offer_amt_corrected

        
        total_offer_amount = sum(corrected_offers.values())
        return - total_expected_spend

    optimizer = Optimizer(dimensions=space, random_state=42, acq_func='LCB')
    initial_points_for_y0 = [row['MinOfferAmt'] for _, row in data.iterrows()]
    y0 = objective(initial_points_for_y0)
    optimizer.tell(initial_points_for_y0, y0)  # Initialize the optimizer with the initial points

    current_offer_amt = initial_points_for_y0

    for round_index in range(rounds):
        next_offer_amt = [adjust_offer_amount(val + increment_amount, 0, space[i].high) for i, val in enumerate(current_offer_amt)]
        
        # Check total offers against the budget
        if sum(corrected_offers.values()) >= budget:
            # Scale down uniformly to fit the budget if exceeded
            scale_factor = budget / sum(next_offer_amt)
            next_offer_amt = [adjust_offer_amount(val * scale_factor, 0, space[i].high) for i, val in enumerate(next_offer_amt)]
        
        # Perform optimization with adjusted offers
        y = objective(next_offer_amt)
        optimizer.tell(next_offer_amt, y)
        
        # Update current offers for the next iteration
        current_offer_amt = next_offer_amt

        # Check if we are close to the budget to make final adjustments
        print(f"Offer: {sum(corrected_offers.values())}")
        print(f"Budget: {budget}")
        if sum(corrected_offers.values()) >= budget * 0.99:
            break

    optimized_data = apply_optimized_offers(data, corrected_offers, model, feature_names)
    return optimized_data

def apply_optimized_offers(data, corrected_offers, model, feature_names):
    for i, offer_amt in corrected_offers.items():
        data.at[i, 'BayesOptOfferAmt'] = offer_amt
        row_for_prediction = data.iloc[i].copy()
        row_for_prediction['OfferAmt'] = offer_amt
        prob = runMachineLearningPrediction(row_for_prediction, model, feature_names)
        data.at[i, 'MLRedeemPrediction'] = prob
        data.at[i, 'BayesOptExpSpend'] = calculate_expected_spend(prob, data.iloc[i]['MedianSpendPerOfferAmt'], offer_amt)

    return data
