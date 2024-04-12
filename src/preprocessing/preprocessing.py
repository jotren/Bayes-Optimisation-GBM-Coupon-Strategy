import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import joblib

def count_redeems(series):
    return (series == 1).sum()

def create_lagged_features(df, date_col='UseStartDate', id_col='PatronID', spend_col='Spend'):
    # Sort values by PatronID and the specified date column to ensure correct chronological order
    df_sorted = df.sort_values(by=[id_col, date_col])
    
    # Group by PatronID and shift the Spend column to get n-1, n-2, and n-3 spends
    df_sorted['n-1_spend'] = df_sorted.groupby(id_col)[spend_col].shift(1)
    df_sorted['n-2_spend'] = df_sorted.groupby(id_col)[spend_col].shift(2)
    df_sorted['n-3_spend'] = df_sorted.groupby(id_col)[spend_col].shift(3)
    df_sorted['n-4_spend'] = df_sorted.groupby(id_col)[spend_col].shift(4)
    df_sorted['n-5_spend'] = df_sorted.groupby(id_col)[spend_col].shift(5)
    
    # Replace NaN spends with 0
    df_sorted[['n-1_spend', 'n-2_spend', 'n-3_spend', 'n-4_spend', 'n-5_spend']] = df_sorted[['n-1_spend', 'n-2_spend', 'n-3_spend', 'n-4_spend', 'n-5_spend']].fillna(0)
    
    # For each PatronID, keep the last row only (this row has the latest n-1, n-2, n-3 spends)
    df_result = df_sorted.groupby(id_col).last().reset_index()
    
    # Select only relevant columns to return
    df_result = df_result[[id_col, 'n-1_spend', 'n-2_spend', 'n-3_spend', 'n-4_spend', 'n-5_spend']]
    
    return df_result

def preprocess_and_split(df_train, df_test, df_gradient):
    # Merge the sensitivity gradient data with both training and testing data based on PatronID
    df_train = df_train.merge(df_gradient, on='PatronID', how='left')
    df_test = df_test.merge(df_gradient, on='PatronID', how='left')
    df_train.to_csv('../data/processed/df_train.csv')
    df_train_n_spend = create_lagged_features(df_train)

    # Convert 'RedeemedYN' to binary (assuming 'Y' and 'N' values)
    df_train['RedeemedYN'] = df_train['RedeemedYN'].apply(lambda x: 1 if x == 'Y' else 0)
    df_train['SpendPerOfferRatio'] = df_train['Spend'] / df_train['OfferAmt']
    df_test['RedeemedYN'] = df_test['RedeemedYN'].apply(lambda x: 1 if x == 'Y' else 0)
    
    # Calculate aggregated features on training data
    agg_features = df_train.groupby('PatronID').agg(
        CouponUsageRate=('RedeemedYN', 'mean'),
        TotalOfferAmtReceived=('OfferAmt', 'sum'),
        AverageSpendPerVisit=('Spend', 'mean'),
        AverageOfferAmtPerVisit=('OfferAmt', 'mean'),
        TotalSpendAmt=('Spend', 'sum'),
        TotalOfferAmtRedeemed=('OfferAmtRedeemed', 'sum'),
        TotalVisits=('PatronID', 'count'),
        TotalCouponRedeem=('RedeemedYN', count_redeems),
        MaxSpend=('Spend', 'max'),
        MaxOfferAmt=('OfferAmt', 'max'),
        MinOfferAmt=('OfferAmt', 'min'),
        MedianSpendPerOfferAmt=('SpendPerOfferRatio', 'median')  # Calculate Median Spend Per Offer Ratio for each PatronID
    ).reset_index()

    # Filter out customers with less than 3 visits
    agg_features['MedianSpendPerOfferAmt'].replace([np.inf, -np.inf], 1, inplace=True)


    # Merge filtered aggregated features and median spend per coupon back to training and test sets based on PatronID
    df_train = df_train.merge(agg_features, on='PatronID', how='inner')
    df_test = df_test.merge(agg_features, on='PatronID', how='inner')
    df_train = df_train.merge(df_train_n_spend, on='PatronID', how='inner')
    df_test = df_test.merge(df_train_n_spend, on='PatronID', how='inner')
    # df_test.to_csv("../data/processed/processed-test-data.csv")
    
    # Combine both to ensure uniform preprocessing, after merging additional features
    combined_df = pd.concat([df_train, df_test], keys=['train', 'test'])

    # Additional feature engineering applied uniformly
    combined_df['weekNum'] = pd.to_datetime(combined_df['UseStartDate']).dt.isocalendar().week
    combined_df['Month'] = pd.to_datetime(combined_df['UseStartDate']).dt.month  # Add Month
    combined_df['Year'] = pd.to_datetime(combined_df['UseStartDate']).dt.year  # Add Month
    combined_df['sensitivity_gradient'] = combined_df['sensitivity_gradient'].fillna(combined_df['sensitivity_gradient'].median())
    combined_df.fillna(0, inplace=True)
    combined_df.to_csv("../data/processed/processed-test-data.csv", index=False)

    
    # Updating features list to include 'Month' and 'Quarter'
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

    
    X = combined_df[features]
    y = combined_df['RedeemedYN']
    
    # Splitting back into train and test datasets
    X_train = X.xs('train')
    X_test = X.xs('test')
    y_train = y.xs('train')
    y_test = y.xs('test')

    # Scaling features (fit on training data, then transform both training and test data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 
    scaler_path = '../models/scalers/scaler.joblib'
    joblib.dump(scaler, scaler_path)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, features