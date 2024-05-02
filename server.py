from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app,)


# Load the trained model
rf_model = joblib.load('random_forest_model.pkl')
ann_model = joblib.load('ann_model.pkl')

label_mapping_df = pd.read_csv('transaction.csv', index_col=0)


def rf_clean_data(df) -> pd.DataFrame:
    # Drop unnecessary columns
    df.drop(['Transaction ID', 'Customer ID', 'IP Address'], axis=1, inplace=True)
    
    # Convert 'Payment Method', 'Product Category', 'Device Used', 'Customer Location' to categorical type
    categorical_columns = ['Payment Method', 'Product Category', 'Device Used', 'Customer Location']
    for column in categorical_columns:
        df[column] = df[column].astype(str)
    
    # Encoding categorical variables using label encoder
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    
    # Add new features
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    df['Account Age Days'] = df['Account Age Days']
    df['Transaction Hour'] = df['Transaction Date'].dt.hour
    df['Address_Mismatch'] = (df['Shipping Address'] != df['Billing Address'])
    df['Transaction_Hour'] = df['Transaction Date'].dt.hour
    

    # Drop redundant columns after encoding
    df.drop(['Shipping Address', 'Billing Address', 'Transaction Date'], axis=1, inplace=True)
    
    return df


@app.route('/predict/randomforest', methods=['POST'])
def rfpredict():
    # Receive the request data
    data = request.get_json()
    
    # Convert data to DataFrame
    df = pd.DataFrame([data])
    
    # Perform data preprocessing
    df = rf_clean_data(df)
    
    
    # Check if shipping address matches billing address
    if df['Address_Mismatch'].iloc[0]:
        # If shipping address does not match billing address, return fraudulent response
        return jsonify({'prediction': 'Fraudulent', 'reason': 'Shipping address does not match billing address'})
    
    transaction_amount = float(data['Transaction Amount'])
    customer_age = int(data['Customer Age'])

    if (transaction_amount > 1000) & (customer_age < 25):
        # If transaction amount is suspicious based on age, return fraudulent response
        return jsonify({'prediction': 'Fraudulent', 'reason': 'Suspicious transaction amount based on age'})


    

    # Manual label encoding for categorical features
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        if column in label_mapping_df:
            label_mapping = label_mapping_df[column].to_dict()
            df[column] = df[column].map(label_mapping)
    
    # Make prediction
    prediction = rf_model.predict(df)
    
    # Convert prediction to human-readable format
    fraud_prediction = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"
    
    return jsonify({'prediction': fraud_prediction})










def ann_clean_data(df) -> pd.DataFrame:
    # Drop unnecessary columns
    df.drop(['Transaction ID', 'Customer ID', 'IP Address'], axis=1, inplace=True)
    
    # Convert 'Payment Method', 'Product Category', 'Device Used', 'Customer Location' to categorical type
    categorical_columns = ['Payment Method', 'Product Category', 'Device Used', 'Customer Location']
    for column in categorical_columns:
        df[column] = df[column].astype(str)
    
    # Encoding categorical variables using label encoder
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    
    # Add new features
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    df['Account Age Days'] = df['Account Age Days']
    df['Transaction Hour'] = df['Transaction Date'].dt.hour
    df['Address_Mismatch'] = (df['Shipping Address'] != df['Billing Address'])
    df['Transaction_Hour'] = df['Transaction Date'].dt.hour
    
    
    # Drop redundant columns after encoding
    df.drop(['Shipping Address', 'Billing Address', 'Transaction Date'], axis=1, inplace=True)
    
    return df


@app.route('/predict/ann', methods=['POST'])
def annpredict():
    # Receive the request data
    data = request.get_json()
    
    # Convert data to DataFrame
    df = pd.DataFrame([data])
    
    # Preprocess the data
    data = ann_clean_data(df)
    
    # Manual label encoding for categorical features
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        if column in label_mapping_df:
            label_mapping = label_mapping_df[column].to_dict()
            data[column] = data[column].map(label_mapping)
    
    # Make prediction
    prediction = ann_model.predict(data)
    
    # Convert prediction to human-readable format
    fraud_prediction = ["Not Fraudulent" if pred == 0 else "Fraudulent" for pred in prediction]
    
    return jsonify({'prediction': fraud_prediction})




if __name__ == '__main__':
    # Allow all IPs to access the Flask app
    app.run(debug=True, host='0.0.0.0')
