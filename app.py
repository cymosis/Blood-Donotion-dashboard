import joblib
from flask import Flask, request, jsonify
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

# Load trained model, scaler, label encoders, and used features
model = joblib.load('blood_donation_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
used_features = joblib.load('used_features.pkl')  # Load the saved list of used features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request
        data = request.get_json()

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])
        numerical_columns = ['Height', 'Weight', 'Age', 'Days Since Last Period', 'Days Since Last Donation', 'Hemoglobin']

        # Initialize label encoders and imputer
        label_encoders = {}

        categorical_cols=df.select_dtypes(include=['object']).columns
        # Encoding categorical columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        # Scaling numerical columns
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        df = df.reindex(columns=used_features, fill_value=None)


        # Make prediction
        prediction = model.predict(df)

        # Return result
        if prediction[0] == 1:
            eligibility = 'Eligible'
        elif prediction[0] == 2:
            eligibility = 'Temporairement Non-eligible'
        elif prediction[0] == 0:
            eligibility = 'Définitivement non-eligible'
        else:
            eligibility = 'To be determined'

        return jsonify({'eligibility': eligibility})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
