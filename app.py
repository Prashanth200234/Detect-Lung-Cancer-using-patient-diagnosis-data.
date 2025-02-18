from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model_data = joblib.load('lung_cancer_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['encoders']

# Initialize Flask app
app = Flask(__name__)

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        data = request.form
        new_patient = pd.DataFrame({
            'age': [float(data['age'])],
            'gender': [data['gender']],
            'country': [data['country']],
            'cancer_stage': [data['cancer_stage']],
            'family_history': [data['family_history']],
            'smoking_status': [data['smoking_status']],
            'bmi': [float(data['bmi'])],
            'cholesterol_level': [float(data['cholesterol_level'])],
            'hypertension': [data['hypertension']],
            'asthma': [data['asthma']],
            'cirrhosis': [data['cirrhosis']],
            'other_cancer': [data['other_cancer']],
            'treatment_type': [data['treatment_type']],
            'treatment_duration': [float(data['treatment_duration'])]
        })

        # Encode categorical features and handle unseen labels
        for col in label_encoders.keys():
            if new_patient[col][0] not in label_encoders[col].classes_:
                label_encoders[col].classes_ = np.append(label_encoders[col].classes_, new_patient[col][0])  # Add new label
            new_patient[col] = label_encoders[col].transform(new_patient[col])

        # Scale numerical features
        numerical_cols = ['age', 'bmi', 'cholesterol_level', 'treatment_duration']
        new_patient[numerical_cols] = scaler.transform(new_patient[numerical_cols])

        # Convert all columns to float
        new_patient = new_patient.astype(float)

        # Make Prediction
        prediction = model.predict(new_patient)
        prediction_prob = model.predict_proba(new_patient)[:, 1]

        # Output prediction
        result = "Yes" if prediction[0] == 1 else "No"
        confidence = round(prediction_prob[0] * 100, 2)

        return render_template('index.html', prediction_text=f"Survival Prediction: {result} ({confidence}% confidence)")

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
