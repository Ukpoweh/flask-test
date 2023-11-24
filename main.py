import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Define the API endpoint for model predictions
@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions based on input data.

    Parameters:
    - features (dict): Dictionary containing 'categorical_features' and 'numeric_features'.

        categorical_features:
        season (planting season)
        label (crop)
        location (country)


        numeric_features:
        temperature
        humidity
        ph
        water availability

    Returns:
    - predictions (dict): Dictionary containing prediction details.
    """
    try:
        # Get input data from the request
        input_data = request.get_json()

        
        # Validate input format
        if 'features' not in input_data or not isinstance(input_data["features"], dict):
            raise ValueError("Invalid input format. 'features' should be a dictionary.")
        

        # Extract features from input data
        features = input_data["features"]

        
        # Validate presence of required keys
        required_keys = ["categorical_features", "numeric_features"]
        for key in required_keys:
            if key not in features:
                raise ValueError(f"Missing required key: {key}")
            

        # Label encode categorical features
        labeled_categories = encoder.fit_transform(features["categorical_features"])
        labeled_categories = np.array([labeled_categories]).reshape(1, -1)
        
        #scaling numerical features
        features_scaled = scaler.transform(np.array(features["numeric_features"]).reshape(1, -1))

        # Concatenate encoded and scaled features
        processed_features = np.concatenate([labeled_categories, features_scaled], axis=1)


        # Make predictions using the trained model
        predictions = model.predict(processed_features)

        predicted_class = int(predictions[0])

        predicted_season = ''
        if predicted_class == 0:
            predicted_season = "rainy"
        elif predicted_class == 1:
            predicted_season ="spring"
        elif predicted_class == 2:
            predicted_season = "summer"
        elif predicted_class == 3:
            predicted_season = "winter"

        
        # Format the predictions along with additional details
        result = {"Predicted harvest season": predicted_season}

        return jsonify(result)

    except ValueError as ve:
        # Handle validation errors
        return jsonify({'error': str(ve)})

    except Exception as e:
        # Handle other errors gracefully
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app on localhost:5000
    app.run(debug=True)

