from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# Load your model (make sure to place your model file in the same directory)
model = tf.keras.models.load_model('cnn_model.h5')  # Update with your model file name

app = Flask(__name__)

def predict_conditions(input_lat, input_lon):
    relevant_columns= [
    		'EVLAND','PRECTOTLAND','GWETPROF', 'GWETROOT', 
    'GWETTOP', 'LAI','TSOIL1','TSURF'
        ]
    # Get the nearest data point based on lat and lon
    nearest_data = filter_data_by_lat_lon(data, input_lat, input_lon)
    
    # Extract features for prediction
    X_input = nearest_data[relevant_columns].values.reshape(1, -1)
    
    # Normalize or standardize the input as done during training
    X_input_scaled = scaler.transform(X_input)
    
    # Reshape the input for the CNN (or any model you use)
    X_input_reshaped = X_input_scaled.reshape(1, 1, X_input_scaled.shape[1])  # For CNN

    # Make predictions using the model
    predictions = model.predict(X_input_reshaped)
    
    # Interpret the predictions (convert numeric outputs into verbal or percentage)
    water_availability = predictions[0][0]
    soil_condition = predictions[0][1]
    crop_health = predictions[0][2]
    frost_risk = predictions[0][3]
    yield_quality = predictions[0][4]
    
    # Convert numeric predictions into simple conditions
        # 1. Water Status based on Water Availability Index (WAI)
    water_status = "Good" if water_availability > 0 else "Poor"

    # 2. Soil Status based on Soil Moisture Index (SMI)
    soil_status = "Optimal" if soil_condition > 0.5 else "Dry"

    # 3. Crop Health Status based on Crop Health Index (CHI)
    crop_health_status = "Healthy" if crop_health == 1 else ("At Risk" if crop_health == -1 else "Average")

    # 4. Frost Risk Status based on Frost Risk Indicator (FRI)
    frost_risk_status = f"Frost risk: {'High' if frost_risk == 1 else 'Low'}"

    # 5. Yield Quality Status based on Yield Quality Index (YQI)
    yield_quality_status = f"Yield quality: {yield_quality * 100:.2f}%"
    
    
    
    # Return the predictions (can also plot a graph if required)
    return {
        "Water Availability": water_status,
        "Soil Condition": soil_status,
        "Crop Health": crop_health_status,
        "Frost Risk": frost_risk_status,
        "Yield Quality": yield_quality_status
    }

@app.route('/predict', methods=['POST'])
def predict():
    # Get latitude and longitude from the request
    data = request.get_json(force=True)
    lat = data['latitude']
    lon = data['longitude']
    
    prediction=predict_conditions(lat, lon)



    # Return the prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
