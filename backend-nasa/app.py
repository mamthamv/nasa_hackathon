from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# Load your model (make sure to place your model file in the same directory)
model = tf.keras.models.load_model('cnn_model.h5')  # Update with your model file name

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get latitude and longitude from the request
    data = request.get_json(force=True)
    lat = data['latitude']
    lon = data['longitude']
    
    # Preprocess the input for your model
    input_data = np.array([[lat, lon]])  # Modify this according to your model's input requirements

    # Get predictions
    prediction = model.predict(input_data)  # For TensorFlow

    # Return the prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
