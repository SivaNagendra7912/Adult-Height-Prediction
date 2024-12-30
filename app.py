from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from flask import Blueprint, render_template

app = Flask(__name__)

predictions = Blueprint('predictions', __name__)

# Define the index route
@predictions.route('/')
def index():
    return render_template('index.html')


# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about/about.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Extract and prepare input features
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        predicted_height = float(prediction[0])  # Convert prediction to float
        prediction_result = f"{predicted_height:.2f} inches"  # Format prediction for readability
        
        # Redirect to the result page
        return render_template('predictions/result.html', prediction_text=prediction_result)
    return render_template('predictions/index.html')

@app.route("/metrics")
def metrics():
    return render_template("metrics/metrics.html")

@app.route("/flowchart")
def flowchart():
    return render_template("flowchart/flowchart.html")

if __name__ == "__main__":
    app.run(debug=True)
