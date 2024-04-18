from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Get input values from the form
    val1 = float(request.form['bedrooms'])
    val2 = float(request.form['bathrooms'])
    val3 = float(request.form['floors'])
    val4 = float(request.form['yr_built'])

    # Create a numpy array with the input values and reshape it
    arr = np.array([val1, val2, val3, val4]).reshape(1, -1)

    # Make prediction using the model
    pred = model.predict(arr)

    # Render the index.html template with the prediction result
    return render_template('index.html', data=int(pred[0]))

if __name__ == '__main__':
    app.run(debug=True)