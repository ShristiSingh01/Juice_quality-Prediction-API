from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('trained_classifier.pkl', 'rb') as file:
    loaded_classifier = pickle.load(file)

# Load the column structure used during training
with open('model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()
    # print("Form data received:", data)
    
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([data])
    # print("Input data DataFrame:\n", input_data)

    # Convert relevant numeric columns to float
    numeric_columns = [
        'pH', 'pH deviation', 'Sugar(g)', 'Sugar deviation', 'Antioxidants(%)', 
        'Antioxidant deviation(%)', 'Carbohydrates(g)', 'Carbohydrate deviation', 
        'Potassium(mg)', 'Potassium deviation', 'Vitamin C(mg)', 'Vitamin C deviation', 
        'Protein(g)', 'protein deviation', 'TVC: 10^2(CFU/ml)', 'TVC deviation'
    ]
    
    for col in numeric_columns:
        if col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
    
    # One-hot encode categorical columns (Company, Flavor, Label, and Storage condition)
    categorical_columns = ['Company', 'Flavor', 'Label', 'Storage condition']
    input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns)
    # print("Encoded input data:\n", input_data_encoded)

    # Align the encoded input data with the columns used during training
    input_data_aligned = input_data_encoded.reindex(columns=model_columns, fill_value=0)
    # print("Aligned input data:\n", input_data_aligned)

    # Predict using the loaded classifier
    try:
        prediction = loaded_classifier.predict(input_data_aligned)
        # print("Prediction result:", prediction)
        return jsonify({'prediction': prediction[0]})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
