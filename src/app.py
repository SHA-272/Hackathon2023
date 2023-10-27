from flask import Flask, render_template, request, jsonify
import predictor
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.form

    # Создайте DataFrame из данных JSON
    input_data = pd.DataFrame(data, index=[0])
    print(input_data)
    # Вызовите функцию predict с использованием DataFrame
    prediction = predictor.predict(input_data).tostring()
    print(prediction)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
