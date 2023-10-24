from flask import Flask, render_template, request, jsonify
import predictor

app = Flask(__name__)

# Пример использования вашей библиотеки для предсказания данных
def predict_data(data):
    result = predictor.predict(data)

    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.form.get('data')  # Предполагаем, что у вас есть форма с полем "data" в вашем HTML
    prediction = predict_data(data)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')