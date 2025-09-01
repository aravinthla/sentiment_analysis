from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('Models/sentiment_classifier.pkl')
vectorizer = joblib.load('Models/tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True)




