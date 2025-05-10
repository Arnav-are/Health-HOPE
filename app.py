from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])

        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)[0][1]

        if probability > 0.8:
            risk = "High Risk ⚠️"
            suggestions = [
                "Immediate consultation with a cardiologist is strongly recommended.",
                "Avoid strenuous activities; complete medical evaluation needed.",
                "Start a heart-healthy diet (low salt, low fat, rich in fruits and vegetables).",
                "Monitor blood pressure, cholesterol, and blood sugar levels regularly.",
                "Strictly follow medication and treatment plans if already prescribed."
            ]
        elif probability > 0.5:
            risk = "Medium Risk ⚠️"
            suggestions = [
                "Schedule a detailed heart health check-up soon.",
                "Begin moderate physical activity (after doctor approval).",
                "Focus on weight control, healthy eating, and stress reduction.",
                "Avoid smoking and limit alcohol consumption.",
                "Regularly monitor blood pressure and cholesterol."
            ]
        else:
            risk = "Low Risk ✅"
            suggestions = [
                "Maintain a healthy lifestyle.",
                "Exercise regularly (with proper guidance).",
                "Continue a balanced diet rich in fruits, vegetables, and low in processed foods.",
                "Have routine health checkups.",
                "Stay away from smoking and manage stress properly."
            ]

        return render_template('index.html', 
                               prediction_text=f'Prediction: {"Heart Disease" if prediction[0] else "No Heart Disease"}',
                               probability_text=f'Probability: {probability*100:.2f}%',
                               risk_text=f'Risk Level: {risk}',
                               suggestions=suggestions)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
