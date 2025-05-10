import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Data Loading
data = pd.read_csv('heart.csv')  # Replace with your actual dataset path
print("Data Loaded Successfully")

# Step 2: Data Inspection
print("\nFirst few rows of the dataset:")
print(data.head())

# Step 3: Data Preprocessing
# Encode categorical features using LabelEncoder
label_encoder = LabelEncoder()

# Encode 'Sex' column
data['Sex'] = label_encoder.fit_transform(data['Sex'])  # 0: female, 1: male

# Encode 'ChestPainType' column
data['ChestPainType'] = label_encoder.fit_transform(data['ChestPainType'])

# Encode 'RestingECG' column
data['RestingECG'] = label_encoder.fit_transform(data['RestingECG'])

# Encode 'ExerciseAngina' column
data['ExerciseAngina'] = label_encoder.fit_transform(data['ExerciseAngina'])

# Encode 'ST_Slope' column
data['ST_Slope'] = label_encoder.fit_transform(data['ST_Slope'])

# Step 4: Define Features and Target
X = data.drop('HeartDisease', axis=1)  # Features
y = data['HeartDisease']  # Target

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split into training and testing sets. Training data size:", X_train.shape[0])

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("\nModel trained successfully.")

# Step 8: Model Prediction
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Step 9: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the model: {accuracy * 100:.2f}%")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Heart Disease'], yticklabels=['No Disease', 'Heart Disease'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 10: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Step 11: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Step 12: Feature Importance
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Predicting Heart Disease')
plt.show()

# Step 13: Save the Trained Model
joblib.dump(model, 'heart_disease_predictor_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save scaler too
print("\nModel and Scaler saved successfully.")

# Step 14: Load the Model and Scaler
loaded_model = joblib.load('heart_disease_predictor_model.pkl')
scaler = joblib.load('scaler.pkl')
print("\nModel and Scaler loaded successfully.")

# Step 15: Option to Take Patient Details
choice = input("\nDo you want to enter patient details for prediction? (y/n): ").lower()

if choice == 'y':
    # Mappings for user input encoding
    sex_map = {'Male': 1, 'Female': 0}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    restecg_map = {'Normal': 1, 'ST-T abnormality': 2, 'Left ventricular hypertrophy': 0}
    exang_map = {'Yes': 1, 'No': 0}
    slope_map = {'Upsloping': 2, 'Flat': 1, 'Downsloping': 0}

    print("\n--- Enter Patient Details ---")
    age = int(input("Age: "))
    if age < 30:
        print("⚠️ Warning: Prediction might be unreliable for patients below 30 years old.")
    sex = input("Sex (Male/Female): ")
    cp = input("Chest Pain Type (Typical Angina/Atypical Angina/Non-Anginal Pain/Asymptomatic): ")
    trestbps = int(input("Resting Blood Pressure (mm Hg): "))
    chol = int(input("Serum Cholesterol (mg/dl): "))
    fbs = int(input("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No): "))
    restecg = input("Resting ECG (Normal/ST-T abnormality/Left ventricular hypertrophy): ")
    thalach = int(input("Maximum Heart Rate Achieved: "))
    exang = input("Exercise Induced Angina (Yes/No): ")
    oldpeak = float(input("ST Depression Induced by Exercise: "))
    slope = input("Slope of the Peak Exercise ST Segment (Upsloping/Flat/Downsloping): ")

    # Encoding user input
    sample = [
        age,
        sex_map.get(sex, 0),
        cp_map.get(cp, 0),
        trestbps,
        chol,
        fbs,
        restecg_map.get(restecg, 1),
        thalach,
        exang_map.get(exang, 0),
        oldpeak,
        slope_map.get(slope, 1)
    ]

    # Scaling
    sample_scaled = scaler.transform([sample])

    # Predict
    prediction = loaded_model.predict(sample_scaled)
    prediction_prob = loaded_model.predict_proba(sample_scaled)[:, 1]

    # Output
    print("\nSample input prediction:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")
    print(f"Sample input probability: {prediction_prob[0]*100:.2f}%")
    if prediction_prob > 0.8:
        print("Risk Level: High Risk ⚠️")
        print("Suggestions:")
        print("- Immediate consultation with a cardiologist is strongly recommended.")
        print("- Avoid strenuous activities; complete medical evaluation needed.")
        print("- Start a heart-healthy diet (low salt, low fat, rich in fruits and vegetables).")
        print("- Monitor blood pressure, cholesterol, and blood sugar levels regularly.")
        print("- Strictly follow medication and treatment plans if already prescribed.")

    elif prediction_prob > 0.5:
        print("Risk Level: Medium Risk ⚠️")
        print("Suggestions:")
        print("- Schedule a detailed heart health check-up soon.")
        print("- Begin moderate physical activity (after doctor approval).")
        print("- Focus on weight control, healthy eating, and stress reduction.")
        print("- Avoid smoking and limit alcohol consumption.")
        print("- Regularly monitor blood pressure and cholesterol.")

    else:
        print("Risk Level: Low Risk ✅")
        print("Suggestions:")
        print("- Maintain a healthy lifestyle: balanced diet, regular exercise, no smoking.")
        print("- Annual heart check-ups for preventive care.")
        print("- Manage stress through mindfulness, yoga, or hobbies.")
        print("- Continue monitoring health parameters (BP, sugar, cholesterol) yearly.")

else:
    print("\nNo patient data entered. Skipping prediction.")

# Step 16: Conclusion
print("\nEnd of Script. You have successfully built, evaluated, and used your Heart Disease Prediction model.")
