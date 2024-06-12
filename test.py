import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocess the dataset
def preprocess_data(df):
    df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    label_encoder = LabelEncoder()
    df['Churn'] = label_encoder.fit_transform(df['Churn'])
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'Churn':
            df[col] = label_encoder.fit_transform(df[col])
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=[np.number]).columns.drop('Churn')
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

data = preprocess_data(data)

# Split dataset into features and target variable
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Reshape input data for LSTM
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build and compile the LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

# Define learning rate schedule
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True)

# Build and compile the LSTM model with the learning rate schedule
input_shape = (X_train.shape[1], X_train.shape[2])
lstm_model = build_lstm_model(input_shape)
lstm_model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

# Train the LSTM model
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Generate predictions from LSTM and XGBoost models
y_pred_prob_lstm = lstm_model.predict(X_test)
xgb_model = XGBClassifier()
xgb_model.fit(X_train.reshape(X_train.shape[0], X_train.shape[2]), y_train)
y_pred_prob_xgb = xgb_model.predict_proba(X_test.reshape(X_test.shape[0], X_test.shape[2]))[:, 1]  # Probability of class 1

# Concatenate predictions as new features
X_train_stacked = np.column_stack((y_pred_prob_lstm.flatten(), y_pred_prob_xgb))
X_test_stacked = np.column_stack((y_pred_prob_lstm.flatten(), y_pred_prob_xgb))

# Train a meta-learner (Random Forest in this case) on the stacked features
meta_learner = RandomForestClassifier(n_estimators=100, random_state=101)
meta_learner.fit(X_train_stacked, y_test)

# Predict with the meta-learner
y_pred_meta = meta_learner.predict(X_test_stacked)

# Evaluate the meta-learner
print("Meta-Learner Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_meta))
print("Meta-Learner Classification Report:")
print(classification_report(y_test, y_pred_meta))
accuracy = accuracy_score(y_test, y_pred_meta)
print("Meta-Learner Accuracy:", accuracy)

# Manual sample input
manual_sample = pd.DataFrame({
    'gender': ['Female'],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['No'],
    'tenure': [28],
    'PhoneService': ['Yes'],
    'MultipleLines': ['Yes'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['Yes'],
    'TechSupport': ['Yes'],
    'StreamingTV': ['Yes'],
    'StreamingMovies': ['Yes'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [104.8],
    'TotalCharges': [3046.05]
})

# Preprocess the manual sample
def preprocess_manual_sample(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

manual_sample_processed = preprocess_manual_sample(manual_sample)

# Reshape manual input data for LSTM
X_manual_lstm = manual_sample_processed.values.reshape((manual_sample_processed.shape[0], 1, manual_sample_processed.shape[1]))

# Generate predictions from LSTM and XGBoost models for the manual input
y_pred_prob_lstm_manual = lstm_model.predict(X_manual_lstm)
y_pred_prob_xgb_manual = xgb_model.predict_proba(manual_sample_processed)[:, 1]  # Probability of class 1

# Stack the predictions as new features for the meta-learner
X_manual_stacked = np.column_stack((y_pred_prob_lstm_manual.flatten(), y_pred_prob_xgb_manual))

# Predict with the meta-learner for the manual input
y_pred_meta_manual = meta_learner.predict(X_manual_stacked)

# Output the prediction for the manual input
predicted_churn_manual = 'Yes' if y_pred_meta_manual[0] == 1 else 'No'
print("Predicted Churn for the manual input:", predicted_churn_manual)

