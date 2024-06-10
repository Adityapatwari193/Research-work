import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
import xgboost as xgb

# Load dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocess the dataset
def preprocess_data(df):
    # Drop the 'customerID' column as it is not a feature
    df = df.drop(columns=['customerID'])

    # Fill missing TotalCharges with 0 and convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Label encode the target variable
    label_encoder = LabelEncoder()
    df['Churn'] = label_encoder.fit_transform(df['Churn'])

    # Handle categorical features
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'Churn':
            df[col] = label_encoder.fit_transform(df[col])

    # Normalize numerical features
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

# Build the MLP model with L2 regularization and dropout layers
def build_mlp_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
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

# Build and compile the MLP model with the learning rate schedule
input_shape = (X_train.shape[1],)
mlp_model = build_mlp_model(input_shape)
mlp_model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

# Train the MLP model
mlp_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Train the XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=101)
xgb_model.fit(X_train, y_train)

# Predict with MLP model
y_pred_prob_mlp = mlp_model.predict(X_test)
y_pred_mlp = (y_pred_prob_mlp > 0.5).astype(int)

# Predict with XGBoost model
y_pred_xgb = xgb_model.predict(X_test)

# Combine predictions (simple average of predictions)
y_pred_combined = ((y_pred_mlp + y_pred_xgb.reshape(-1, 1)) > 1).astype(int)

# Evaluate the combined model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_combined))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_combined))
print(f'Accuracy: {accuracy_score(y_test, y_pred_combined):.4f}')
