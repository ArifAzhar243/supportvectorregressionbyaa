import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR

st.set_page_config(page_title="SVR Rainfall Landslide Prediction", layout="wide")

# Title and description
st.title("📈 Support Vector Regression to Predict Slope Stability Considering Rainfall and Soil Strength Effect")
st.info("This app builds and trains an SVR model to predict Factor of Safety from rainfall and soil strength parameters. Created by Arif Azhar.")

# Step 1: Load Data
with st.expander('Data'):
    st.write('*Raw Data*')
    df = pd.read_csv('https://raw.githubusercontent.com/ArifAzhar243/supportvectorregressionbyaa/refs/heads/master/aa%20Machine%20Learning.csv')
    st.dataframe(df)

# Step 2: Correlation Heatmap
st.subheader("📊 Correlation Heatmap")
correlation_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

# Step 3: Preprocess Data
st.subheader("Data Preprocessing")

required_cols = ['FOS','Friction_Angle', 'Cohesion', 'Slope_Angle', 'Rainfall_Intensity', 'Rainfall_Duration']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Dataset must contain these columns: {required_cols}. Missing: {missing_cols}")
    st.stop()

X = df[['Friction_Angle', 'Cohesion', 'Slope_Angle', 'Rainfall_Intensity', 'Rainfall_Duration']]
y = df['FOS']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

st.write(f"Training samples: {X_train.shape[0]}")
st.write(f"Validation samples: {X_val.shape[0]}")
st.write(f"Test samples: {X_test.shape[0]}")

# Step 4: Train SVR Model (fixed parameters, no float hyperparameters)
st.subheader("⚙️ SVR Model Training")
param_grid = {
    'kernel': ['rbf', 'poly'], 
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.1, 0.5, 1] 
}

with st.spinner("Training SVR model..."):
    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
    grid_search.fit(X_train, y_train)

best_svr = grid_search.best_estimator_
st.success(f"Best Parameters: {grid_search.best_params_}")

# Step 5: Model Evaluation
y_pred = best_svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**MSE:** {mse:.4f}")
st.write(f"**RMSE:** {rmse:.4f}")
st.write(f"**MAE:** {mae:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# Step 6: Actual vs Predicted Plot
st.subheader("📈 Actual vs Predicted")
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.scatter(y_test, y_pred, color='blue', alpha=0.6)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax1.set_xlabel('Actual Factor of Safety')
ax1.set_ylabel('Predicted Factor of Safety')
ax1.set_title('SVR: Actual vs Predicted Factor of Safety')
st.pyplot(fig1)

# Step 7: User Input Prediction
st.subheader("🔮 Predict Factor of Safety from New Parameters")

Friction_Angle = st.number_input("Friction Angle (°)", min_value=0, max_value=90, value=30, step=1, format="%d")
Cohesion = st.number_input("Cohesion (kPa)", min_value=0, value=20, step=1, format="%d")
Slope_Angle = st.number_input("Slope Angle (°)", min_value=0, max_value=90, value=25, step=1, format="%d")
Rainfall_Intensity = st.number_input("Rainfall Intensity (mm/hr)", min_value=0, value=50, step=1, format="%d")
Rainfall_Duration = st.number_input("Rainfall Duration (hours)", min_value=0, value=5, step=1, format="%d")

if st.button("Predict FOS"):
    try:
        input_data = pd.DataFrame([[Friction_Angle, Cohesion, Slope_Angle, Rainfall_Intensity, Rainfall_Duration]],
                                  columns=['Friction_Angle', 'Cohesion', 'Slope_Angle', 'Rainfall_Intensity', 'Rainfall_Duration'])
        input_scaled = scaler.transform(input_data)
        prediction = best_svr.predict(input_scaled)
        st.success(f"Predicted Factor of Safety: {prediction[0]:.4f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
