import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page config
st.set_page_config(page_title="SVR Slope Stability Prediction", layout="wide")

# Title and description
st.title("📈 Support Vector Regression for Slope Stability Considering Rainfall and Soil Strength")
st.info("This app builds and trains an SVR model to predict Factor of Safety (FOS) from soil strength and rainfall parameters. Created by Arif Azhar.")

# Step 1: Load Data
with st.expander("📂 View Raw Data"):
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/ArifAzhar243/dp-aamachinelearning/refs/heads/master/aa%20Machine%20Learning.csv')
        st.write(df.head())
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Ensure correct columns exist
required_columns = ['friction_angle', 'cohesion', 'slope_angle', 'rainfall_intensity', 'rainfall_duration', 'FOS']
if not all(col in df.columns for col in required_columns):
    st.error(f"Dataset must contain columns: {required_columns}")
    st.stop()

# Step 2: Define features and target
X = df[['friction_angle', 'cohesion', 'slope_angle', 'rainfall_intensity', 'rainfall_duration']]
y = df['FOS']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train SVR with GridSearch
st.subheader("🔧 Model Training")
if st.button("Train SVR Model"):
    with st.spinner("Training SVR model... This may take a while."):
        svr = SVR()
        param_grid = {
            'kernel': ['rbf', 'poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'epsilon': [0.01, 0.1, 0.5]
        }
        grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
        grid_search.fit(X_train_scaled, y_train)
        best_svr = grid_search.best_estimator_
        
        st.success("Model training complete.")
        st.write("**Best Parameters:**", grid_search.best_params_)

        # Step 6: Evaluation
        y_pred = best_svr.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 Model Performance")
        st.write(f"**MSE:** {mse:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**MAE:** {mae:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")

        # Step 7: Actual vs Predicted Plot
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.scatter(y_test, y_pred, color='blue', alpha=0.6)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax1.set_xlabel('Actual Factor of Safety')
        ax1.set_ylabel('Predicted Factor of Safety')
        ax1.set_title('SVR Model: Actual vs Predicted FOS')
        ax1.grid(True)
        st.pyplot(fig1)

        # Step 8: Feature Importance (only for linear kernel)
        if best_svr.kernel == 'linear':
            importance = best_svr.coef_[0]
            feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
            feature_importance = feature_importance.sort_values('Importance', ascending=False)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.barh(feature_importance['Feature'], feature_importance['Importance'])
            ax2.set_xlabel('Importance')
            ax2.set_title('Feature Importance for FOS Prediction')
            st.pyplot(fig2)
        else:
            st.info("Feature importance is only available for linear kernel models.")
