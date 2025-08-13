import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

st.set_page_config(page_title="ANN Rainfall Landslide Prediction", layout="wide")

# Title and description
st.title("🤖 Support Vector Regression to Predict Slope Stability Considering Rainfall and Soil Strength Effect")
st.info("This app builds and trains an SVR model to predict Factor of Safety from rainfall and soil strength parameters. Created by Arif Azhar.")

# Step 1: Load Data
with st.expander('Data'):
  st.write('*Raw Data*')
  df = pd.read_csv('https://raw.githubusercontent.com/ArifAzhar243/dp-aamachinelearning/refs/heads/master/aa%20Machine%20Learning.csv')
  df

