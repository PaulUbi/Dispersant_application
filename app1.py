import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from playsound import playsound
import os

# Load the saved XGBoost model for initial efficiency prediction
model_file_path = "xgboost_model.pkl"

try:
    with open(model_file_path, 'rb') as f:
        XGBoost_final = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load XGBoost model: {e}")
    st.stop()


# Define custom CSS for Google Font
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

body {
    font-family: 'Inter', sans-serif;
}
h1 {
    font-family: 'Inter', sans-serif;
}
</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Define the app interface with styled title
st.markdown("<h1 style='text-align: center; font-family: Inter, sans-serif;'>DRS-Система Рекомендации</h1>", unsafe_allow_html=True)

# Create two columns for the images
col1, col2 = st.columns(2)
col1.image('Dipersant_image.png', caption='Диспергент', width=300)
col2.image('DRS.png', width=300)

st.markdown('''
Это приложение прогнозирует эффективность диспергента.
''')

# Sidebar
st.sidebar.image('KNITU.png', width=500)

# Input fields for user input
st.sidebar.header('Входные параметры')
st.sidebar.markdown('Отрегулируйте свои параметры ниже:')
temperature = st.sidebar.number_input('Температура (C)', min_value=0.0, max_value=30.0, value=25.0)
salinity = st.sidebar.number_input('Соленость', min_value=0.0, max_value=35.0, value=15.0)
viscosity = st.sidebar.number_input('Кинематическая вязкость (мм^2/с)', min_value=0.0, max_value=80.0, value=50.0)
density = st.sidebar.number_input('Плотность', min_value=0.5, max_value=0.9, value=0.7, format="%.4f")
oil_field = st.sidebar.selectbox('Нефтяное месторождение', ["Хохряковское", "Усинское", "Правдинское", "Нагорн.(Турней)", "Нагорн.(Башкир)"])
dispersant_ratio = st.sidebar.selectbox('Соотношение диспергента к нефти', ["1:10", "1:20"])

# Horizontal bar for weather conditions above the graph
st.header('Погодные условия')
col_temp, col_wind, col_wave, col_days = st.columns(4)
initial_temperature = col_temp.number_input('Температура (C)', min_value=0.0, max_value=50.0, value=25.0, key='initial_temperature')
wind_speed = col_wind.number_input('Скорость ветра (м/с)', min_value=0.0, max_value=50.0, value=5.0, key='wind_speed')
wave_length = col_wave.number_input('Высота волны (м)', min_value=0.0, max_value=20.0, value=1.0, key='wave_length')
days = col_days.number_input('Количество дней', min_value=1, max_value=30, value=30, key='days')


# Map the selected oil field to its corresponding encoded value
oil_field_encoded = {"Хохряковское": 4, "Усинское": 3, "Правдинское": 2, "Нагорн.(Турней)": 1, "Нагорн.(Башкир)": 0}
encoded_oil_field = oil_field_encoded[oil_field]

# Map the selected dispersant ratio to its corresponding encoded value
dispersant_ratio_encoded = {"1:10": 1, "1:20": 0}
encoded_dispersant_ratio = dispersant_ratio_encoded[dispersant_ratio]

# Function to simulate efficiency over time
def simulate_efficiency(days, initial_temperature, wind_speed, wave_length, base_efficiency):
    efficiencies = []
    for day in range(days):
        # Adjust factors for each day
        temp_factor = initial_temperature * (1.01 ** day)  # Efficiency increases with temperature
        wind_factor = wind_speed * (0.98 ** day)  # Efficiency decreases with wind speed
        wave_factor = wave_length * (1.01 ** day)  # Efficiency decreases with wave length
        
        # Calculate efficiency
        efficiency = base_efficiency * (1.01 ** temp_factor) * (0.95 ** wind_factor) * (0.95 ** wave_factor) * (0.95 ** day)
        efficiencies.append(max(efficiency, 0))  # Ensure efficiency is non-negative
    return efficiencies

# Button to trigger predictions
if st.sidebar.button('Прогноз'):
    # Prepare input data for initial efficiency prediction
    input_data = np.array([[temperature, salinity, viscosity, density, encoded_oil_field, encoded_dispersant_ratio]])
    
    # Make initial prediction
    prediction = XGBoost_final.predict(xgb.DMatrix(input_data))
    base_efficiency = prediction[0]
    
    # Display initial prediction
    st.subheader('Результат прогноза')
    if base_efficiency < 50:
        st.markdown('<style>@keyframes blink { 50% { opacity: 0; } } .blinking { animation: blink 1s infinite; color: red; }</style>', unsafe_allow_html=True)
        st.markdown(f'<p class="blinking">Прогнозируемая эффективность диспергента: {base_efficiency:.2f}% - Предупреждение: Использование диспергента может оказаться нецелесообразным</p>', unsafe_allow_html=True)
        playsound("beep.mp3")
    else:
        st.success(f'Прогнозируемая эффективность диспергента: {base_efficiency:.2f}% - Использование диспергента возможно')
        
        # Simulate efficiency over time
        efficiencies = simulate_efficiency(days, initial_temperature, wind_speed, wave_length, base_efficiency)
        
        # Find the optimum number of reserve days
        reserve_days = next((i + 1 for i, efficiency in enumerate(efficiencies) if efficiency < 50), days + 1)
        
        st.info(f'У вас есть в запасе не более {reserve_days - 1} дней, прежде чем диспергент будет не эффективен.')
        
        # Plot the efficiency over time
        st.subheader('График эффективности диспергента с течением времени')
        days_range = np.arange(1, days + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.set(style='darkgrid')
        sns.lineplot(x=days_range, y=efficiencies, marker='o', ax=ax, label='Efficiency')
        ax.axhline(y=50, color='gray', linestyle='--', label='50% Efficiency Threshold')
        ax.axvline(x=reserve_days - 1, color='red', linestyle='--', label='Optimum Reserve Days')
        ax.text(reserve_days - 1, 50, 'Резервные дни', color='red', ha='center', va='bottom')
        ax.set_title('Эффективность диспергента с течением времени', fontsize=16, fontweight='bold')
        ax.set_xlabel('Количество дней', fontsize=14)
        ax.set_ylabel('Эффективности (%)', fontsize=14)
        ax.set_ylim(0, base_efficiency + 10)  # Set y-axis limit to a bit higher than base efficiency for better visualization
        ax.legend(loc='upper right')
        st.pyplot(fig)
