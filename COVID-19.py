# Streamlit --> It helps to create interactive websites
import streamlit as st

# Pandas --> It helps manipulate data
import pandas as pd

# Matplotlib --> It helps create visualizations
import matplotlib.pyplot as plt

# Numpy --> It helps with statistical data analysis
import numpy as np

# Warnings --> It helps to display a warning message
import warnings

# Datetime --> It helps to process dates and times
from datetime import datetime

# Curve fit --> It helps to fit a curve to a set of data points
from scipy.optimize import curve_fit

# Mean Squared Error -->  It helps to calculate the Mean Squared Error (MSE) 
# between two sets of data: the true values and the predicted values
from sklearn.metrics import mean_squared_error

# Min Max Scaler --> It helps to normalize or standardize data by transforming features within a specific range, 
# typically between 0 and 1.
from sklearn.preprocessing import MinMaxScaler

########################################## TITLE #############################################
##############################################################################################

st.set_page_config(layout="wide")
st.header("Transmission Rate Estimation of COVID-19 in Bandung Using SIR Model and RK4")

######################################## LOAD DATA ###########################################
##############################################################################################

# Change the Dropbox link to download directly
data = "https://raw.githubusercontent.com/Dafebecca/STREAMLIT---COVID-19-BANDUNG/main/DATASET%20COVID-19%20BANDUNG.csv"

# Read the dataset into a Pandas DataFrame
df = pd.read_csv(data, delimiter=";")

######################################## FORMAT DATA #########################################
##############################################################################################

# Change the data type for the susceptible, infected, and recovered columns to numeric
df["Susceptible"] = pd.to_numeric(df["Susceptible"], errors="coerce")
df["Infected"] = pd.to_numeric(df["Terinfeksi"], errors="coerce")
df["Recovered"] = pd.to_numeric(df["Removed"], errors="coerce")

# Convert the "Date" column to datetime
df["Tanggal"] = pd.to_datetime(df["Tanggal"])

# Set the "Date" column as DateTimeIndex
df = df.set_index("Tanggal")
df.index = pd.to_datetime(df.index)

########################################## DATA RANGE #########################################
###############################################################################################

def sample_data_by_date_range(df, start_date, end_date):
    """
    Samples data from a DataFrame based on a specific date range

    Parameters:
    df (pd.DataFrame): DataFrame containing data.
    start_date (str): Start date (inclusive), in format "YYYY-MM-DD HH:MM:SS"
    end_date (str): End date (inclusive), in format "YYYY-MM-DD HH:MM:SS"

    Returns:
    pd.DataFrame: A DataFrame containing data from a specified date range
    """
    # Make sure the date column is of type datetime
    df.index = pd.to_datetime(df.index)
    
    # Time-based indexing to retrieve data within a date range
    return df[start_date:end_date]

##################################### SIR MODEL - RK4 #########################################
###############################################################################################

# Function SIR Model - Runge Kutta 4th Order
def S_RK4(t, beta, gamma, population):
    xplot = [0]
    N = population
    h = 1
    I = [1]
    S = [N-3]
    R = [0]

    for i in range(len(t)):
        Sn = S[i]
        In = I[i]
        Rn = R[i]

        # RK calculations
        k_1S = -beta * Sn * In / N
        k_1I = beta * Sn * In / N - gamma * In
        k_1R = gamma * In

        k_2S = -beta * (Sn + k_1S * h / 2) * (In + k_1I * h / 2) / N
        k_2I = beta * (Sn + k_1S * h / 2) * (In + k_1I * h / 2) / N - gamma * (In + k_1I * h / 2)
        k_2R = gamma * (In + k_1I * h / 2)

        k_3S = -beta * (Sn + k_2S * h / 2) * (In + k_2I * h / 2) / N
        k_3I = beta * (Sn + k_2S * h / 2) * (In + k_2I * h / 2) / N - gamma * (In + k_2I * h / 2)
        k_3R = gamma * (In + k_2I * h / 2)

        k_4S = -beta * (Sn + k_3S * h) * (In + k_3I * h) / N
        k_4I = beta * (Sn + k_3S * h) * (In + k_3I * h) / N - gamma * (In + k_3I * h)
        k_4R = gamma * (In + k_3I * h)

        yS = Sn + (h / 6) * (k_1S + 2 * k_2S + 2 * k_3S + k_4S)
        yI = In + (h / 6) * (k_1I + 2 * k_2I + 2 * k_3I + k_4I)
        yR = Rn + (h / 6) * (k_1R + 2 * k_2R + 2 * k_3R + k_4R)

        xplot.append(xplot[-1] + h)
        S.append(yS)
        I.append(yI)
        R.append(yR)

    return S[:-1]

def I_RK4(t, beta, gamma, population):
    xplot = [0]
    N = population
    h = 1
    I = [1]
    S = [N-3]
    R = [0]

    for i in range(len(t)):
        Sn = S[i]
        In = I[i]
        Rn = R[i]

        k_1S = -beta * Sn * In / N
        k_1I = beta * Sn * In / N - gamma * In
        k_1R = gamma * In

        k_2S = -beta * (Sn + k_1S * h / 2) * (In + k_1I * h / 2) / N
        k_2I = beta * (Sn + k_1S * h / 2) * (In + k_1I * h / 2) / N - gamma * (In + k_1I * h / 2)
        k_2R = gamma * (In + k_1I * h / 2)

        k_3S = -beta * (Sn + k_2S * h / 2) * (In + k_2I * h / 2) / N
        k_3I = beta * (Sn + k_2S * h / 2) * (In + k_2I * h / 2) / N - gamma * (In + k_2I * h / 2)
        k_3R = gamma * (In + k_2I * h / 2)

        k_4S = -beta * (Sn + k_3S * h) * (In + k_3I * h) / N
        k_4I = beta * (Sn + k_3S * h) * (In + k_3I * h) / N - gamma * (In + k_3I * h)
        k_4R = gamma * (In + k_3I * h)

        yS = Sn + (h / 6) * (k_1S + 2 * k_2S + 2 * k_3S + k_4S)
        yI = In + (h / 6) * (k_1I + 2 * k_2I + 2 * k_3I + k_4I)
        yR = Rn + (h / 6) * (k_1R + 2 * k_2R + 2 * k_3R + k_4R)

        xplot.append(xplot[-1] + h)
        S.append(yS)
        I.append(yI)
        R.append(yR)

    return I[:-1]

def R_RK4(t, beta, gamma, population):
    xplot = [0]
    N = population
    h = 1
    I = [1]
    S = [N-3]
    R = [0]

    for i in range(len(t)):
        Sn = S[i]
        In = I[i]
        Rn = R[i]

        k_1S = -beta * Sn * In / N
        k_1I = beta * Sn * In / N - gamma * In
        k_1R = gamma * In

        k_2S = -beta * (Sn + k_1S * h / 2) * (In + k_1I * h / 2) / N
        k_2I = beta * (Sn + k_1S * h / 2) * (In + k_1I * h / 2) / N - gamma * (In + k_1I * h / 2)
        k_2R = gamma * (In + k_1I * h / 2)

        k_3S = -beta * (Sn + k_2S * h / 2) * (In + k_2I * h / 2) / N
        k_3I = beta * (Sn + k_2S * h / 2) * (In + k_2I * h / 2) / N - gamma * (In + k_2I * h / 2)
        k_3R = gamma * (In + k_2I * h / 2)

        k_4S = -beta * (Sn + k_3S * h) * (In + k_3I * h) / N
        k_4I = beta * (Sn + k_3S * h) * (In + k_3I * h) / N - gamma * (In + k_3I * h)
        k_4R = gamma * (In + k_3I * h)

        yS = Sn + (h / 6) * (k_1S + 2 * k_2S + 2 * k_3S + k_4S)
        yI = In + (h / 6) * (k_1I + 2 * k_2I + 2 * k_3I + k_4I)
        yR = Rn + (h / 6) * (k_1R + 2 * k_2R + 2 * k_3R + k_4R)

        xplot.append(xplot[-1] + h)
        S.append(yS)
        I.append(yI)
        R.append(yR)

    return R[:-1]

##################################### EVALUATION (RMSE) #######################################
###############################################################################################

# Initialize MinMaxScaler
scaler = MinMaxScaler()

def normalize_data(data):
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

def calculate_rmse(actual, predicted):
    # Normalize data
    actual_normalized = normalize_data(actual)
    predicted_normalized = normalize_data(np.array(predicted))
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_normalized, predicted_normalized))
    return rmse

################################## VISUALIZATION SELECTED DATA ################################
###############################################################################################

def visualize_covid(df_selected, selected_category, start_date, end_date):
    fig, ax2 = plt.subplots(figsize=(20, 8))
    ax2.plot(df_selected.index, df_selected[selected_category], color="blue")
    
    # Set title and labels
    ax2.set_title(f"Selected COVID-19 Data Range {start_date} - {end_date}", fontsize="15")
    ax2.set_xlabel("Date",fontsize="15")
    ax2.set_ylabel("Total Cases", fontsize="15")
    
    # Calculate tick positions (every 10 days)
    tick_positions = df_selected.index[::10]
    tick_labels = [date.strftime('%Y-%m-%d') for date in tick_positions]
    
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=90)
    
    # Add grid for better readability
    ax2.grid(True)
    
    # Display the plot
    st.write(fig)

##################################### CONFIGURASI MODEL #######################################
###############################################################################################

# Create a section for selecting data categories
st.sidebar.subheader("Categories")
select_categories = st.sidebar.selectbox("Select Categories", options=["Infected", "Recovered", "Susceptible"])

fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(df.index, df[select_categories], color="darkblue")
ax.set_title("Dataset - COVID-19 Bandung")
st.write(fig)

# Create a time frame section
st.sidebar.divider()
st.sidebar.subheader("Data Range")

options_date = df.index
counter = len(options_date) - 1 

start_date = st.sidebar.date_input("Start Period", value=datetime.strptime("2021-05-01", "%Y-%m-%d").date())
end_date = st.sidebar.date_input("End Period", value=datetime.strptime("2021-07-28", "%Y-%m-%d").date())

if st.sidebar.button("Select Data", key="select_data"):
    ne = sample_data_by_date_range(df, start_date=start_date, end_date=end_date)
    visualize_covid(ne, select_categories, start_date, end_date)

st.sidebar.divider()
st.sidebar.subheader("SIR - RK4 Model")

_initialP = st.sidebar.number_input("Initial Population", value=180000)
_beta = st.sidebar.text_input("Parameter Beta (Transmission Rate)", value=0.145)
_beta = float(_beta)

# Fixed value of gamma
gamma = 0.07142857

# Create a button for running the SIR Model
if st.sidebar.button("Run", key="run_sir"):
    # Select the appropriate actual data
    if select_categories == "Infected":
        actual_data = df.loc[start_date:end_date, "Infected"].to_numpy()
        predicted_data = I_RK4(np.arange(len(actual_data)), _beta, gamma, _initialP)
    elif select_categories == "Recovered":
        actual_data = df.loc[start_date:end_date, "Recovered"].to_numpy()
        predicted_data = R_RK4(np.arange(len(actual_data)), _beta, gamma, _initialP)
    elif select_categories == "Susceptible":
        actual_data = df.loc[start_date:end_date, "Susceptible"].to_numpy()
        predicted_data = S_RK4(np.arange(len(actual_data)), _beta, gamma, _initialP)

    # Ensure x and y data are the same size
    if len(actual_data) == len(predicted_data):
        try:
            plt.figure(figsize=(14, 7))
            plt.plot(df.loc[start_date:end_date].index, actual_data, label="Actual Data")
            plt.scatter(df.loc[start_date:end_date].index, predicted_data, color="orange", label=f"SIR Model Prediction (RK4) - {select_categories}")
            plt.plot(df.loc[start_date:end_date].index, predicted_data, color="orange", label=f"SIR Model Prediction (RK4) - {select_categories}")
            plt.xlabel("Date", fontsize="15")

            date_range = df.loc[start_date:end_date].index
            plt.xticks(date_range[::10], rotation=90)
            plt.ylabel("Total Cases", fontsize="15")
            plt.title(f"Comparison of Actual COVID-19 Cases with SIR Model Predictions - {select_categories}", fontsize="15")
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

            # Normalize data for RMSE calculation
            rmse = calculate_rmse(actual_data, predicted_data)
            st.write(f"RMSE for {select_categories} Data After Normalization: {rmse:.4f}")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("The lengths of actual and predicted data do not match. Please check the data ranges.")
