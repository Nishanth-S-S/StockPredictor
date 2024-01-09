import yfinance as yf
import pandas as pd
from prophet import Prophet
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta


st.title(f"Stock Price Predictor")
st.write("""The trend simply reflects the current direction or pattern of something, and it's important to note that it describes what's happening now,
          rather than foretelling what will happen in the future.""")
def get_current_date():
    current_date = datetime.now().date()
    formatted_date = current_date.strftime("%Y-%-m-%-d")
    return formatted_date

today_date = get_current_date()

# Set the ticker symbol and the start and end dates
ticker = str(st.text_input('Stock Ticker'))
if not ticker:  # Check if ticker symbol is empty
    st.warning("Please enter a valid stock ticker symbol.")
    st.stop()
start_date = "2015-01-01"
end_date = today_date

# Fetch the data
try:
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data available for the specified ticker symbol.")
        st.stop()
except Exception as e:
    st.error("Error occurred while fetching the data: " + str(e))
    st.stop()

# Extract the closing prices
df = data[["Close"]].reset_index()
df.columns = ["ds", "y"]

# Create the Prophet model
model = Prophet()

# Fit the model to the data
model.fit(df)

# Generate future dates
future_dates = model.make_future_dataframe(periods=365)

# Make predictions
forecast = model.predict(future_dates)

# Create a Streamlit app
st.title(f"{ticker} Stock Price Forecast")

# Plot the full forecast and actual data together using Plotly
fig_full_forecast = go.Figure()

# Add the full forecasted values
fig_full_forecast.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Full Forecast',
    line=dict(color='blue')
))

# Add the actual values
fig_full_forecast.add_trace(go.Scatter(
    x=df['ds'],
    y=df['y'],
    mode='markers',
    name='Actual',
    marker=dict(color='red')
))

# Configure the layout for the full forecast
fig_full_forecast.update_layout(
    title=f"Full Stock Price Forecast",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Stock Price"),
    hovermode="x",
    showlegend=True
)

# Display the full forecast plot using Streamlit with full width
st.plotly_chart(fig_full_forecast, use_container_width=True)

# Plot the forecast for the next month using Plotly
today = pd.Timestamp(datetime.today().date())
next_month = today + pd.Timedelta(days=30)

# Filter forecast for the next month
forecast_next_month = forecast[(forecast['ds'] >= today) & (forecast['ds'] <= next_month)]

# Create a separate plot for the forecast of the next month with adjusted dimensions
fig_next_month_forecast = go.Figure()

# Add the forecasted values for the next month
fig_next_month_forecast.add_trace(go.Scatter(
    x=forecast_next_month['ds'],
    y=forecast_next_month['yhat'],
    mode='lines',
    name='Forecast Next Month',
    line=dict(color='green')
))

# Configure the layout for next month's forecast with smaller dimensions
fig_next_month_forecast.update_layout(
    title=f"Stock Price Forecast for Next Month",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Stock Price"),
    hovermode="x",
    showlegend=True,
    width=700,  # Adjust width to make it smaller
    height=300  # Adjust height to make it smaller
)

# Display the forecast for the next month plot using Streamlit
st.plotly_chart(fig_next_month_forecast)

dataframe_52_week = yf.download(ticker, period="1y", auto_adjust=True, prepost=True, threads=True)
high_52_week = dataframe_52_week['High'].max()
low_52_week = dataframe_52_week['Low'].min()

# Determine the trend direction
# Define function to get trend direction
def get_trend_direction(dataframe, forecast, period):
    latest_actual_value = dataframe.iloc[-1]['y']
    previous_actual_value = dataframe.iloc[-period]['y']
    latest_predicted_value = forecast.iloc[-1]['yhat']
    previous_predicted_value = forecast.iloc[-period]['yhat']

    # Compare the latest actual value with the latest predicted value
    if latest_actual_value > latest_predicted_value:
        trend = "↑ Upward"
    elif latest_actual_value < latest_predicted_value:
        trend = "↓ Downward"
    else:
        trend = "→ Neutral"

    # If the predicted trend is different from the actual trend, update the trend direction
    if latest_predicted_value != previous_predicted_value:
        trend = f"{trend}"
    
    return trend

# Calculate trend directions for different periods using actual and predicted values
long_term_direction = get_trend_direction(df, forecast, 300)  # Considering 250 trading days for a year
mid_term_direction = get_trend_direction(df, forecast, 80)   # Considering 90 trading days for approximately 3 months
short_term_direction = get_trend_direction(df, forecast, 30) # Considering 30 trading days for approximately 1 month



# Organize information in a Streamlit table
stock_info = {
    '52-Week High': [high_52_week],
    '52-Week Low': [low_52_week],
    'Long Term Trend': [long_term_direction],
    'Mid Term Trend': [mid_term_direction],
    'Short Term Trend': [short_term_direction],
}
stock_info_df = pd.DataFrame(stock_info)

# Display the table using Streamlit
st.write(stock_info_df)


