from dash import Dash, dcc, html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Sample data
df = pd.DataFrame({
    'Date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'Quantity': range(100)
})
df.set_index('Date', inplace=True)

# Ensure frequency is set
df.index.freq = 'D'

# Feature Engineering: Add lag features (optional)
df['lag_1'] = df['Quantity'].shift(1)
df.dropna(inplace=True)

# Split the data
test_size = int(len(df) * 0.2)
train = df[:-test_size]
test = df[-test_size:]

# Train the ARIMA model
model = ARIMA(train['Quantity'], order=(5, 1, 0))
model_fit = model.fit()

# Forecast the test set
predictions = model_fit.forecast(steps=test_size)

# Create a DataFrame for comparison
test['predictions'] = predictions

# Combine train and test data for plotting
comparison_df = pd.concat([train[['Quantity']], test[['Quantity', 'predictions']]], axis=0)

# Create traces for the dashboard
trace_actual = go.Scatter(x=comparison_df.index, y=comparison_df['Quantity'], mode='lines', name='Actual Sales')
trace_predicted = go.Scatter(x=test.index, y=test['predictions'], mode='lines', name='Predicted Sales', line=dict(color='red'))

# Create figure
fig = go.Figure(data=[trace_actual, trace_predicted])

# Define the layout of the app
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Sales Forecast Dashboard"),
    dcc.Graph(figure=fig)
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
