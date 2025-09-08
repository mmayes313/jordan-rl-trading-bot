import plotly.graph_objects as go

def create_pnl_chart(pnl_series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=pnl_series, mode='lines', name='P&L'))
    fig.update_layout(title='P&L Over Time', xaxis_title='Time', yaxis_title='P&L')
    return fig

def create_candlestick_chart(data):
    fig = go.Figure(data=[go.Candlestick(x=data['time'], open=data['open'], high=data['high'], low=data['low'], close=data['close'])])
    fig.update_layout(title='Candlestick Chart', xaxis_title='Time', yaxis_title='Price')
    return fig

# Add more chart functions as needed
