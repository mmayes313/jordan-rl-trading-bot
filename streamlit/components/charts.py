import plotly.graph_objects as go

def create_pnl_chart(data):
    fig = go.Figure(data=go.Scatter(y=data['pnl']))
    return fig
