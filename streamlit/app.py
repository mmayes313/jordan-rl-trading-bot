import streamlit as st
import time
from config.dashboard_config import DASHBOARD_THEME, UPDATE_INTERVAL, EMOJIS
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Jordan RL Bot", layout="wide", initial_sidebar_state="expanded")
st.markdown('<style>' + open('assets/styles.css').read() + '</style>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🤖 Jordan RL Trading Bot")
page = st.sidebar.selectbox("Select Tab", ["📊 Live Dashboard", "🚀 Top Signals", "🧠 Model Insights", "📈 Performance", "💬 Jordan Chat"])

# Auto-refresh
st_autorefresh(interval=UPDATE_INTERVAL)

if page == "📊 Live Dashboard":
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{EMOJIS['pnl']} P&L", "1.2%", delta="0.5%")
        st.progress(0.75)  # Daily target progress
    with col2:
        st.metric(f"{EMOJIS['drawdown']} Drawdown", "2.1%", delta="-0.3%")
    st.metric("Balance", "$10,500", delta="$500")
    st.metric("Equity", "$10,400")
    st.metric("Free Margin", "$9,800")
    st.metric("Open Trades", 3)
    st.metric("Win Rate", "65%")
    st.metric("Consec Wins", 2)
    # Success calc
    target_pct = st.number_input("Daily Target %", value=1.0)
    dd_pct = st.number_input("Max Drawdown %", value=5.0)
    prob = 1 / (target_pct + dd_pct) * 100  # Mock
    st.metric("Success Probability", f"{prob:.1f}%")

    # Current trades
    st.subheader("Current Trades")
    st.write("Trade 1: EURUSD Buy 0.5 lots - Entry: CCI < -60")

elif page == "🚀 Top Signals":
    st.subheader("Top 10 Opportunities (5m + 30m Momentum)")
    for i in range(10):
        st.write(f"{i+1}. EURUSD - Confidence: 85% - Profit/Risk: 2:1 {EMOJIS['signals']}")
    # Interactive chart
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Candlestick(x=['1m','2m'], open=[1.1,1.105], high=[1.11,1.115], low=[1.09,1.095], close=[1.105,1.11]))
    st.plotly_chart(fig, use_container_width=True)

elif page == "🧠 Model Insights":
    st.subheader("Model Reasoning")
    st.write("Current: No trade - CCI masks blocking sell.")
    st.metric("TD Error", 0.05)
    st.metric("Daily Reward", 150)
    st.metric("KL Divergence", 0.01)
    # Decision viz
    import plotly.express as px
    fig = px.bar(x=['CCI', 'SMA', 'ATR'], y=[0.3, 0.2, 0.1], title="Indicator Weights")
    st.plotly_chart(fig)

elif page == "📈 Performance":
    st.subheader("P&L Charts")
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=[10000, 10200, 10500], mode='lines', name='Daily P&L'))
    st.plotly_chart(fig)
    st.write("Drawdown Peaks: 3.2%")
    st.write("Trade Freq: 600/day")
    st.write("Asset Breakdown: EURUSD 40%, GBPUSD 30%")

elif page == "💬 Jordan Chat":
    st.subheader("Chat with Jordan Belfort 💬")
    import sys
    sys.path.append('..')
    from streamlit_chat import st_chat
    user_input = st_chat.chat_input("Ask me about markets, code, or trades...")
    if user_input:
        st_chat.write_message("Jordan: " + f"Yo, {user_input}? Let's crush it – buy EURUSD, you fucker! Market's pumping. 💰", is_user=False)
    # Daily commentary (mock)
    st.write("Daily News: ForexFactory alert – Fed rate hike, dump USD pairs! ⚠️")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🚀 Ready to trade? Run --train in scripts!")
