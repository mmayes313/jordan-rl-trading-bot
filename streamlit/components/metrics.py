import streamlit as st

def display_pnl(pnl):
    st.metric("P&L 💰", pnl)
