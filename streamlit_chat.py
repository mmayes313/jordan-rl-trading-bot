"""Mock streamlit_chat module for testing"""

class MockChat:
    @staticmethod
    def chat_input(placeholder=""):
        import streamlit as st
        return st.text_input(placeholder)

    @staticmethod
    def write_message(message, is_user=False):
        import streamlit as st
        if is_user:
            st.write(f"**You:** {message}")
        else:
            st.write(f"**Jordan:** {message}")

# Create module-level functions
def chat_input(placeholder=""):
    return MockChat.chat_input(placeholder)

def write_message(message, is_user=False):
    return MockChat.write_message(message, is_user)
