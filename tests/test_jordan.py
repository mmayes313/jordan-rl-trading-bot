import pytest
from unittest.mock import patch, MagicMock
from streamlit.components.jordan_personality import get_jordan_response

def test_jordan_personality():
    with patch('streamlit.components.jordan_personality.client') as mock_client:
        # Create a mock response object
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Fuck yeah!'
        
        mock_client.chat.completions.create.return_value = mock_response
        response = get_jordan_response("Test", "context", "news")
        assert "Fuck yeah!" in response
