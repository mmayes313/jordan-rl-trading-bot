from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Clone or copy the repo
os.system('git clone https://github.com/your-username/jordan_rl_trading_bot /content/jordan_rl_trading_bot')

# Change directory
os.chdir('/content/jordan_rl_trading_bot')

# Install requirements
os.system('pip install -r requirements.txt')

# Add to path
import sys
sys.path.append('/content/jordan_rl_trading_bot/src')

print("Colab setup complete!")
