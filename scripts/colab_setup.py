# Run in Colab cell
!pip install -r requirements.txt
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/mmayes313/jordan-rl-trading-bot.git /content/bot
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
