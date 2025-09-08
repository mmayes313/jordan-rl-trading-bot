# Training Guide

## Local Training
- CPU: 100 days (~hours)
- Command: `python scripts/simple_interface.py train_local --days=100`
- Saves to data/models/best_model

## Colab Training
- Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
- Clone repo or upload files
- Run notebook: training_colab.ipynb
- GPU: 1 year simulation (days, free tier limits – upgrade for speed)
- Always resume from best P&L checkpoint
- Save to Drive: model.save('/content/drive/MyDrive/data/models/colab_best.zip')

## Tips
- Monitor TD error and rewards in logs
- Use meta-learning for cross-asset adaptation
- Validate with backtesting on historical data
