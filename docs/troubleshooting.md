# Troubleshooting Guide

## Common Issues
- **MT5 not connecting?** Restart MT5 terminal and ensure it's logged in.
- **Low success probability?** Retrain with tighter DD limits or more data.
- **Torch GPU issues?** Ensure CUDA is installed for Colab.
- **API errors?** Check API keys in config/dashboard_config.py.
- **Memory errors?** Reduce batch size or use CPU for local training.

## Debug Steps
1. Run tests: `python tests/run_all_tests.py`
2. Check logs in data/logs/
3. Simulate a day: Use env.reset() and step() manually.
4. Verify data: Ensure CSVs are exported correctly.
