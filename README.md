# Paper Investor (DQN, paper money, real prices)

Setup (Python 3.10+):
    python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
    pip install -r requirements.txt

Run in this order:
    python data.py        # 1. download prices (needs internet)
    python agent.py       # 2. train the DQN (about 10-30 min on CPU)
    python backtest.py    # 3. test on unseen 2025+ data vs equal-weight basket
    uvicorn api:app --reload   # 4. open http://127.0.0.1:8000

Notes: yfinance data is free and delayed, so "real time" means latest daily close. The period is in trading days
(weekends only are skipped; exchange holidays are not). Sells happen at the end date or if price falls 8% below buy.
Change stocks, costs, stop-loss and dates in config.py, then delete models/ and retrain.
