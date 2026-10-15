TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "M&M.NS", "LT.NS"]
NAMES = {"RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services",
         "HDFCBANK.NS": "HDFC Bank", "M&M.NS": "Mahindra & Mahindra", "LT.NS": "Larsen & Toubro"}
TRAIN_END = "2024-12-31"      # model trains up to here
TEST_START = "2025-01-01"     # backtest runs on dates the model never saw
COST = 0.001                  # 0.1% per buy and per sell (brokerage + slippage estimate)
STOP = 0.08                   # early-sell level for short investments: 8% below buy price
MIN_DAYS, MAX_DAYS = 5, 60    # period range the DQN is trained on, in trading days
MODEL_PATH = "models/dqn.zip"
DB = "app.db"
MODE = "simulation"           # "simulation": trades recorded at real prices, no money moves. "live": needs broker.py connected
MAX_HOLD_DAYS = 252 * 30      # longest period a user can choose (30 years of trading days)

def stop_for(period):
    """Early-sell level. Wider for long holds so normal ups and downs don't end a long investment early."""
    return STOP if period <= MAX_DAYS else 0.15 if period <= 252 else 0.25

ABOUT = {
    "RELIANCE.NS": "One of India's biggest companies. It runs oil and gas, retail stores and the Jio mobile network.",
    "TCS.NS": "India's largest IT company. It builds and runs software for big businesses around the world.",
    "HDFCBANK.NS": "India's largest private bank. It gives loans and holds savings for millions of people.",
    "M&M.NS": "Makes tractors, SUVs like the Thar and Scorpio, and other vehicles.",
    "LT.NS": "Builds big projects such as bridges, metros, power plants and factories.",
}