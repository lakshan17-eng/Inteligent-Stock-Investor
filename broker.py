# EXECUTION LAYER: the only place that would touch real money.
# MODE "simulation": trades are recorded at real market prices, no money moves.
# MODE "live": connect a licensed broker API here (e.g. Zerodha Kite Connect, Upstox, Angel One).
from config import MODE, COST

def buy(ticker, amount, price):
    """Place a buy order for `amount` rupees. Returns quantity bought."""
    if MODE == "live":
        raise NotImplementedError("Connect a broker API in broker.buy().")
    return amount * (1 - COST) / price

def sell(ticker, qty, price):
    """Sell `qty` shares. Returns rupees received."""
    if MODE == "live":
        raise NotImplementedError("Connect a broker API in broker.sell().")
    return qty * price * (1 - COST)