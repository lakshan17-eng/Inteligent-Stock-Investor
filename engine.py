# MODULE 6a - ACCOUNTS + INVESTING ENGINE (SQLite). Real market prices.
import sqlite3, secrets, hashlib, hmac, pandas as pd
from stable_baselines3 import DQN
from config import *
from data import latest
from features import make_features
from agent import pick_stock
import broker

_m = None
def model():
    global _m
    if _m is None: _m = DQN.load(MODEL_PATH)
    return _m

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c

def init():
    with db() as c:
        c.executescript("""
        create table if not exists users(id integer primary key autoincrement, email text unique, salt text, pw text);
        create table if not exists sessions(token text primary key, user_id int);
        create table if not exists account(user_id integer primary key, balance real);
        create table if not exists trades(id integer primary key autoincrement, user_id int, ticker text, qty real,
            buy_price real, buy_date text, end_date text, period int, amount real, status text,
            sell_price real, sell_date text, pnl real);""")

# ---- accounts ----
def _hash(pw, salt): return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()
def _session(c, uid):
    t = secrets.token_urlsafe(32); c.execute("insert into sessions values(?,?)", (t, uid)); return t

def signup(email, pw):
    email = email.strip().lower()
    if "@" not in email or len(pw) < 6: raise ValueError("Enter a valid email and a password of at least 6 characters.")
    salt = secrets.token_hex(16)
    with db() as c:
        try: cur = c.execute("insert into users(email, salt, pw) values(?,?,?)", (email, salt, _hash(pw, salt)))
        except sqlite3.IntegrityError: raise ValueError("This email is already registered. Log in instead.")
        return _session(c, cur.lastrowid)

def login(email, pw):
    with db() as c:
        u = c.execute("select * from users where email=?", (email.strip().lower(),)).fetchone()
        if not u or not hmac.compare_digest(u["pw"], _hash(pw, u["salt"])): raise ValueError("Wrong email or password.")
        return _session(c, u["id"])

def user(token):
    if not token: return None
    with db() as c:
        r = c.execute("select u.id, u.email from sessions s join users u on u.id=s.user_id where s.token=?", (token,)).fetchone()
    return dict(r) if r else None

def logout(token):
    with db() as c: c.execute("delete from sessions where token=?", (token,))

# ---- investing ----
def available(uid):
    """Stocks this user can get next: skips stocks picked earlier in the current round and stocks still held."""
    with db() as c:
        rows = c.execute("select ticker, status from trades where user_id=? order by id", (uid,)).fetchall()
    used = set()
    for r in rows:
        used.add(r["ticker"])
        if len(used) == len(TICKERS): used = set()      # all picked once: start a new round
    held = {r["ticker"] for r in rows if r["status"] == "open"}
    left = [t for t in TICKERS if t not in used and t not in held] or [t for t in TICKERS if t not in held] or TICKERS
    return [TICKERS.index(t) for t in left]

def span(d):
    """Trading days as plain words, e.g. 'about 3 months'."""
    if d < 21: n, u = max(1, round(d / 5)), "week"
    elif d < 252: n, u = round(d / 21), "month"
    else:
        n, u = round(d / 252, 1), "year"
        n = int(n) if n == int(n) else n
    return f"{n} {u}" + ("" if n == 1 else "s")

def plan(uid, amount, period):
    prices = latest(); feats = make_features(prices); row = feats.iloc[-1]
    allowed = available(uid)
    s, _ = pick_stock(model(), row.values, period, allowed)
    t = TICKERS[s]; price = float(prices[t].iloc[-1]); stop = stop_for(period)
    r20, vol, ma = float(row[t + "_r20"]), float(row[t + "_vol"]), float(row[t + "_ma"])
    reasons = []
    if r20 >= 0:
        reasons.append(dict(title="Its price has been rising", fact=f"Up {r20*100:.1f}% in the last month",
            text="A rising price means more people have been buying than selling lately. The AI likes this momentum, but a rise in the past does not guarantee a rise in the future."))
    else:
        reasons.append(dict(title="It has dipped recently", fact=f"Down {abs(r20)*100:.1f}% in the last month",
            text="The price has fallen lately. The AI weighs this with its other signals and sees a chance of a bounce back. That can pay off, but it is riskier because a dip can also continue."))
    calm = "calm" if vol < 0.01 else "moderate" if vol < 0.018 else "bumpy"
    reasons.append(dict(title=f"Its ups and downs are {calm}", fact=f"Moves about {vol*100:.1f}% on a typical day",
        text="This is called volatility. A calmer stock changes in value less from day to day, which is easier on a first-time investor. A bumpier one can gain or lose more quickly."))
    if ma > 0.03:
        reasons.append(dict(title="Trading above its recent average", fact=f"{ma*100:.1f}% above its 1-month average",
            text="The price is higher than its usual level over the past month, which is a sign of strength. It can also mean it has already run up, so further gains may be slower."))
    elif ma < -0.03:
        reasons.append(dict(title="Trading below its recent average", fact=f"{abs(ma)*100:.1f}% below its 1-month average",
            text="It costs less than it usually did over the past month. That can be a good time to buy, or a sign of weakness. The AI weighs this together with its other signals."))
    else:
        reasons.append(dict(title="Priced close to its usual level", fact=f"{ma*100:+.1f}% versus its 1-month average",
            text="The price is near its recent average, so it is not unusually expensive or cheap right now."))
    if period <= MAX_DAYS:
        reasons.append(dict(title="It suits your time frame", fact=f"You chose about {span(period)}",
            text="The AI learned from thousands of practice investments of similar length, so it has seen how stocks like these tend to behave over this kind of time."))
    else:
        reasons.append(dict(title="A longer investment", fact=f"You chose about {span(period)}",
            text="The AI was only trained on holding times up to about 3 months. For your longer plan, treat this as a sensible starting pick, not a forecast for the whole period."))
    skip = [NAMES[x] for i, x in enumerate(TICKERS) if i not in allowed]
    if skip:
        reasons.append(dict(title="Spreading your money", fact="Avoiding repeats",
            text="You already invested in " + ", ".join(skip) + " earlier, so we chose from the other companies. Spreading money across different companies reduces the damage if one of them does badly."))
    end = (prices.index[-1] + pd.offsets.BDay(period)).date().isoformat()
    return dict(ticker=t, name=NAMES[t], about=ABOUT[t], price=round(price, 2), amount=amount, period=period,
                end_date=end, stop_loss=round(price * (1 - stop), 2), stop_pct=round(stop * 100), reasons=reasons,
                summary=f"The AI looked at {len(allowed)} large Indian companies and scored {NAMES[t]} highest for a {span(period)} investment today.",
                price_date=prices.index[-1].date().isoformat())

def accept(uid, ticker, amount, period):
    prices = latest(); price = float(prices[ticker].iloc[-1]); d = prices.index[-1]
    qty = broker.buy(ticker, amount, price)
    with db() as c:
        c.execute("""insert into trades(user_id, ticker, qty, buy_price, buy_date, end_date, period, amount, status)
            values(?,?,?,?,?,?,?,?,'open')""", (uid, ticker, qty, price, d.date().isoformat(),
            (d + pd.offsets.BDay(period)).date().isoformat(), period, amount))

def update():
    """Sell every user's positions that hit the end date or the early-sell level."""
    prices = latest(); today = prices.index[-1].date().isoformat()
    with db() as c:
        for t in c.execute("select * from trades where status='open'").fetchall():
            price = float(prices[t["ticker"]].iloc[-1])
            if price <= t["buy_price"] * (1 - stop_for(t["period"])) or today >= t["end_date"]:
                proceeds = broker.sell(t["ticker"], t["qty"], price)
                c.execute("update trades set status='closed', sell_price=?, sell_date=?, pnl=? where id=?",
                          (price, today, proceeds - t["amount"], t["id"]))

def sell_now(uid, trade_id):
    """Sell one open investment early, at the latest available price."""
    prices = latest(); today = prices.index[-1].date().isoformat()
    with db() as c:
        t = c.execute("select * from trades where id=? and user_id=? and status='open'", (trade_id, uid)).fetchone()
        if not t: raise ValueError("This investment is not open.")
        price = float(prices[t["ticker"]].iloc[-1]); proceeds = broker.sell(t["ticker"], t["qty"], price)
        c.execute("update trades set status='closed', sell_price=?, sell_date=?, pnl=? where id=?",
                  (price, today, proceeds - t["amount"], t["id"]))

def portfolio(uid):
    prices = latest()
    with db() as c:
        rows = [dict(r) for r in c.execute("select * from trades where user_id=? order by id desc", (uid,))]
    for r in rows:
        if r["status"] == "open":
            r["price"] = float(prices[r["ticker"]].iloc[-1]); r["value"] = r["qty"] * r["price"]; r["pnl"] = r["value"] - r["amount"]
            s = stop_for(r["period"]); r["stop_price"] = r["buy_price"] * (1 - s); r["stop_pct"] = round(s * 100)
        r["name"] = NAMES[r["ticker"]]
    op = [r for r in rows if r["status"] == "open"]
    return dict(invested=sum(r["amount"] for r in op), open_value=sum(r["value"] for r in op),
                open_pnl=sum(r["pnl"] for r in op), closed_pnl=sum(r["pnl"] for r in rows if r["status"] != "open"), trades=rows)

def chart(ticker, days=120):
    s = latest()[ticker].tail(days)
    return dict(dates=[d.date().isoformat() for d in s.index], close=[round(float(v), 2) for v in s])