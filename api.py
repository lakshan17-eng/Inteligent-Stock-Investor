# MODULE 6b - API + login + scheduler.  Run: uvicorn api:app --reload
import json, os, engine
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from config import *

app = FastAPI()
class Auth(BaseModel): email: str; password: str
class Req(BaseModel): amount: float; period: int
class Acc(Req): ticker: str
class Sell(BaseModel): id: int

def me(request: Request):
    u = engine.user(request.cookies.get("sid"))
    if not u: raise HTTPException(401, "Please log in.")
    return u

def check(r, uid):
    if r.period < MIN_DAYS or r.period > MAX_HOLD_DAYS: raise HTTPException(400, "Choose a period between 1 week and 30 years.")
    if r.amount < 1000: raise HTTPException(400, "Minimum amount is 1,000.")

def _auth(fn, a, resp):
    try: t = fn(a.email, a.password)
    except ValueError as e: raise HTTPException(400, str(e))
    resp.set_cookie("sid", t, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
    return {"ok": True}

@app.on_event("startup")
def start():
    engine.init()
    s = BackgroundScheduler(); s.add_job(engine.update, "interval", minutes=30); s.start()

@app.get("/")
def index(): return FileResponse("static/index.html")

@app.post("/api/signup")
def signup(a: Auth, resp: Response): return _auth(engine.signup, a, resp)

@app.post("/api/login")
def login(a: Auth, resp: Response): return _auth(engine.login, a, resp)

@app.post("/api/logout")
def logout(request: Request, resp: Response):
    engine.logout(request.cookies.get("sid")); resp.delete_cookie("sid"); return {"ok": True}

@app.get("/api/mode")
def mode(): return {"mode": MODE}

@app.get("/api/me")
def get_me(u=Depends(me)): return {"email": u["email"], "mode": MODE}

@app.post("/api/plan")
def plan(r: Req, u=Depends(me)):
    check(r, u["id"])
    try: return engine.plan(u["id"], r.amount, r.period)
    except Exception as e: raise HTTPException(500, f"Could not build a plan: {e}. Did you run 'python agent.py' to train the model?")

@app.post("/api/accept")
def accept(r: Acc, u=Depends(me)):
    check(r, u["id"])
    if r.ticker not in TICKERS: raise HTTPException(400, "Unknown stock.")
    try: engine.accept(u["id"], r.ticker, r.amount, r.period)
    except ValueError as e: raise HTTPException(400, str(e))
    return {"ok": True}

@app.post("/api/sell")
def sell(r: Sell, u=Depends(me)):
    try: engine.sell_now(u["id"], r.id)
    except ValueError as e: raise HTTPException(400, str(e))
    return {"ok": True}

@app.get("/api/portfolio")
def portfolio(u=Depends(me)): engine.update(); return engine.portfolio(u["id"])

@app.get("/api/chart/{ticker}")
def chart(ticker: str, u=Depends(me)):
    if ticker not in TICKERS: raise HTTPException(404, "Unknown stock.")
    return engine.chart(ticker)

@app.get("/api/report")
def report(u=Depends(me)):
    if not os.path.exists("reports/backtest.json"): raise HTTPException(404, "Run 'python backtest.py' first.")
    return json.load(open("reports/backtest.json"))