import yfinance as yf

INDEX_INFO = {
    "S&P 500": {"ticker": "^GSPC", "vol_ticker": "^VIX", "rf_ticker": "^TNX"},
    "Nasdaq 100": {"ticker": "^NDX", "vol_ticker": "^VXN", "rf_ticker": "^TNX"},
    "Dow Jones": {"ticker": "^DJI", "vol_ticker": "^VXD", "rf_ticker": "^TNX"},
}


def safe_history(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")["Close"]
        return float(data.iloc[-1]) if not data.empty else None
    except:
        return None

def get_spot(ticker):
    spot = safe_history(ticker)
    return spot if spot is not None else 100.0 

def get_dividend_yield(ticker):
    if ticker == "^GSPC":
        return 0.0113
    if ticker == "^NDX":
        return 0.0067
    if ticker == "^DJI":
        return 0.0172

    
def get_vol(vol_ticker):
    if vol_ticker is None:
        return 0.20 
    try:
        data = yf.Ticker(vol_ticker).history(period="5d")["Close"]
        data = data.dropna()
        if data.empty:
            return 0.20
        return float(data.iloc[-1]) / 100
    except:
        return 0.20


def get_risk_free(rf_ticker):
    rf = safe_history(rf_ticker)
    return rf / 100 if rf is not None else 0.03

def get_market_data(index_name):
    info = INDEX_INFO[index_name]

    return {
        "spot": get_spot(info["ticker"]),
        "dividend_yield": get_dividend_yield(info["ticker"]),
        "volatility": get_vol(info["vol_ticker"]),
        "risk_free_rate": get_risk_free(info["rf_ticker"]),
        "ticker": info["ticker"],
    }
