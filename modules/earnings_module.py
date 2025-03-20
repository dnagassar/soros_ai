# modules/earnings_module.py
import requests
from config import FMP_API_KEY

def get_upcoming_earnings(symbol):
    url = f"https://financialmodelingprep.com/api/v3/earnings_calendar?symbol={symbol}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    return "EVENT_PENDING" if data else "NO_EVENT"

if __name__ == "__main__":
    print("Earnings Signal:", get_upcoming_earnings("AAPL"))
