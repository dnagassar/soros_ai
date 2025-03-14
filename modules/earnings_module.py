import requests
from config import FMP_API_KEY

def get_upcoming_earnings(symbol):
    url = f"https://financialmodelingprep.com/api/v3/earnings_calendar?symbol={symbol}&apikey={FMP_API_KEY}"
    response = requests.get(url).json()
    return "EVENT_PENDING" if response else "NO_EVENT"
