# modules/macro_module.py
import requests
from config import FMP_API_KEY

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3/"

def get_gdp_indicator():
    url = f"{FMP_BASE_URL}historical-economic-indicators/GDP?apikey={FMP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if not data:
        return "NEUTRAL"
    latest = data[0]
    return "BULLISH" if latest.get('value', 0) > 20000 else "BEARISH"

if __name__ == "__main__":
    print("GDP Signal:", get_gdp_indicator())
