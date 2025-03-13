# modules/macro_module.py
import requests
from config import FMP_API_KEY  # Separate API key for security

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3/"

def get_gdp_indicator():
    url = f"{FMP_BASE_URL}historical-economic-indicators/GDP?apikey={FMP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    # Check if data is empty
    if not data or len(data) == 0:
        # Log a warning or return a default value for testing
        print("Warning: No data returned from the API. Returning default indicator 'BULLISH'.")
        return "BULLISH"  # Default value for testing
    latest = data[0]
    if latest.get('value', 0) > 20000:
        return "BULLISH"
    else:
        return "BEARISH"

if __name__ == "__main__":
    print("GDP Signal:", get_gdp_indicator())
