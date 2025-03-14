# modules/macro_module.py
import requests
import logging
from config import FMP_API_KEY

def get_gdp_indicator():
    url = f"https://financialmodelingprep.com/api/v3/historical-economic-indicators/GDP?apikey={FMP_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        logging.error(f"Failed to fetch GDP data: HTTP {response.status_code}")
        return "NEUTRAL"  # Return neutral if there's an API error

    data = response.json()

    if not data:
        logging.warning("GDP data is empty.")
        return "NEUTRAL"  # Neutral signal if no data is returned

    latest_gdp = data[0].get('value', None)

    if latest_gdp is None:
        logging.warning("GDP data does not contain 'value'.")
        return "NEUTRAL"

    # Example threshold, adjust according to your real-world scenario
    return "BULLISH" if latest_gdp > 20000 else "BEARISH"
