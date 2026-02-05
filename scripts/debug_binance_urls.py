import requests

def check_url(url):
    try:
        response = requests.head(url)
        print(f"Status: {response.status_code} | URL: {url}")
    except Exception as e:
        print(f"Error: {e} | URL: {url}")

symbol = "BTCUSDT"
base = "https://data.binance.vision/data/spot/monthly"

months = ["2020-01"]

print("Checking Depth Updates:")
for m in months:
    url = f"{base}/depthUpdate/{symbol}/{symbol}-depthUpdate-{m}.zip"
    check_url(url)

print("\nChecking Snapshots:")
for m in months:
    url = f"{base}/depthSnapshot/{symbol}/{symbol}-depthSnapshot-{m}.zip"
    check_url(url)
