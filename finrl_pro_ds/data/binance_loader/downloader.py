import os
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly"

def generate_monthly_urls(symbol, start_date, end_date):
    """
    Generates URLs for monthly depth updates and snapshots between start_date and end_date.
    """
    urls = []
    current_date = start_date.replace(day=1)
    end_date = end_date.replace(day=1)

    while current_date <= end_date:
        year_month = current_date.strftime("%Y-%m")

        # Depth Update URL
        # Format: https://data.binance.vision/data/spot/monthly/depthUpdate/BTCUSDT/BTCUSDT-depthUpdate-2023-01.zip
        update_url = f"{BASE_URL}/depthUpdate/{symbol}/{symbol}-depthUpdate-{year_month}.zip"

        # Depth Snapshot URL
        # Format: https://data.binance.vision/data/spot/monthly/depthSnapshot/BTCUSDT/BTCUSDT-depthSnapshot-2023-01.zip
        snapshot_url = f"{BASE_URL}/depthSnapshot/{symbol}/{symbol}-depthSnapshot-{year_month}.zip"

        # Klines URL
        # Format: https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2023-01.zip
        # Note: We are hardcoding '1m' interval for now as per requirement.
        kline_url = f"{BASE_URL}/klines/{symbol}/1m/{symbol}-1m-{year_month}.zip"

        urls.append({
            'date': year_month,
            'type': 'depthUpdate',
            'url': update_url,
            'filename': f"{symbol}-depthUpdate-{year_month}.zip"
        })
        urls.append({
            'date': year_month,
            'type': 'depthSnapshot',
            'url': snapshot_url,
            'filename': f"{symbol}-depthSnapshot-{year_month}.zip"
        })
        urls.append({
            'date': year_month,
            'type': 'klines',
            'url': kline_url,
            'filename': f"{symbol}-1m-{year_month}.zip"
        })

        current_date += relativedelta(months=1)

    return urls

def download_file(url, save_path):
    """
    Downloads a file from a URL with a progress bar.
    """
    if os.path.exists(save_path):
        logger.info(f"File already exists: {save_path}, skipping download.")
        return True

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192 # 8KB

        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=block_size):
                size = f.write(chunk)
                bar.update(size)
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path) # Clean up partial download
        return False

def verify_checksum(filepath, expected_checksum=None):
    """
    Verifies the SHA256 checksum of a file.
    Note: Binance Vision provides .CHECKSUM files (SHA256).
    Ideally, we should download the .CHECKSUM file and verify against it.
    For now, this function is a placeholder or can be extended to download the checksum file.
    """
    # TODO: Implement full checksum logic if typically required.
    # Binance Vision usually has a .CHECKSUM file alongside the .zip
    # e.g., .../BTCUSDT-depthUpdate-2023-01.zip.CHECKSUM
    pass

def main():
    parser = argparse.ArgumentParser(description="Download Binance LOB Data")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM)")
    parser.add_argument("--output_dir", type=str, default="raw_data", help="Directory to save downloaded files")

    args = parser.parse_args()

    start_dt = datetime.strptime(args.start_date, "%Y-%m")
    end_dt = datetime.strptime(args.end_date, "%Y-%m")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    download_tasks = generate_monthly_urls(args.symbol, start_dt, end_dt)

    logger.info(f"Found {len(download_tasks)} files to download for {args.symbol} from {args.start_date} to {args.end_date}")

    success_count = 0
    for task in download_tasks:
        file_path = os.path.join(args.output_dir, task['filename'])
        logger.info(f"Processing {task['type']} for {task['date']}...")
        if download_file(task['url'], file_path):
            success_count += 1

    logger.info(f"Completed. Successfully downloaded {success_count}/{len(download_tasks)} files.")

if __name__ == "__main__":
    main()
