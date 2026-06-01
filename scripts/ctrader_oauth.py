#!/usr/bin/env python3
"""cTrader Open API OAuth 2.0 helper.

Run on the remote desktop (where Docker orchestrator lives).
Opens a browser for authorization, catches the callback, exchanges for tokens,
and saves credentials to .env.ctrader for Docker Compose.

Usage:
    # Step 1: Set your app credentials
    export CTRADER_CLIENT_ID=your_client_id
    export CTRADER_CLIENT_SECRET=your_client_secret

    # Step 2: Run the helper — opens browser, catches callback
    python scripts/ctrader_oauth.py

    # Step 3: (Optional) Refresh an expired token
    python scripts/ctrader_oauth.py --refresh

    # Step 4: Credentials saved to .env.ctrader — Docker reads them automatically
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ctrader-oauth")

# ── cTrader Open API endpoints ──────────────────────────────────────
AUTH_URL = "https://openapi.ctrader.com/apps/auth"
TOKEN_URL = "https://openapi.ctrader.com/apps/token"
ACCOUNTS_URL = "https://api.ctrader.com/connect/tradingaccounts"

CALLBACK_PORT = 5000
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/callback"
ENV_FILE = ".env.ctrader"


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP handler to capture the OAuth authorization code."""

    auth_code: str | None = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/callback" and "code" in params:
            OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Authorization successful!</h2>"
                b"<p>You can close this tab. Tokens are being saved...</p>"
                b"</body></html>"
            )
            # Shut down server after capturing code
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        elif parsed.path == "/callback" and "error" in params:
            error = params.get("error", ["unknown"])[0]
            desc = params.get("error_description", [""])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<html><body><h2>Authorization failed</h2>"
                f"<p>Error: {error}</p><p>{desc}</p></body></html>".encode()
            )
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP logs."""
        pass


def exchange_code_for_tokens(
    client_id: str, client_secret: str, auth_code: str
) -> dict:
    """Exchange authorization code for access + refresh tokens."""
    import urllib.request

    data = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": auth_code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
    }).encode()

    req = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def refresh_access_token(
    client_id: str, client_secret: str, refresh_token: str
) -> dict:
    """Use refresh token to get a new access token."""
    import urllib.request

    data = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode()

    req = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_trading_accounts(access_token: str) -> list[dict]:
    """Fetch trading accounts linked to this access token."""
    import urllib.request

    req = urllib.request.Request(
        f"{ACCOUNTS_URL}?access_token={access_token}",
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data if isinstance(data, list) else data.get("data", [])
    except Exception as e:
        logger.warning(f"Could not fetch accounts: {e}")
        return []


def save_env_file(
    client_id: str,
    client_secret: str,
    access_token: str,
    refresh_token: str,
    account_id: str,
    env_path: Path,
) -> None:
    """Write credentials to .env.ctrader (gitignored)."""
    content = (
        f"# cTrader Open API credentials\n"
        f"# Generated by scripts/ctrader_oauth.py\n"
        f"# DO NOT commit this file.\n"
        f"\n"
        f"CTRADER_CLIENT_ID={client_id}\n"
        f"CTRADER_CLIENT_SECRET={client_secret}\n"
        f"CTRADER_ACCESS_TOKEN={access_token}\n"
        f"CTRADER_REFRESH_TOKEN={refresh_token}\n"
        f"CTRADER_ACCOUNT_ID={account_id}\n"
    )
    env_path.write_text(content, encoding="utf-8")
    logger.info(f"Credentials saved to {env_path}")


def load_env_file(env_path: Path) -> dict[str, str]:
    """Load existing .env.ctrader if it exists."""
    env = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def prompt_account_selection(accounts: list[dict]) -> str:
    """Let user pick a trading account if multiple exist."""
    if not accounts:
        return input("\nEnter your cTrader Account ID manually: ").strip()

    if len(accounts) == 1:
        acct = accounts[0]
        acct_id = str(acct.get("login", acct.get("accountId", acct.get("id", ""))))
        broker = acct.get("brokerTitle", acct.get("broker", ""))
        is_live = acct.get("isLive", acct.get("live", False))
        mode = "LIVE" if is_live else "DEMO"
        logger.info(f"Found 1 account: {acct_id} ({broker}, {mode})")
        return acct_id

    print("\n  Available trading accounts:")
    print(f"  {'#':<4} {'Account ID':<15} {'Broker':<25} {'Type':<6} {'Balance'}")
    print(f"  {'─'*4} {'─'*15} {'─'*25} {'─'*6} {'─'*15}")

    for i, acct in enumerate(accounts, 1):
        acct_id = acct.get("login", acct.get("accountId", acct.get("id", "")))
        broker = acct.get("brokerTitle", acct.get("broker", ""))
        is_live = acct.get("isLive", acct.get("live", False))
        balance = acct.get("balance", "")
        mode = "LIVE" if is_live else "DEMO"
        print(f"  {i:<4} {str(acct_id):<15} {broker:<25} {mode:<6} {balance}")

    while True:
        choice = input(f"\n  Select account [1-{len(accounts)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(accounts):
            selected = accounts[int(choice) - 1]
            return str(
                selected.get("login", selected.get("accountId", selected.get("id", "")))
            )
        print(f"  Invalid choice. Enter 1-{len(accounts)}.")


def run_auth_flow(client_id: str, client_secret: str) -> None:
    """Full OAuth authorization flow: browser -> callback -> tokens -> save."""

    # Build auth URL
    auth_params = urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": "trading",
    })
    auth_url = f"{AUTH_URL}?{auth_params}"

    # Start callback server
    server = http.server.HTTPServer(
        ("127.0.0.1", CALLBACK_PORT), OAuthCallbackHandler
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("\n" + "=" * 60)
    print("  cTrader OAuth Authorization")
    print("=" * 60)
    print("\n  1. A browser window will open.")
    print("  2. Log in with your cTrader account (IC Markets demo or FTMO).")
    print("  3. Click 'Allow' to authorize the application.")
    print("  4. The browser will redirect back here automatically.\n")
    print("  If the browser doesn't open, copy this URL:\n")
    print(f"  {auth_url}\n")
    print("=" * 60)

    # Open browser on this machine (remote desktop)
    webbrowser.open(auth_url)

    logger.info(f"Waiting for callback on http://localhost:{CALLBACK_PORT}/callback ...")
    server_thread.join(timeout=300)  # 5 min timeout

    if not OAuthCallbackHandler.auth_code:
        logger.error("No authorization code received. Timed out or user denied.")
        sys.exit(1)

    auth_code = OAuthCallbackHandler.auth_code
    logger.info("Authorization code received. Exchanging for tokens...")

    # Exchange code for tokens
    token_data = exchange_code_for_tokens(client_id, client_secret, auth_code)

    access_token = token_data.get("accessToken", token_data.get("access_token", ""))
    refresh_token = token_data.get("refreshToken", token_data.get("refresh_token", ""))

    if not access_token:
        logger.error(f"Token exchange failed: {token_data}")
        sys.exit(1)

    logger.info("Access token obtained successfully.")

    # Fetch trading accounts
    accounts = fetch_trading_accounts(access_token)
    account_id = prompt_account_selection(accounts)

    # Save to .env.ctrader
    env_path = Path(ENV_FILE)
    save_env_file(client_id, client_secret, access_token, refresh_token, account_id, env_path)

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print(f"\n  Credentials saved to: {env_path.resolve()}")
    print(f"  Account ID: {account_id}")
    print("\n  To use with Docker Compose, add to your .env:")
    print(f"    source {env_path}")
    print("\n  Or pass directly:")
    print(f"    docker compose --env-file .env --env-file {env_path} up")
    print("\n  To refresh an expired token later:")
    print("    python scripts/ctrader_oauth.py --refresh")
    print("=" * 60 + "\n")


def run_refresh(client_id: str, client_secret: str) -> None:
    """Refresh an expired access token using stored refresh token."""
    env_path = Path(ENV_FILE)
    existing = load_env_file(env_path)

    refresh_token = existing.get("CTRADER_REFRESH_TOKEN", "")
    if not refresh_token:
        logger.error(f"No refresh token found in {env_path}. Run without --refresh first.")
        sys.exit(1)

    account_id = existing.get("CTRADER_ACCOUNT_ID", "")

    logger.info("Refreshing access token...")
    token_data = refresh_access_token(client_id, client_secret, refresh_token)

    new_access = token_data.get("accessToken", token_data.get("access_token", ""))
    new_refresh = token_data.get("refreshToken", token_data.get("refresh_token", refresh_token))

    if not new_access:
        logger.error(f"Refresh failed: {token_data}")
        sys.exit(1)

    save_env_file(client_id, client_secret, new_access, new_refresh, account_id, env_path)
    logger.info("Token refreshed successfully.")


def main():
    global CALLBACK_PORT, REDIRECT_URI, ENV_FILE

    parser = argparse.ArgumentParser(
        description="cTrader Open API OAuth helper — get/refresh trading credentials",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Refresh an expired access token (requires existing .env.ctrader)",
    )
    parser.add_argument(
        "--port", type=int, default=CALLBACK_PORT,
        help=f"Callback server port (default: {CALLBACK_PORT})",
    )
    parser.add_argument(
        "--env-file", type=str, default=ENV_FILE,
        help=f"Output env file path (default: {ENV_FILE})",
    )
    args = parser.parse_args()

    CALLBACK_PORT = args.port
    REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/callback"
    ENV_FILE = args.env_file

    # Get client credentials from env or existing .env.ctrader
    env_path = Path(ENV_FILE)
    existing = load_env_file(env_path)

    client_id = os.environ.get("CTRADER_CLIENT_ID", existing.get("CTRADER_CLIENT_ID", ""))
    client_secret = os.environ.get("CTRADER_CLIENT_SECRET", existing.get("CTRADER_CLIENT_SECRET", ""))

    if not client_id or not client_secret:
        print("\n  Missing cTrader app credentials. Set them first:\n")
        print("    export CTRADER_CLIENT_ID=your_client_id")
        print("    export CTRADER_CLIENT_SECRET=your_client_secret\n")
        print("  Find these at: https://openapi.ctrader.com → Applications\n")
        sys.exit(1)

    if args.refresh:
        run_refresh(client_id, client_secret)
    else:
        run_auth_flow(client_id, client_secret)


if __name__ == "__main__":
    main()
