"""
Setup and verify PostgreSQL database for distributed Optuna HPO.

Modes:
    default        — Test connectivity AND verify Optuna storage (create/delete test study)
    --test         — Test PostgreSQL connectivity only (simple query, report latency)
    --instructions — Print human-readable setup guide for the chosen provider

Examples:
    # Test connectivity
    python scripts/setup_distributed_hpo_db.py --db_url "postgresql://user:pass@host/db" --test

    # Full Optuna verification
    python scripts/setup_distributed_hpo_db.py --db_url "postgresql://user:pass@host/db"

    # Print Neon setup instructions
    python scripts/setup_distributed_hpo_db.py --instructions --provider neon

    # Print self-hosted setup instructions
    python scripts/setup_distributed_hpo_db.py --instructions --provider self_hosted
"""

import argparse
import logging
import sys
import time

logger = logging.getLogger("setup_distributed_hpo_db")


# ---------------------------------------------------------------------------
# Connectivity test
# ---------------------------------------------------------------------------

def test_connectivity(db_url: str) -> bool:
    """Connect to PostgreSQL, run a simple query, report success and latency."""
    try:
        import sqlalchemy
    except ImportError:
        logger.error(
            "sqlalchemy is not installed. Install with: pip install sqlalchemy psycopg2-binary"
        )
        return False

    logger.info("Testing PostgreSQL connectivity...")
    logger.info("URL (masked): %s", _mask_url(db_url))

    try:
        engine = sqlalchemy.create_engine(db_url, connect_args={"connect_timeout": 10})
        t0 = time.monotonic()
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text("SELECT 1"))
            row = result.fetchone()
            latency_ms = (time.monotonic() - t0) * 1000
        engine.dispose()
    except Exception as exc:
        logger.error("Connection FAILED: %s", exc)
        return False

    if row and row[0] == 1:
        logger.info("Connection OK  (latency: %.1f ms)", latency_ms)
        return True
    else:
        logger.error("Unexpected query result: %s", row)
        return False


# ---------------------------------------------------------------------------
# Optuna storage verification
# ---------------------------------------------------------------------------

def verify_optuna_storage(db_url: str) -> bool:
    """Create a throwaway Optuna study to verify the storage backend works."""
    try:
        import optuna
    except ImportError:
        logger.error("optuna is not installed. Install with: pip install optuna")
        return False

    study_name = "__distributed_hpo_setup_test__"
    storage_url = db_url

    logger.info("Verifying Optuna storage...")
    try:
        # Create test study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            load_if_exists=False,
        )
        logger.info("Created test study: %s (id=%d)", study.study_name, study._study_id)

        # Enqueue a dummy trial to verify read/write
        study.enqueue_trial({"x": 0.5})
        logger.info("Enqueued dummy trial OK")

        # Clean up
        optuna.delete_study(study_name=study_name, storage=storage_url)
        logger.info("Deleted test study OK")

    except optuna.exceptions.DuplicatedStudyError:
        logger.warning(
            "Test study '%s' already exists — deleting stale study and retrying.",
            study_name,
        )
        try:
            optuna.delete_study(study_name=study_name, storage=storage_url)
            return verify_optuna_storage(db_url)
        except Exception as exc:
            logger.error("Failed to clean up stale test study: %s", exc)
            return False
    except Exception as exc:
        logger.error("Optuna storage verification FAILED: %s", exc)
        return False

    logger.info("Optuna storage verification PASSED")
    return True


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

def print_neon_instructions() -> None:
    """Print step-by-step Neon free tier setup instructions."""
    print(
        """
=============================================================
  Neon Free Tier - PostgreSQL for Distributed Optuna HPO
=============================================================

1. Go to https://neon.tech and sign up (GitHub/Google SSO).

2. Create a new project:
   - Name:   finrl-hpo
   - Region: Pick the closest to your GPU workers
             (e.g., AWS us-east-1 for US Vast.ai,
                    AWS ap-southeast-1 for SG GPUHub)
   - Compute: Free tier (0.25 CU) is sufficient for Optuna metadata.

3. Create a database:
   - In the Neon dashboard, go to "Databases" tab.
   - Click "New Database".
   - Name:  optuna_hpo
   - Owner: (default role is fine)

4. Get the connection string:
   - Go to "Connection Details" in the dashboard.
   - Select "Connection string" format.
   - It looks like:
       postgresql://user:password@ep-xxx-yyy-123.us-east-1.aws.neon.tech/optuna_hpo?sslmode=require

5. Set the environment variable:
       export DISTRIBUTED_HPO_DB_URL="postgresql://user:password@ep-xxx.neon.tech/optuna_hpo?sslmode=require"

   Or add to your .env file:
       DISTRIBUTED_HPO_DB_URL=postgresql://user:password@ep-xxx.neon.tech/optuna_hpo?sslmode=require

6. Verify connectivity:
       python scripts/setup_distributed_hpo_db.py --db_url "$DISTRIBUTED_HPO_DB_URL" --test

7. Verify Optuna storage (full check):
       python scripts/setup_distributed_hpo_db.py --db_url "$DISTRIBUTED_HPO_DB_URL"

Notes:
  - Neon free tier: 0.5 GB storage, 1 project, always-on compute.
    Optuna metadata is tiny (~1 KB/trial), so this is more than enough
    for thousands of trials.
  - Neon auto-suspends after 5 min idle. First connection after idle
    takes ~1-2s (cold start). This is fine for HPO workers.
  - Workers need outbound internet to reach Neon. Vast.ai workers
    have this by default. GPUHub workers may need port 5432 open.
  - Always use ?sslmode=require for Neon connections.
=============================================================
"""
    )


def print_self_hosted_instructions() -> None:
    """Print instructions for installing PostgreSQL on a GPUHub instance."""
    print(
        """
=============================================================
  Self-Hosted PostgreSQL on GPUHub - Distributed Optuna HPO
=============================================================

Use this if you want the DB on the same network as your GPUHub
workers (lower latency, no external dependency).

1. SSH into a GPUHub instance that will host the DB:
       ssh gpuhub-1

2. Install PostgreSQL:
       sudo apt-get update
       sudo apt-get install -y postgresql postgresql-contrib

3. Start the service:
       sudo systemctl enable postgresql
       sudo systemctl start postgresql

4. Create the database and user:
       sudo -u postgres psql <<SQL
         CREATE USER optuna_user WITH PASSWORD 'your_secure_password';
         CREATE DATABASE optuna_hpo OWNER optuna_user;
         GRANT ALL PRIVILEGES ON DATABASE optuna_hpo TO optuna_user;
       SQL

5. Allow remote connections (if workers are on different machines):

   a. Edit postgresql.conf:
       sudo nano /etc/postgresql/*/main/postgresql.conf
       # Change: listen_addresses = '*'

   b. Edit pg_hba.conf to allow your worker IPs:
       sudo nano /etc/postgresql/*/main/pg_hba.conf
       # Add line:
       host  optuna_hpo  optuna_user  0.0.0.0/0  scram-sha-256

   c. Restart:
       sudo systemctl restart postgresql

6. Set the environment variable:
       export DISTRIBUTED_HPO_DB_URL="postgresql://optuna_user:your_secure_password@<gpuhub-ip>:5432/optuna_hpo"

7. Verify connectivity from your local machine:
       python scripts/setup_distributed_hpo_db.py --db_url "$DISTRIBUTED_HPO_DB_URL" --test

8. Verify Optuna storage (full check):
       python scripts/setup_distributed_hpo_db.py --db_url "$DISTRIBUTED_HPO_DB_URL"

Notes:
  - GPUHub instances may have firewalls. Ensure port 5432 is
    accessible from all worker IPs (including Vast.ai if mixed).
  - For Tailscale networks, use the Tailscale IP (100.x.x.x)
    which bypasses firewall issues.
  - PostgreSQL memory: default config is fine for Optuna metadata.
    No tuning needed.
  - Backup: Optuna studies are recreatable. No backup needed
    unless you want to preserve trial history.
=============================================================
"""
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_url(url: str) -> str:
    """Mask password in a database URL for safe logging."""
    # postgresql://user:PASSWORD@host/db -> postgresql://user:****@host/db
    import re

    return re.sub(r"(?<=:)[^:@]+(?=@)", "****", url, count=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Setup and verify PostgreSQL for distributed Optuna HPO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  %(prog)s --db_url "postgresql://user:pass@host/db" --test\n'
            '  %(prog)s --db_url "postgresql://user:pass@host/db"\n'
            "  %(prog)s --instructions --provider neon\n"
            "  %(prog)s --instructions --provider self_hosted\n"
        ),
    )
    parser.add_argument(
        "--db_url",
        type=str,
        default=None,
        help="PostgreSQL connection URL (or set DISTRIBUTED_HPO_DB_URL env var)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test connectivity only (no Optuna verification)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["neon", "self_hosted"],
        default="neon",
        help="Provider for --instructions mode (default: neon)",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print setup instructions for the chosen provider and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Instructions mode ---
    if args.instructions:
        if args.provider == "neon":
            print_neon_instructions()
        elif args.provider == "self_hosted":
            print_self_hosted_instructions()
        return

    # --- Connectivity / verification modes require db_url ---
    import os

    db_url = args.db_url or os.environ.get("DISTRIBUTED_HPO_DB_URL")
    if not db_url:
        logger.error(
            "No database URL provided. Use --db_url or set DISTRIBUTED_HPO_DB_URL env var."
        )
        sys.exit(1)

    # --- Test mode ---
    if args.test:
        ok = test_connectivity(db_url)
        sys.exit(0 if ok else 1)

    # --- Default mode: connectivity + Optuna verification ---
    if not test_connectivity(db_url):
        logger.error("Aborting: PostgreSQL connectivity failed.")
        sys.exit(1)

    if not verify_optuna_storage(db_url):
        logger.error("Aborting: Optuna storage verification failed.")
        sys.exit(1)

    logger.info("All checks passed. Database is ready for distributed HPO.")


if __name__ == "__main__":
    main()
