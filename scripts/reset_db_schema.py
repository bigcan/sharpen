"""
Reset DB Schema Script
Drops LOB tables to clean state and re-initializes the schema.
"""
import sys
import os
import psycopg

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finrl_pro_ds.data.db import DatabaseClient

def reset_lob_schema():
    print("Initializing DatabaseClient...")
    db = DatabaseClient()
    dsn = db._dsn
    if not dsn:
        print("Error: FINRL_PRO_DB_DSN environment variable not set.")
        # Try to load from .env if possible, but python-dotenv might not be loaded yet
        from dotenv import load_dotenv
        load_dotenv()
        dsn = os.getenv("FINRL_PRO_DB_DSN")
        if not dsn:
            print("Fatal: Could not find DSN.")
            return

    print("Dropping LOB tables...")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS lob_snapshots CASCADE")
            # Also drop hypertable if needed, but CASCADE on table usually implies it for chunks? 
            # TimescaleDB might keep metadata. 
            # We can just drop the table.
        conn.commit()

    print("Re-initializing schema...")
    db.init_schema()
    print("Done. LOB Schema reset.")

if __name__ == "__main__":
    reset_lob_schema()
