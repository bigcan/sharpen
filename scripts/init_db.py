
import os
import time
import psycopg
from finrl_pro_ds.data.db import DatabaseClient

def init():
    dsn = os.getenv("FINRL_PRO_DB_DSN", "postgresql://postgres:password@localhost:5432/finrl_pro_ds")
    print(f"Connecting to {dsn}...")
    
    # Retry loop for DB readiness
    for i in range(10):
        try:
            conn = psycopg.connect(dsn)
            conn.close()
            print("Connection successful.")
            break
        except psycopg.OperationalError as e:
            print(f"Waiting for DB... ({i+1}/10)")
            time.sleep(2)
    else:
        print("Could not connect to DB.")
        exit(1)

    db = DatabaseClient(dsn=dsn)
    print("Initializing Schema...")
    db.init_schema()
    print("Initializing Feature Store...")
    db.init_feature_store()
    print("DB Initialization Complete.")

if __name__ == "__main__":
    init()
