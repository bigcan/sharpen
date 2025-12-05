import os
from notion_client import Client
import json
import pprint

# Read key from config
CONFIG_PATH = r"~\.gemini\antigravity\mcp_config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
    NOTION_TOKEN = config["mcpServers"]["notion"]["env"]["NOTION_API_KEY"]

DATABASE_ID = "1a1df15f-7333-8176-b748-d0125b00d2cf"

def main():
    notion = Client(auth=NOTION_TOKEN)
    
    print(f"Retrieving database {DATABASE_ID}...")
    try:
        db = notion.databases.retrieve(database_id=DATABASE_ID)
        print("Database retrieved successfully.")
        print("Properties:")
        pprint.pprint(db["properties"])
    except Exception as e:
        print(f"Error retrieving database: {e}")

if __name__ == "__main__":
    main()
