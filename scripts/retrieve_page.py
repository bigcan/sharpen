import os
from notion_client import Client
import json
import pprint

# Read key from config
CONFIG_PATH = r"~\.gemini\antigravity\mcp_config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
    NOTION_TOKEN = config["mcpServers"]["notion"]["env"]["NOTION_API_KEY"]

PAGE_ID = "1a1df15f-7333-8113-94b7-000b58dc95e1"

def main():
    notion = Client(auth=NOTION_TOKEN)
    
    print(f"Retrieving page {PAGE_ID}...")
    try:
        page = notion.pages.retrieve(page_id=PAGE_ID)
        print("Page retrieved successfully.")
        print("Object type:", page["object"])
    except Exception as e:
        print(f"Error retrieving page: {e}")

if __name__ == "__main__":
    main()
