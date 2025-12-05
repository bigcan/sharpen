import os
from notion_client import Client
import json

# Read key from config
CONFIG_PATH = r"~\.gemini\antigravity\mcp_config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
    NOTION_TOKEN = config["mcpServers"]["notion"]["env"]["NOTION_API_KEY"]

DATABASE_ID = "1a1df15f-7333-8176-b748-d0125b00d2cf"

def main():
    notion = Client(auth=NOTION_TOKEN)
    
    print(f"Adding page to parent page {DATABASE_ID}...")
    
    try:
        # For child pages, the title property key is "title"
        page_props = {
            "title": [
                {
                    "text": {
                        "content": "FinRL_Podracer"
                    }
                }
            ]
        }
        res = notion.pages.create(parent={"page_id": DATABASE_ID}, properties=page_props)
        print(f"Successfully added page: {res['id']}")
    except Exception as e:
        print(f"Error adding page: {e}")

if __name__ == "__main__":
    main()
