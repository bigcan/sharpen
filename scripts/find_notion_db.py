import os
from notion_client import Client
import json

# Read key from config
CONFIG_PATH = r"~\.gemini\antigravity\mcp_config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
    NOTION_TOKEN = config["mcpServers"]["notion"]["env"]["NOTION_API_KEY"]

def main():
    notion = Client(auth=NOTION_TOKEN)
    
    print("Searching for EVERYTHING...")
    results = notion.search()
    
    for res in results["results"]:
        title = ""
        if "title" in res and res["title"]:
            title = res["title"][0]["plain_text"]
        elif "name" in res and res["name"]: # For some objects
             title = res["name"]
             
        print(f"Found: {title} (ID: {res['id']} Type: {res['object']})")
        with open("notion_all.txt", "a") as f:
            f.write(f"Name: {title} ID: {res['id']} Type: {res['object']}\n")

if __name__ == "__main__":
    main()
