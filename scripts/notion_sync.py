
import os
import requests
from datetime import datetime
import json
from dotenv import load_dotenv

# Load env from .env file if it exists
load_dotenv()

# Configuration
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
DATABASE_ID = "2fedf15f-7333-80c2-9a20-d1a74a284d1f"
NOTION_VERSION = "2022-06-28"

def update_mission_control(status, sharpe, logs, command=None):
    """
    Updates the Ralph Mission Control database in Notion.
    """
    if not NOTION_API_KEY:
        print("Error: NOTION_API_KEY environment variable not set.")
        return

    url = "https://api.notion.com/v1/pages"
    
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION
    }

    # First, we need to find if there is an active entry or create a new one. 
    # For simplicity in this first pass, we will create a new entry for every "run" or update the latest one if possible.
    # However, a better approach for "Mission Control" is to have a SINGLE row that represents the current state.
    # Let's search for an existing page to update first.

def get_mission_control_page_id():
    """
    Finds the singleton 'Ralph Status' page ID, or creates it if missing.
    """
    if not NOTION_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION
    }
    
    # query for Name="Ralph Status"
    search_url = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
    query = {
        "filter": {
            "property": "Name",
            "title": {
                "equals": "Ralph Status"
            }
        }
    }
    
    response = requests.post(search_url, headers=headers, json=query)
    if response.status_code == 200:
        results = response.json().get("results")
        if results:
            return results[0]["id"]
            
    # If not found, create it
    create_url = "https://api.notion.com/v1/pages"
    data = {
        "parent": { "database_id": DATABASE_ID },
        "properties": {
            "Name": { "title": [{"text": {"content": "Ralph Status"}}] },
            "Status": { "select": {"name": "Running"} },
            "Logs": { "rich_text": [{"text": {"content": "Initializing..."}}] }
        }
    }
    resp = requests.post(create_url, headers=headers, json=data)
    if resp.status_code == 200:
        return resp.json()["id"]
    return None

def update_mission_control(status, sharpe, logs, remark=None, command=None):
    """
    Updates the Ralph Mission Control database in Notion (Singleton Row).
    """
    if not NOTION_API_KEY:
        print("Error: NOTION_API_KEY environment variable not set.")
        return

    page_id = get_mission_control_page_id()
    if not page_id:
        print("Error: Could not find or create Mission Control page.")
        return

    url = f"https://api.notion.com/v1/pages/{page_id}"
    
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION
    }

    data = {
        "properties": {
            "Status": { "select": { "name": status } },
            "Sharpe": { "number": float(sharpe) },
            "Logs": { "rich_text": [{ "text": { "content": logs[-2000:] } }] },
            "Timestamp": { "date": { "start": datetime.now().isoformat() } }
        }
    }
    
    if remark:
         data["properties"]["Remark"] = { "rich_text": [{ "text": { "content": remark[:2000] } }] }

    if command:
         data["properties"]["Command"] = { "select": { "name": command } }

    resp = requests.patch(url, headers=headers, json=data)
    if resp.status_code == 200:
         print(f"Successfully updated Notion Mission Control (ID: {page_id})")
    else:
         print(f"Failed to update Notion: {resp.text}")

def log_run(run_id, iteration, outcome, sharpe, logs, remark=None, start_time=None, end_time=None):
    """
    Creates a NEW history entry for a completed run.
    """
    if not NOTION_API_KEY:
        return

    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION
    }
    
    name = f"Run {run_id} (Iter {iteration})"
    status = "Done" if outcome == "success" else "Error" # Mapping outcome to Status options if needed, or just use Outcome
    
    # Notion properties mapping
    data = {
        "parent": { "database_id": DATABASE_ID },
        "properties": {
            "Name": { "title": [{"text": {"content": name}}] },
            "Status": { "select": {"name": status} },
            "Sharpe": { "number": float(sharpe) },
            "Logs": { "rich_text": [{"text": {"content": f"Outcome: {outcome}\n{logs}"[-2000:]}}] },
            # We use Timestamp for the "Start/End" range if both provided, otherwise just End
        }
    }
    
    if remark:
         data["properties"]["Remark"] = { "rich_text": [{ "text": { "content": remark[:2000] } }] }
    
    if start_time and end_time:
         data["properties"]["Timestamp"] = { "date": { "start": start_time, "end": end_time } }
    elif end_time:
         data["properties"]["Timestamp"] = { "date": { "start": end_time } }
    else:
         data["properties"]["Timestamp"] = { "date": { "start": datetime.now().isoformat() } }

    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code == 200:
        print(f"Successfully logged run history: {name}")
    else:
        print(f"Failed to log history: {resp.text}")

def check_command():
    """
    Checks the latest command from Notion (Singleton Row).
    """
    if not NOTION_API_KEY:
        return None

    page_id = get_mission_control_page_id()
    if not page_id:
        return None

    url = f"https://api.notion.com/v1/pages/{page_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        props = response.json().get("properties", {})
        command_prop = props.get("Command", {}).get("select")
        if command_prop:
            return command_prop.get("name")
    return None

if __name__ == "__main__":
    # Test run
    import sys
    # Mock data for testing
    status = "Running"
    sharpe = 1.5
    logs = "System init... all systems go."
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            cmd = check_command()
            print(f"Current Command: {cmd}")
        else:
            update_mission_control(status, sharpe, logs)
    else:
        update_mission_control(status, sharpe, logs)
