import gspread
import os
import json
from typing import List, Dict, Any


JSON_IDENTITY = '1024'

#for ID: https://docs.google.com/spreadsheets/d/THIS_IS_THE_ID/edit#gid=0
SPREADSHEET_ID = '1OpaPaAgZ8GYe5upqgxkmDIk80-WQ2OJFyQE7ruHzODA' 
SERVICE_ACCOUNT_FILE = 'service_account.json'

def load_data_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads experimental results from a JSON file
    """
    print(f"Attempting to load data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: JSON data file not found at '{file_path}'. Check the path.")
        return []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                print(f"Successfully loaded {len(data)} records from JSON.")
                return data
            else:
                print("Error: JSON file content is not a list of objects.")
                return []
    except json.JSONDecodeError:
        print("Error: Could not decode JSON. Check the file for valid JSON format.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return []


def transform_data_for_sheets(data: List[Dict[str, Any]]) -> tuple[List[str], List[List[Any]]]:
    """
    Transforms a list of dictionaries into a header row and data rows.

    Args:
        data (list): List of dictionaries containing the experimental results.

    Returns:
        tuple: (headers, rows), where headers is a list of strings and rows is a list of lists (the data)
    """
    if not data:
        return [], []

    headers = list(data[0].keys())

    rows = []
    for item in data:
        row_values = [item.get(key, '') for key in headers]
        rows.append(row_values)

    return headers, rows

def write_data_to_sheets(spreadsheet_id: str, worksheet_name: str, data: List[Dict[str, Any]]):
    """
    Authenticates, opens the sheet, and writes the data.
    """
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"big error: Authentication file '{SERVICE_ACCOUNT_FILE}' not found.")
        return

    try:
        print("Authenticating with Google Sheets API...")
        gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)

        spreadsheet = gc.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)
        print(f"Successfully connected to '{spreadsheet.title}' ({worksheet_name}).")

        first_row = worksheet.row_values(1)

        if first_row:
            print("Sheet is not empty. Exiting to prevent data loss.")
            return

        headers, data_rows = transform_data_for_sheets(data)
        if not data_rows:
            print("No data rows to write. Exiting.")
            return

        all_rows_to_append = []
        
        if not first_row or first_row != headers:
            if not first_row:
                print("Worksheet appears empty. Writing headers first.")
            else:
                print("Headers are missing or mismatched. Writing headers first.")
            all_rows_to_append.append(headers)
        else:
            print("Headers already present (Row 1 matches data keys). Appending data rows only.")

        all_rows_to_append.extend(data_rows)

        result = worksheet.append_rows(
            values=all_rows_to_append, 
            value_input_option='USER_ENTERED'
        )
        
        range_updated = result.get('updates', {}).get('updatedRange')
        count = len(data_rows)
        print(f"\n--- SUCCESS ---")
        print(f"Successfully appended {count} rows of data.")
        print(f"Updated range: {range_updated}")
        print(f"Data written to: {spreadsheet.title} - {worksheet_name}")

    except gspread.exceptions.SpreadsheetNotFound:
        print(f"ERROR: Spreadsheet with ID '{spreadsheet_id}' not found. Check the ID.")
    except gspread.exceptions.WorksheetNotFound:
        print(f"ERROR: Worksheet named '{worksheet_name}' not found. Check the worksheet name.")
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
        print("Ensure the Service Account has 'Editor' access to the Google Sheet.")


if __name__ == "__main__":
    
    choice = int(input("Enter 1 for unified and 2 for enlarged: "))

    if choice == 1:
        JSON_DATA_PATH = '/Users/norranyu/Documents/coding/2025_programs/data_management/training_data_output/'+JSON_IDENTITY+'_unified_network.json'
        WORKSHEET_NAME = JSON_IDENTITY + '_unified'
    elif choice == 2:
        JSON_DATA_PATH = '/Users/norranyu/Documents/coding/2025_programs/data_management/training_data_output/'+JSON_IDENTITY+'_enlarged.json'
        WORKSHEET_NAME = JSON_IDENTITY + '_specific'
    else:
        print("Process aborted")

    if SPREADSHEET_ID == '': #safety
        print("Please update the SPREADSHEET_ID variable before running")
    else:
        results_data = load_data_from_json(JSON_DATA_PATH)

        if results_data:
            write_data_to_sheets(SPREADSHEET_ID, WORKSHEET_NAME, results_data)