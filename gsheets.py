import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly"
]

def _clean_headers(headers):
    cleaned, seen = [], {}
    for i, h in enumerate(headers):
        h = h.strip() if h else f"column_{i+1}"
        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"
        else:
            seen[h] = 0
        cleaned.append(h)
    return cleaned

def load_all_sheets(service_account_file, spreadsheet_name):
    creds = Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open(spreadsheet_name)

    data = {}
    for ws in spreadsheet.worksheets():
        values = ws.get_all_values()
        if len(values) < 2:
            data[ws.title] = pd.DataFrame()
            continue
        headers = _clean_headers(values[0])
        df = pd.DataFrame(values[1:], columns=headers)
        df = df.apply(pd.to_numeric, errors="ignore")
        data[ws.title] = df

    return data