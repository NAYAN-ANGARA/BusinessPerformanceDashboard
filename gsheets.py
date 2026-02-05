import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
import json
import os

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
    """
    Load all sheets from Google Spreadsheet.
    Works both locally (with JSON file) and on Streamlit Cloud (with secrets).
    """
    try:
        creds = None
        
        # FIRST: Try to use Streamlit secrets (for deployed app)
        try:
            # Method 1: JSON string in secrets
            if 'gcp_service_account_json' in st.secrets:
                credentials_dict = json.loads(st.secrets['gcp_service_account_json'])
                creds = Credentials.from_service_account_info(credentials_dict, scopes=SCOPES)
            # Method 2: TOML format in secrets
            elif 'gcp_service_account' in st.secrets:
                credentials_dict = dict(st.secrets['gcp_service_account'])
                creds = Credentials.from_service_account_info(credentials_dict, scopes=SCOPES)
        except Exception as e:
            # Secrets not available or failed, will try JSON file next
            pass
        
        # SECOND: Try JSON file (for local development)
        if creds is None and os.path.exists(service_account_file):
            creds = Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
        
        # If still no credentials, show error
        if creds is None:
            st.error(
                "❌ Could not find Google Sheets credentials!\n\n"
                "**For Streamlit Cloud:**\n"
                "1. Go to Settings > Secrets\n"
                "2. Add your Google credentials\n\n"
                f"**For Local Development:**\n"
                f"Place '{service_account_file}' in the project directory"
            )
            st.stop()
        
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
    
    except Exception as e:
        st.error(f"❌ Error loading Google Sheets: {str(e)}")
        st.stop()
