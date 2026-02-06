import os
import json
import pandas as pd
import gspread
import streamlit as st
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

def _clean_headers(headers):
    """Clean and deduplicate column headers."""
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

def _fix_private_key_newlines(credentials_dict: dict) -> dict:
    """
    Fix common Streamlit secrets/env formatting issue where private_key contains literal '\\n'
    instead of real newline characters '\n', causing invalid_grant: Invalid JWT Signature.
    """
    if isinstance(credentials_dict, dict) and "private_key" in credentials_dict and credentials_dict["private_key"]:
        credentials_dict["private_key"] = credentials_dict["private_key"].replace("\\n", "\n")
    return credentials_dict

def load_all_sheets(service_account_file: str, spreadsheet_name: str):
    """
    Load all sheets from a Google Spreadsheet.
    Works both locally (with JSON file) and on Streamlit Cloud (with secrets / env vars).
    
    Returns:
        dict: Dictionary with sheet names as keys and DataFrames as values
    """
    try:
        creds = None

        # ------------------------------------------------------------
        # 1) Streamlit secrets (deployed)
        # ------------------------------------------------------------
        try:
            # Method A: JSON string stored in Streamlit Secrets
            if "gcp_service_account_json" in st.secrets:
                credentials_dict = json.loads(st.secrets["gcp_service_account_json"])
                credentials_dict = _fix_private_key_newlines(credentials_dict)
                creds = Credentials.from_service_account_info(credentials_dict, scopes=SCOPES)

            # Method B: TOML table stored in Streamlit Secrets
            elif "gcp_service_account" in st.secrets:
                credentials_dict = dict(st.secrets["gcp_service_account"])
                credentials_dict = _fix_private_key_newlines(credentials_dict)
                creds = Credentials.from_service_account_info(credentials_dict, scopes=SCOPES)

        except Exception as e:
            # secrets not available / malformed -> fall through to env/file methods
            pass

        # ------------------------------------------------------------
        # 2) Environment variable (optional)
        #    GOOGLE_SERVICE_ACCOUNT_JSON = full JSON string
        # ------------------------------------------------------------
        if creds is None and os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"):
            try:
                credentials_dict = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
                credentials_dict = _fix_private_key_newlines(credentials_dict)
                creds = Credentials.from_service_account_info(credentials_dict, scopes=SCOPES)
            except Exception as e:
                st.warning(f"⚠️ Failed to load credentials from environment variable: {str(e)}")

        # ------------------------------------------------------------
        # 3) Local JSON file (local dev)
        # ------------------------------------------------------------
        if creds is None and service_account_file and os.path.exists(service_account_file):
            try:
                creds = Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
            except Exception as e:
                st.warning(f"⚠️ Failed to load credentials from file: {str(e)}")

        # If still no credentials
        if creds is None:
            st.error(
                "❌ Could not find Google Sheets credentials!\n\n"
                "**For Streamlit Cloud:**\n"
                "1. Go to Settings → Secrets\n"
                "2. Add `[gcp_service_account]` TOML or `gcp_service_account_json`\n\n"
                "**For Local Development:**\n"
                f"Place '{service_account_file}' in the project directory"
            )
            st.stop()

        # ------------------------------------------------------------
        # Connect and load sheets
        # ------------------------------------------------------------
        try:
            client = gspread.authorize(creds)
        except Exception as e:
            st.error(f"❌ Failed to authorize with Google Sheets: {str(e)}")
            st.stop()

        try:
            spreadsheet = client.open(spreadsheet_name)
        except gspread.exceptions.SpreadsheetNotFound:
            st.error(f"❌ Spreadsheet '{spreadsheet_name}' not found. Please check the name and sharing permissions.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error opening spreadsheet: {str(e)}")
            st.stop()

        # Load all worksheets
        data = {}
        failed_sheets = []
        
        try:
            worksheets = spreadsheet.worksheets()
        except Exception as e:
            st.error(f"❌ Error retrieving worksheets: {str(e)}")
            st.stop()

        for ws in worksheets:
            try:
                values = ws.get_all_values()
                
                # Handle empty sheets
                if not values or len(values) < 1:
                    data[ws.title] = pd.DataFrame()
                    continue
                
                # Handle sheets with only headers
                if len(values) < 2:
                    headers = _clean_headers(values[0])
                    data[ws.title] = pd.DataFrame(columns=headers)
                    continue

                # Normal processing
                headers = _clean_headers(values[0])
                df = pd.DataFrame(values[1:], columns=headers)
                
                # Try to convert numeric columns
                df = df.apply(pd.to_numeric, errors="ignore")
                
                data[ws.title] = df
                
            except Exception as e:
                st.warning(f"⚠️ Error loading sheet '{ws.title}': {str(e)}")
                failed_sheets.append(ws.title)
                # Create empty DataFrame as fallback
                data[ws.title] = pd.DataFrame()

        # Show summary only if there were failures
        if failed_sheets:
            st.warning(f"⚠️ Failed to load {len(failed_sheets)} sheet(s): {', '.join(failed_sheets)}")
        
        if not data:
            st.error("❌ No sheets were loaded successfully")
            st.stop()

        return data

    except Exception as e:
        st.error(f"❌ Unexpected error loading Google Sheets: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()