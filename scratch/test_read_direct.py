import os
import sys
import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

# Mock Streamlit secrets and run in a context
# We can use the internal connection object directly or instantiate it.
try:
    # Initialize connection
    # For streamlit connections, we can call st.connection inside streamlit,
    # but outside we might need to mock or initialize it.
    # Let's see if we can read the spreadsheet.
    # Let's create a dummy streamlit app and run it headlessly,
    # or just use gspread directly to test connection with the URL.
    import gspread
    print("gspread version:", gspread.__version__)
    
    # Let's try to fetch public spreadsheet metadata using requests
    import requests
    url = "https://docs.google.com/spreadsheets/d/1KvrFlNj3czRAEAmPzvsnKD6mqamWJvb6LpzlV1tuwyo/gviz/tq?tqx=out:csv"
    res = requests.get(url)
    if res.status_code == 200:
        print("Spreadsheet CSV export successful. Length:", len(res.text))
        print("First line:", res.text.split("\n")[0] if res.text else "empty")
    else:
        print("CSV export failed status code:", res.status_code)
except Exception as e:
    print("Error:", e)
