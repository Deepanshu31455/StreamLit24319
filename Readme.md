# Sales Analytics Dashboard

A Streamlit app to upload sales data (CSV, Excel, JSON), validate and clean it, and explore it with filters, KPIs and interactive Plotly charts.

## Features
- **Upload & validation:** dimensions, column names, data types, missing values and duplicates.
- **Preprocessing:** duplicate removal, missing-value handling (median / zero / drop), text standardisation, currency-string parsing, mixed date-format parsing, and a cleaning report.
- **Statistics:** reusable functions for numeric, categorical and grouped summaries.
- **Filters:** date range plus any categorical columns (Region, Category, Segment by default).
- **KPIs:** Total Sales, Total Profit, Total Orders (unique Order IDs), Total Quantity, Profit Margin %.
- **Charts:** monthly trend, sales by dimension, sales share, sales vs profit scatter, profit histogram.
- Built-in messy sample data, plus `sample_data.csv` for upload testing.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q
```

## Deploy (Streamlit Community Cloud)
1. Push `app.py`, `requirements.txt`, `README.md` (and tests) to a public GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repo, branch `main`, main file `app.py`, then **Deploy**.

Expected columns (names auto-detected, adjustable in the sidebar): Order ID, Order Date, Sales, Profit, Quantity, plus any categorical columns.