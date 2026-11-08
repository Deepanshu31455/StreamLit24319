"""Sales Analytics Dashboard (Streamlit).

Upload CSV / Excel / JSON sales data, validate it, clean it, filter it and
explore KPIs and interactive charts.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
SUPPORTED_TYPES = ["csv", "xlsx", "json"]

# Column "roles" the dashboard needs, with common header aliases
# (aliases are compared after lower-casing and removing non-alphanumerics).
ROLE_ALIASES: Dict[str, List[str]] = {
    "order_id": ["orderid", "ordernumber", "orderno", "order", "invoiceno"],
    "order_date": ["orderdate", "date", "orderdt", "purchasedate", "invoicedate"],
    "sales": ["sales", "revenue", "amount", "totalsales", "salesamount"],
    "profit": ["profit", "netprofit", "totalprofit"],
    "quantity": ["quantity", "qty", "unitssold", "units", "quantitysold"],
}
REQUIRED_ROLES = ["order_id", "sales", "profit", "quantity"]
SUGGESTED_FILTER_COLUMNS = [
    "region", "category", "segment", "subcategory", "shipmode", "state", "country",
]
NULL_TOKENS = ["", "nan", "none", "null", "n/a", "na", "-", "--", "#n/a"]


# --------------------------------------------------------------------------
# Loading & validation
# --------------------------------------------------------------------------
def normalize_name(name: str) -> str:
    """Lower-case a column name and strip everything except letters/digits."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def get_excel_sheets(file_bytes: bytes) -> List[str]:
    """Return the sheet names of an Excel workbook."""
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def load_data(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV, XLSX or JSON file (as bytes) into a DataFrame.

    Raises ValueError with a readable message when the file cannot be used.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type '.{ext}'. Use CSV, XLSX or JSON.")
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    buffer = io.BytesIO(file_bytes)
    try:
        if ext == "csv":
            try:
                df = pd.read_csv(buffer)
            except UnicodeDecodeError:
                buffer.seek(0)
                df = pd.read_csv(buffer, encoding="latin-1")
        elif ext == "xlsx":
            df = pd.read_excel(buffer, sheet_name=sheet_name or 0)
        else:
            payload = json.loads(file_bytes.decode("utf-8-sig"))
            if isinstance(payload, dict):
                values = list(payload.values())
                lists = [v for v in values if isinstance(v, list)]
                if values and len(lists) == len(values) and len({len(v) for v in lists}) == 1 \
                        and not any(isinstance(x, dict) for v in lists for x in v):
                    df = pd.DataFrame(payload)  # column-oriented: {"col": [..], ...}
                    payload = None
                elif lists:
                    payload = max(lists, key=len)  # wrapper such as {"data": [...records...]}
                else:
                    payload = [payload]  # a single record
            if payload is not None:
                df = pd.json_normalize(payload)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not parse the file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if df.empty:
        raise ValueError("The file contains no data rows.")
    return df


def validate_data(df: pd.DataFrame) -> Dict[str, object]:
    """Profile a DataFrame: shape, dtypes, missing values and duplicates."""
    missing = df.isna().sum()
    info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(t) for t in df.dtypes],
            "Missing": missing.values,
            "Missing %": (missing.values / max(len(df), 1) * 100).round(2),
            "Unique": [df[c].nunique(dropna=True) for c in df.columns],
        }
    )
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_info": info,
        "total_missing": int(missing.sum()),
        "duplicate_count": int(df.duplicated().sum()),
    }


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Guess which column plays each role (order id, date, sales, ...)."""
    normalized = {c: normalize_name(c) for c in df.columns}
    roles: Dict[str, Optional[str]] = {}
    used: set = set()
    for role, aliases in ROLE_ALIASES.items():
        match = None
        for alias in aliases:  # exact match first, in alias priority order
            match = next((c for c, n in normalized.items() if n == alias and c not in used), None)
            if match:
                break
        if match is None:  # then partial match
            for alias in aliases:
                match = next((c for c, n in normalized.items() if alias in n and c not in used), None)
                if match:
                    break
        roles[role] = match
        if match:
            used.add(match)
    return roles


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def to_numeric_clean(series: pd.Series) -> pd.Series:
    """Convert text like '$1,234.50' to numbers; unparseable values become NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = series.astype("string").str.replace(r"[,$€£₹\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_dates(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    """Parse mixed date formats into datetime64; invalid values become NaT."""
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, format="mixed")


def clean_data(
    df: pd.DataFrame,
    roles: Dict[str, Optional[str]],
    numeric_strategy: str = "median",
    dayfirst: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean a raw DataFrame and return (clean_df, report).

    Steps: tidy headers -> null tokens -> text standardisation -> numeric and
    date conversion -> duplicate removal -> missing value handling.
    numeric_strategy: 'median', 'zero' or 'drop'.
    """
    log: List[Tuple[str, str, int]] = []
    out = df.copy()

    # 1. Column names
    out.columns = [str(c).strip() for c in out.columns]
    roles = {r: (c.strip() if isinstance(c, str) else c) for r, c in roles.items()}

    # 2. Whitespace + null tokens in text columns
    text_cols = [
        c for c in out.columns
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c])
    ]
    for col in text_cols:
        s = out[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        is_null_token = s.str.lower().isin(NULL_TOKENS)
        if is_null_token.sum():
            log.append(("Null tokens -> NaN", col, int(is_null_token.sum())))
        out[col] = s.mask(is_null_token)

    # 3. Numeric columns
    for role in ("sales", "profit", "quantity"):
        col = roles.get(role)
        if col in out.columns:
            before = out[col].isna().sum()
            out[col] = to_numeric_clean(out[col])
            bad = int(out[col].isna().sum() - before)
            if bad > 0:
                log.append(("Non-numeric values -> NaN", col, bad))

    # 4. Dates
    date_col = roles.get("order_date")
    if date_col in out.columns:
        before = out[date_col].isna().sum()
        out[date_col] = parse_dates(out[date_col], dayfirst)
        bad = int(out[date_col].isna().sum() - before)
        if bad > 0:
            log.append(("Unparseable dates", date_col, bad))

    # 5. Standardise inconsistent categorical text (e.g. ' west', 'WEST' -> 'West')
    skip = {roles.get("order_id"), date_col}
    for col in text_cols:
        if col in skip or col not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        original = out[col]
        standardised = original.str.title()
        changed = int((original.fillna("") != standardised.fillna("")).sum())
        if changed:
            log.append(("Text standardised (Title Case)", col, changed))
        out[col] = standardised

    # 6. Order IDs: trim + upper-case so 'ca-1' and 'CA-1' match
    id_col = roles.get("order_id")
    if id_col in out.columns:
        out[id_col] = out[id_col].astype("string").str.upper()
        missing_ids = int(out[id_col].isna().sum())
        if missing_ids:
            out = out[out[id_col].notna()]
            log.append(("Rows dropped: missing Order ID", id_col, missing_ids))

    # 7. Duplicates
    dups = int(out.duplicated().sum())
    if dups:
        out = out.drop_duplicates()
        log.append(("Duplicate rows removed", "all columns", dups))

    # 8. Rows with invalid dates cannot be filtered/plotted over time
    if date_col in out.columns:
        bad_dates = int(out[date_col].isna().sum())
        if bad_dates:
            out = out[out[date_col].notna()]
            log.append(("Rows dropped: invalid Order Date", date_col, bad_dates))

    # 9. Missing numeric values
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for col in numeric_cols:
        n_missing = int(out[col].isna().sum())
        if not n_missing:
            continue
        if numeric_strategy == "drop":
            out = out[out[col].notna()]
            log.append(("Rows dropped: missing numeric", col, n_missing))
        elif numeric_strategy == "zero":
            out[col] = out[col].fillna(0)
            log.append(("Filled with 0", col, n_missing))
        else:
            out[col] = out[col].fillna(out[col].median())
            log.append(("Filled with median", col, n_missing))

    # 10. Missing categorical values
    for col in text_cols:
        if col in out.columns and col not in skip:
            n_missing = int(out[col].isna().sum())
            if n_missing:
                out[col] = out[col].fillna("Unknown")
                log.append(("Filled with 'Unknown'", col, n_missing))

    out = out.reset_index(drop=True)
    report = pd.DataFrame(log, columns=["Step", "Column", "Rows affected"])
    return out, report


# --------------------------------------------------------------------------
# Statistics & KPIs
# --------------------------------------------------------------------------
def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics (count, mean, median, std, quartiles, skew...)."""
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    summary = num.describe().T
    summary["median"] = num.median()
    summary["skew"] = num.skew()
    summary["variance"] = num.var()
    return summary[["count", "mean", "median", "std", "variance", "min", "25%", "75%", "max", "skew"]].round(2)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Unique counts, most frequent value and its share for text columns."""
    rows = []
    for col in df.select_dtypes(exclude=["number", "datetime"]).columns:
        counts = df[col].value_counts(dropna=True)
        if counts.empty:
            continue
        rows.append(
            {
                "Column": col,
                "Unique": int(counts.size),
                "Top value": str(counts.index[0]),
                "Top count": int(counts.iloc[0]),
                "Top share %": round(counts.iloc[0] / len(df) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def compute_kpis(df: pd.DataFrame, roles: Dict[str, Optional[str]]) -> Dict[str, float]:
    """Total Sales, Total Profit, Total Orders (unique IDs), Quantity, Margin %."""
    total_sales = float(df[roles["sales"]].sum())
    total_profit = float(df[roles["profit"]].sum())
    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "total_orders": int(df[roles["order_id"]].nunique()),
        "total_quantity": float(df[roles["quantity"]].sum()),
        "profit_margin": (total_profit / total_sales * 100) if total_sales else 0.0,
    }


def grouped_summary(df: pd.DataFrame, dimension: str, roles: Dict[str, Optional[str]]) -> pd.DataFrame:
    """KPIs per value of a categorical dimension."""
    g = df.groupby(dimension, dropna=False).agg(
        Sales=(roles["sales"], "sum"),
        Profit=(roles["profit"], "sum"),
        Orders=(roles["order_id"], "nunique"),
        Quantity=(roles["quantity"], "sum"),
    )
    g["Profit Margin %"] = np.where(g["Sales"] != 0, g["Profit"] / g["Sales"] * 100, 0.0)
    return g.sort_values("Sales", ascending=False).round(2).reset_index()


def monthly_trend(df: pd.DataFrame, roles: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Sales and profit aggregated by month."""
    date_col = roles["order_date"]
    trend = (
        df.set_index(date_col)[[roles["sales"], roles["profit"]]]
        .resample("MS")
        .sum()
        .reset_index()
    )
    return trend.rename(columns={date_col: "Month", roles["sales"]: "Sales", roles["profit"]: "Profit"})


def apply_filters(
    df: pd.DataFrame,
    roles: Dict[str, Optional[str]],
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
    selections: Dict[str, List[str]],
) -> pd.DataFrame:
    """Filter by date range and by selected values of categorical columns."""
    mask = pd.Series(True, index=df.index)
    date_col = roles.get("order_date")
    if date_col and date_range:
        start, end = date_range
        mask &= df[date_col].between(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    for col, values in selections.items():
        if col in df.columns and values:
            mask &= df[col].isin(values)
    return df[mask]


# --------------------------------------------------------------------------
# Sample data (deliberately messy so the cleaning steps have work to do)
# --------------------------------------------------------------------------
def generate_sample_data(n_orders: int = 700, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic, intentionally messy sales dataset."""
    rng = np.random.default_rng(seed)
    cats = {
        "Technology": ["Phones", "Laptops", "Accessories"],
        "Furniture": ["Chairs", "Tables", "Bookcases"],
        "Office Supplies": ["Paper", "Binders", "Storage"],
    }
    regions = ["West", "East", "Central", "South"]
    segments = ["Consumer", "Corporate", "Home Office"]
    rows = []
    for i in range(n_orders):
        order_id = f"CA-2025-{100000 + i}"
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 700)))
        region, segment = rng.choice(regions), rng.choice(segments)
        for _ in range(int(rng.integers(1, 4))):
            cat = rng.choice(list(cats))
            sub = rng.choice(cats[cat])
            qty = int(rng.integers(1, 10))
            price = {"Technology": 250, "Furniture": 180, "Office Supplies": 25}[cat] * rng.uniform(0.6, 1.6)
            sales = round(price * qty, 2)
            margin = {"Technology": 0.18, "Furniture": 0.05, "Office Supplies": 0.22}[cat] + rng.normal(0, 0.12)
            rows.append([order_id, date, region, segment, cat, sub, qty, sales, round(sales * margin, 2)])
    df = pd.DataFrame(
        rows,
        columns=["Order ID", "Order Date", "Region", "Segment", "Category", "Sub-Category", "Quantity", "Sales", "Profit"],
    )

    # --- inject data-quality problems ---
    fmts = ["%Y-%m-%d", "%d-%b-%Y", "%m/%d/%Y"]
    df["Order Date"] = [d.strftime(fmts[int(rng.integers(0, 3))]) for d in df["Order Date"]]
    for col in ["Region", "Segment", "Category"]:  # inconsistent text
        idx = rng.choice(df.index, size=len(df) // 12, replace=False)
        df.loc[idx, col] = [rng.choice([v.upper(), v.lower(), f" {v} "]) for v in df.loc[idx, col]]
    for col in ["Sales", "Profit", "Quantity", "Region", "Category"]:  # missing values
        idx = rng.choice(df.index, size=len(df) // 40, replace=False)
        df[col] = df[col].astype(object)
        df.loc[idx, col] = np.nan
    idx = rng.choice(df.index, size=len(df) // 50, replace=False)  # currency text
    df["Sales"] = df["Sales"].astype(object)
    df.loc[idx, "Sales"] = df.loc[idx, "Sales"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else v)
    df = pd.concat([df, df.sample(40, random_state=seed)], ignore_index=True)  # duplicates
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def chart_trend(df: pd.DataFrame, roles) :
    trend = monthly_trend(df, roles)
    fig = px.line(trend, x="Month", y=["Sales", "Profit"], markers=True, title="Monthly Sales and Profit")
    fig.update_layout(legend_title_text="", yaxis_title="Amount", hovermode="x unified")
    return fig


def chart_bar(df: pd.DataFrame, roles, dimension: str):
    g = grouped_summary(df, dimension, roles)
    return px.bar(
        g, x=dimension, y="Sales", color="Profit Margin %", color_continuous_scale="RdYlGn",
        hover_data=["Profit", "Orders", "Quantity"], title=f"Sales by {dimension} (colour = profit margin %)",
    )


def chart_scatter(df: pd.DataFrame, roles, dimension: str):
    qty = roles["quantity"]
    size = qty if (df[qty] > 0).all() else None  # bubble size needs positive values
    return px.scatter(
        df, x=roles["sales"], y=roles["profit"], color=dimension, size=size,
        opacity=0.65, title="Sales vs Profit per order line", hover_data=[roles["order_id"]],
    )


def chart_share(df: pd.DataFrame, roles, dimension: str):
    g = grouped_summary(df, dimension, roles)
    g = g[g["Sales"] > 0]
    return px.pie(g, names=dimension, values="Sales", hole=0.45, title=f"Sales share by {dimension}")


def chart_profit_hist(df: pd.DataFrame, roles):
    return px.histogram(df, x=roles["profit"], nbins=40, title="Profit distribution per order line")


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def fmt_money(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.2f}"


def render_validation(df: pd.DataFrame) -> None:
    v = validate_data(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{v['rows']:,}")
    c2.metric("Columns", v["columns"])
    c3.metric("Missing values", f"{v['total_missing']:,}")
    c4.metric("Duplicate rows", f"{v['duplicate_count']:,}")
    st.markdown("**Column names, data types and missing values**")
    st.dataframe(v["column_info"], width="stretch", hide_index=True)
    if v["duplicate_count"]:
        with st.expander(f"View {v['duplicate_count']} duplicate rows"):
            st.dataframe(df[df.duplicated(keep=False)].head(200), width="stretch")
    with st.expander("Preview first 20 rows"):
        st.dataframe(df.head(20), width="stretch")


def render_kpis(kpis: Dict[str, float]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales", fmt_money(kpis["total_sales"]))
    c2.metric("Total Profit", fmt_money(kpis["total_profit"]))
    c3.metric("Total Orders", f"{kpis['total_orders']:,}")
    c4.metric("Total Quantity", f"{kpis['total_quantity']:,.0f}")
    c5.metric("Profit Margin", f"{kpis['profit_margin']:.2f}%")


def sidebar_data_source():
    """Upload widget (or sample data). Returns (raw_df, source_name) or (None, None)."""
    st.sidebar.header("1. Data")
    uploaded = st.sidebar.file_uploader("Upload CSV, Excel or JSON", type=SUPPORTED_TYPES)
    use_sample = st.sidebar.checkbox("Use built-in sample data", value=uploaded is None)

    if uploaded is not None:
        try:
            data = uploaded.getvalue()
            sheet = None
            if uploaded.name.lower().endswith(".xlsx"):
                sheets = get_excel_sheets(data)
                if len(sheets) > 1:
                    sheet = st.sidebar.selectbox("Excel sheet", sheets)
            return load_data(data, uploaded.name, sheet), uploaded.name
        except Exception as exc:  # show any parsing problem to the user
            st.sidebar.error(f"Could not read file: {exc}")
            return None, None
    if use_sample:
        return generate_sample_data(), "sample data"
    return None, None


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Sales Analytics Dashboard", page_icon="📊", layout="wide")
    st.title("📊 Sales Analytics Dashboard")
    st.caption("Upload data, validate it, clean it, then explore KPIs and interactive charts.")

    raw, source = sidebar_data_source()
    if raw is None:
        st.info("Upload a CSV, Excel or JSON file in the sidebar, or tick **Use built-in sample data**.")
        return

    # ---- column mapping ----
    st.sidebar.header("2. Column mapping")
    detected = detect_columns(raw)
    options = ["(none)"] + list(raw.columns)
    roles: Dict[str, Optional[str]] = {}
    labels = {"order_id": "Order ID", "order_date": "Order date", "sales": "Sales",
              "profit": "Profit", "quantity": "Quantity"}
    with st.sidebar.expander("Adjust detected columns", expanded=any(detected[r] is None for r in REQUIRED_ROLES)):
        for role, label in labels.items():
            default = options.index(detected[role]) if detected[role] in options else 0
            choice = st.selectbox(label, options, index=default, key=f"map_{role}")
            roles[role] = None if choice == "(none)" else choice

    # ---- preprocessing options ----
    st.sidebar.header("3. Cleaning options")
    strategy = st.sidebar.selectbox(
        "Missing numeric values", ["median", "zero", "drop"],
        format_func=lambda s: {"median": "Fill with median", "zero": "Fill with 0", "drop": "Drop rows"}[s],
    )
    dayfirst = st.sidebar.checkbox("Dates are day-first (DD/MM/YYYY)", value=False)

    missing_roles = [labels[r] for r in REQUIRED_ROLES if roles.get(r) is None]
    chosen = [c for c in roles.values() if c]
    tab_raw, tab_clean, tab_stats, tab_dash = st.tabs(
        ["1 · Upload & validation", "2 · Preprocessing", "3 · Statistics", "4 · Dashboard"]
    )

    with tab_raw:
        st.subheader(f"Raw data — {source}")
        render_validation(raw)

    if missing_roles:
        for t in (tab_clean, tab_stats, tab_dash):
            with t:
                st.warning(f"Map these required columns in the sidebar first: {', '.join(missing_roles)}.")
        return
    if len(set(chosen)) != len(chosen):
        st.sidebar.error("The same column is mapped to more than one role.")
        return

    try:
        clean, report = clean_data(raw, roles, strategy, dayfirst)
    except Exception as exc:
        st.error(f"Cleaning failed: {exc}")
        return
    if clean.empty:
        st.error("No rows left after cleaning. Try a different missing-value option or check the column mapping.")
        return
    roles = {r: (c.strip() if isinstance(c, str) else c) for r, c in roles.items()}

    with tab_clean:
        st.subheader("Cleaning report")
        if report.empty:
            st.success("No issues found; nothing needed cleaning.")
        else:
            st.dataframe(report, width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        c1.metric("Rows before", f"{len(raw):,}")
        c2.metric("Rows after", f"{len(clean):,}")
        st.subheader("Cleaned data validation")
        render_validation(clean)
        st.download_button("⬇️ Download cleaned CSV", clean.to_csv(index=False).encode("utf-8"),
                           "cleaned_data.csv", "text/csv")

    with tab_stats:
        st.subheader("Numeric summary")
        st.dataframe(numeric_summary(clean), width="stretch")
        st.subheader("Categorical summary")
        cat_summary = categorical_summary(clean.drop(columns=[roles["order_id"]]))
        st.dataframe(cat_summary, width="stretch", hide_index=True)
        cat_cols = [c for c in clean.select_dtypes(exclude=["number", "datetime"]).columns if c != roles["order_id"]
                    and clean[c].nunique() <= 50]
        if cat_cols:
            dim = st.selectbox("Group by", cat_cols, key="stats_dim")
            st.dataframe(grouped_summary(clean, dim, roles), width="stretch", hide_index=True)

    # ---- filters ----
    st.sidebar.header("4. Filters")
    date_col = roles["order_date"]
    date_range = None
    if date_col:
        dmin, dmax = clean[date_col].min().date(), clean[date_col].max().date()
        picked = st.sidebar.date_input("Order date range", (dmin, dmax), min_value=dmin, max_value=dmax)
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            date_range = (pd.Timestamp(picked[0]), pd.Timestamp(picked[1]))
    filter_candidates = [c for c in clean.select_dtypes(exclude=["number", "datetime"]).columns
                         if c != roles["order_id"] and 1 < clean[c].nunique() <= 50]
    defaults = [c for c in filter_candidates if normalize_name(c) in SUGGESTED_FILTER_COLUMNS][:3] or filter_candidates[:3]
    filter_cols = st.sidebar.multiselect("Filter by columns", filter_candidates, default=defaults)
    selections = {c: st.sidebar.multiselect(c, sorted(clean[c].dropna().unique())) for c in filter_cols}

    filtered = apply_filters(clean, roles, date_range, selections)

    with tab_dash:
        if filtered.empty:
            st.warning("No data matches the current filters.")
            return
        st.caption(f"Showing {len(filtered):,} of {len(clean):,} rows")
        render_kpis(compute_kpis(filtered, roles))
        st.divider()

        dims = [c for c in filter_candidates] or [c for c in clean.columns if clean[c].nunique() <= 50]
        if not dims:
            st.info("No categorical column available for category charts.")
        else:
            dim = st.selectbox("Break charts down by", dims, key="chart_dim")
            if date_col:
                st.plotly_chart(chart_trend(filtered, roles), width="stretch")
            left, right = st.columns(2)
            left.plotly_chart(chart_bar(filtered, roles, dim), width="stretch")
            right.plotly_chart(chart_share(filtered, roles, dim), width="stretch")
            left2, right2 = st.columns(2)
            left2.plotly_chart(chart_scatter(filtered, roles, dim), width="stretch")
            right2.plotly_chart(chart_profit_hist(filtered, roles), width="stretch")
        with st.expander("Filtered data"):
            st.dataframe(filtered, width="stretch")
            st.download_button("⬇️ Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"),
                               "filtered_data.csv", "text/csv")


if __name__ == "__main__":
    main()