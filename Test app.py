"""Unit tests: run with `pytest -q`."""
import io
import json

import pandas as pd
import pytest

import app

ROLES = {"order_id": "Order ID", "order_date": "Order Date", "sales": "Sales",
         "profit": "Profit", "quantity": "Quantity"}


@pytest.fixture
def messy():
    return pd.DataFrame({
        "Order ID": ["a-1", "A-1", "B-2", "B-2", "C-3", None],
        "Order Date": ["2025-01-05", "01/05/2025", "05-Feb-2025", "05-Feb-2025", "not a date", "2025-03-01"],
        "Region": [" west", "WEST", "East", "East", "East", "East"],
        "Sales": ["$1,000.00", 1000, 200, 200, 50, 10],
        "Profit": [100, 100, None, None, 5, 1],
        "Quantity": [2, 2, 1, 1, 1, 1],
    })


def test_load_csv_json_xlsx(messy):
    csv = messy.to_csv(index=False).encode()
    assert app.load_data(csv, "x.csv").shape == messy.shape
    js = messy.to_json(orient="records").encode()
    assert app.load_data(js, "x.json").shape == messy.shape
    nested = json.dumps({"data": messy.to_dict(orient="records")}).encode()
    assert app.load_data(nested, "x.json").shape == messy.shape
    buf = io.BytesIO()
    messy.to_excel(buf, index=False)
    assert app.load_data(buf.getvalue(), "x.xlsx").shape == messy.shape


def test_load_errors():
    with pytest.raises(ValueError):
        app.load_data(b"abc", "x.txt")
    with pytest.raises(ValueError):
        app.load_data(b"", "x.csv")
    with pytest.raises(ValueError):
        app.load_data(b"{not json", "x.json")


def test_validate(messy):
    v = app.validate_data(messy)
    assert v["rows"] == 6 and v["columns"] == 6
    assert v["duplicate_count"] == 1
    assert v["total_missing"] == 3
    assert set(v["column_info"].columns) >= {"Column", "Data Type", "Missing"}


def test_detect_columns(messy):
    assert app.detect_columns(messy) == ROLES
    assert app.detect_columns(pd.DataFrame({"qty": [1], "Revenue": [1]}))["sales"] == "Revenue"


def test_clean(messy):
    clean, report = app.clean_data(messy, ROLES, "median")
    assert clean["Region"].tolist().count("West") == 1 or "West" in clean["Region"].values
    assert clean["Sales"].dtype.kind in "fi"
    assert clean["Order ID"].notna().all()
    assert clean["Order Date"].notna().all()           # invalid date row dropped
    assert not clean.duplicated().any()
    assert clean["Profit"].isna().sum() == 0
    assert not report.empty


def test_clean_strategies(messy):
    zero, _ = app.clean_data(messy, ROLES, "zero")
    drop, _ = app.clean_data(messy, ROLES, "drop")
    assert (zero["Profit"] == 0).any()
    assert len(drop) < len(zero)


def test_kpis():
    df = pd.DataFrame({"Order ID": ["1", "1", "2"], "Sales": [100.0, 50.0, 50.0],
                       "Profit": [20.0, 10.0, 0.0], "Quantity": [1, 2, 3]})
    k = app.compute_kpis(df, ROLES)
    assert k["total_sales"] == 200 and k["total_profit"] == 30
    assert k["total_orders"] == 2 and k["total_quantity"] == 6
    assert k["profit_margin"] == pytest.approx(15.0)


def test_kpis_zero_sales():
    df = pd.DataFrame({"Order ID": ["1"], "Sales": [0.0], "Profit": [0.0], "Quantity": [1]})
    assert app.compute_kpis(df, ROLES)["profit_margin"] == 0.0


def test_sample_pipeline_and_filters():
    raw = app.generate_sample_data(200)
    assert app.validate_data(raw)["duplicate_count"] > 0
    clean, _ = app.clean_data(raw, app.detect_columns(raw))
    assert not clean.duplicated().any()
    assert set(clean["Region"].unique()) <= {"West", "East", "Central", "South", "Unknown"}
    full = app.compute_kpis(clean, ROLES)
    west = app.apply_filters(clean, ROLES, None, {"Region": ["West"]})
    assert app.compute_kpis(west, ROLES)["total_sales"] < full["total_sales"]
    dated = app.apply_filters(clean, ROLES, (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31")), {})
    assert dated["Order Date"].max() <= pd.Timestamp("2024-03-31 23:59:59")
    assert not app.monthly_trend(clean, ROLES).empty
    assert not app.numeric_summary(clean).empty
    assert not app.grouped_summary(clean, "Category", ROLES).empty