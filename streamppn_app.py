import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from pathlib import Path

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Pan Pepín Orders & Inventory (10‑Day PAR)", layout="wide")
st.title("🥖 Pan Pepín Orders & Inventory Dashboard")
st.caption("Sun–Sat sales window • Order Tuesdays → Deliver Fridays @ 5:00 AM • PAR = 10 days")

# -----------------------------
# Constants & Paths
# -----------------------------
PACK_SIZES = {"Hamburger": 8, "Hot Dog Buns": 10}
VALID_PRODUCTS = list(PACK_SIZES.keys())
PAR_DAYS_DEFAULT = 10
DEFAULT_HISTORY_WEEKS = 8   # used for projections; adjustable in UI

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PO_HISTORY_PATH = DATA_DIR / "po_history.csv"
WEEKLY_HISTORY_PATH = DATA_DIR / "weekly_sales_history.csv"

# -----------------------------
# Helpers
# -----------------------------

def coerce_date(s):
    """Robust date parser for DATE column."""
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def last_full_sun_sat_week(today: datetime) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the most recent full Sunday–Saturday week BEFORE 'today'."""
    # Find the last Saturday before today
    dow = today.weekday()  # Monday=0 ... Sunday=6
    # Convert to pandas Timestamp (normalize date only)
    today_date = pd.Timestamp(today.date())
    # Saturday is 5 by Python weekday; but we want Sunday=6, Saturday=5 mapping consideration
    # We'll compute last Sunday: shift back to the previous Sunday strictly before today
    # Compute yesterday to ensure previous full week even if today is Sunday
    yday = today_date - pd.Timedelta(days=1)
    last_sunday = yday - pd.Timedelta(days=(yday.weekday() - 6) % 7)
    start = last_sunday
    end = start + pd.Timedelta(days=6)
    return (start.normalize(), end.normalize())


def next_weekday(d: datetime, target_weekday: int) -> datetime:
    """Return the next datetime on target_weekday (Mon=0..Sun=6), strictly after d."""
    days_ahead = (target_weekday - d.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return d + timedelta(days=days_ahead)


def next_order_and_delivery(now: datetime) -> tuple[datetime, datetime]:
    """Order placed on Tuesdays, delivered Fridays @ 05:00 AM local."""
    next_tue = next_weekday(now, 1)  # Tue
    next_fri = next_weekday(now, 4)  # Fri
    # Ensure delivery Friday is after the order Tuesday in the same cycle.
    if next_fri <= next_tue:
        next_fri = next_fri + timedelta(days=7)
    delivery_dt = datetime.combine(next_fri.date(), time(5, 0))
    return next_tue, delivery_dt


def end_of_day(dt: datetime) -> datetime:
    return datetime.combine(dt.date(), time(23, 59, 59))


def clean_sales_df(df: pd.DataFrame) -> pd.DataFrame:
    # Expected columns: DATE, Modifier Name (or Product), Modifier Sold (or Units)
    cols = {c.lower(): c for c in df.columns}
    # Attempt flexible mapping
    date_col = next((cols[k] for k in cols if k in ("date", "fecha")), None)
    prod_col = next((cols[k] for k in cols if k in ("modifier name", "product", "item", "sku")), None)
    units_col = next((cols[k] for k in cols if k in ("modifier sold", "units", "qty", "quantity")), None)
    if not all([date_col, prod_col, units_col]):
        raise ValueError("CSV must include DATE, Product (Modifier Name), and Units (Modifier Sold) columns.")

    out = pd.DataFrame({
        "DATE": coerce_date(df[date_col]),
        "Product": df[prod_col].astype(str).str.strip(),
        "Units": pd.to_numeric(df[units_col], errors="coerce").fillna(0).astype(int),
    })
    out = out[out["Product"].isin(VALID_PRODUCTS)]
    out = out.dropna(subset=["DATE"]).sort_values("DATE")
    return out


def sun_sat_week_key(d: pd.Timestamp) -> str:
    # Returns label like 2025-08-03_to_2025-08-09
    start = d - pd.Timedelta(days=(d.weekday() - 6) % 7)
    end = start + pd.Timedelta(days=6)
    return f"{start.date()}_to_{end.date()}"


def summarize_weekly_units(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Week", "Product", "Units"])  
    tmp = df.copy()
    tmp["Week"] = tmp["DATE"].dt.to_period("W-SAT").apply(lambda p: f"{(p.start_time - pd.Timedelta(days=6)).date()}_to_{p.end_time.date()}")
    weekly = tmp.groupby(["Week", "Product"], as_index=False)["Units"].sum()
    return weekly.sort_values(["Week", "Product"])  


def weekday_profile(df: pd.DataFrame, lookback_weeks: int) -> pd.DataFrame:
    """Compute average daily UNITS by weekday for each product over the last N full weeks."""
    if df.empty:
        return pd.DataFrame(columns=["Product", "Weekday", "AvgUnits"])  
    # Filter to last N full weeks (Sun–Sat) using W-SAT periods
    df = df.copy()
    df["Period"] = df["DATE"].dt.to_period("W-SAT")
    periods = sorted(df["Period"].unique())[-lookback_weeks:]
    df = df[df["Period"].isin(periods)]
    df["Weekday"] = df["DATE"].dt.weekday  # Mon=0
    prof = df.groupby(["Product", "Weekday"], as_index=False)["Units"].mean().rename(columns={"Units": "AvgUnits"})
    return prof


def to_packs(units: float, product: str) -> float:
    return units / PACK_SIZES[product]


def simulate_consumption_until_delivery(on_hand_packs: dict, prof_units_by_dow: pd.DataFrame, now: datetime, delivery_dt: datetime):
    """Day-by-day consumption (rounded to 2 decimals) until delivery_dt. Returns projected on-hand at delivery and per-day usage table."""
    # Build a dict of avg UNITS per weekday
    usage = {p: {int(row.Weekday): float(row.AvgUnits) for _, row in prof_units_by_dow[prof_units_by_dow["Product"]==p].iterrows()} for p in VALID_PRODUCTS}
    day = datetime.combine(now.date(), time(0,0))
    per_day_rows = []
    stock = {p: float(on_hand_packs.get(p, 0)) for p in VALID_PRODUCTS}

    while day < delivery_dt:
        dow = day.weekday()
        for p in VALID_PRODUCTS:
            daily_units = usage.get(p, {}).get(dow, 0.0)
            daily_packs = daily_units / PACK_SIZES[p]
            stock[p] = round(stock[p] - daily_packs, 4)
        per_day_rows.append({"Date": pd.Timestamp(day.date()), **{f"{p} Used (packs)": round(usage.get(p, {}).get(dow, 0.0)/PACK_SIZES[p], 3) for p in VALID_PRODUCTS}, **{f"{p} EoD Stock (packs)": round(stock[p], 3) for p in VALID_PRODUCTS}})
        day += timedelta(days=1)

    proj_at_delivery = {p: round(stock[p], 3) for p in VALID_PRODUCTS}
    per_day_df = pd.DataFrame(per_day_rows)
    return proj_at_delivery, per_day_df


def recommend_po(on_hand_packs_at_delivery: dict, avg_daily_packs: dict, par_days: int) -> dict:
    """Return recommended PO packs to reach PAR days of cover after delivery."""
    po = {}
    for p in VALID_PRODUCTS:
        target = par_days * avg_daily_packs.get(p, 0.0)
        need = target - on_hand_packs_at_delivery.get(p, 0.0)
        po[p] = int(np.ceil(max(0.0, need)))
    return po


def append_po_history(po_dict: dict, order_dt: datetime, delivery_dt: datetime, avg_daily_packs: dict, par_days: int):
    rows = []
    for p, qty in po_dict.items():
        rows.append({
            "OrderDate": order_dt.date(),
            "DeliveryDateTime": delivery_dt,
            "Product": p,
            "PacksOrdered": qty,
            "PackSize": PACK_SIZES[p],
            "PAR_Days": par_days,
            "AvgDailyPacks": round(avg_daily_packs.get(p, 0.0), 3)
        })
    new = pd.DataFrame(rows)
    if PO_HISTORY_PATH.exists():
        old = pd.read_csv(PO_HISTORY_PATH, parse_dates=["DeliveryDateTime"]) 
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(PO_HISTORY_PATH, index=False)


def write_weekly_history(weekly_df: pd.DataFrame):
    # Append-or-update by week/product. Simpler: overwrite full table snapshot.
    weekly_df.to_csv(WEEKLY_HISTORY_PATH, index=False)

# -----------------------------
# Sidebar Config
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    par_days = st.number_input("PAR level (days)", min_value=1, max_value=30, value=PAR_DAYS_DEFAULT, step=1)
    lookback_weeks = st.slider("Weeks of history for projections", min_value=2, max_value=26, value=DEFAULT_HISTORY_WEEKS)
    st.markdown("---")
    st.subheader("📦 Current On‑Hand (packs)")
    on_hand_packs = {}
    for p in VALID_PRODUCTS:
        on_hand_packs[p] = st.number_input(f"{p} (packs)", min_value=0.0, step=0.5, value=0.0, format="%.2f")
    st.markdown("---")
    st.caption("Upload sales CSV with columns: DATE, Modifier Name (Product), Modifier Sold (Units)")
    file = st.file_uploader("Upload sales CSV", type=["csv"]) 

# -----------------------------
# Load & Prepare Sales
# -----------------------------
if file is None:
    st.info("Upload a CSV to begin. The app filters to products: Hamburger, Hot Dog Buns.")
    st.stop()

raw = pd.read_csv(file)
sales = clean_sales_df(raw)

# Weekly history (Sun–Sat)
weekly = summarize_weekly_units(sales)
write_weekly_history(weekly)

# Last full Sun–Sat week summary (previous week)
now = datetime.now()
prev_start, prev_end = last_full_sun_sat_week(now)
mask_prev = (sales["DATE"].dt.date >= prev_start.date()) & (sales["DATE"].dt.date <= prev_end.date())
prev_week_df = sales.loc[mask_prev]
prev_week_summary = prev_week_df.groupby("Product", as_index=False)["Units"].sum()

# Projections profile (Avg UNITS per weekday) -> convert to packs/day
prof_units = weekday_profile(sales, lookback_weeks)
if prof_units.empty:
    st.warning("Not enough data to compute projections. Provide at least a few weeks of sales.")
    st.stop()

avg_daily_packs = {p: 0.0 for p in VALID_PRODUCTS}
for p in VALID_PRODUCTS:
    avg_units_per_day = prof_units.loc[prof_units["Product"]==p, "AvgUnits"].mean() if not prof_units.loc[prof_units["Product"]==p].empty else 0.0
    avg_daily_packs[p] = round(to_packs(avg_units_per_day, p), 4)

# Next cycle dates
order_dt, delivery_dt = next_order_and_delivery(now)

# Simulate Tue→Thu consumption (and more generally from today until delivery)
proj_at_delivery, per_day_usage = simulate_consumption_until_delivery(on_hand_packs, prof_units, now, delivery_dt)

# Recommended PO to reach PAR after delivery
po_reco = recommend_po(proj_at_delivery, avg_daily_packs, par_days)

# Risk flags Tue/Wed/Thu
risk_rows = []
for p in VALID_PRODUCTS:
    risk = {
        "Tuesday": False,
        "Wednesday": False,
        "Thursday": False,
    }
    for dname, offset in [("Tuesday", (1 - now.weekday()) % 7), ("Wednesday", (2 - now.weekday()) % 7), ("Thursday", (3 - now.weekday()) % 7)]:
        target_date = (now + timedelta(days=offset)).date()
        row = per_day_usage[per_day_usage["Date"].dt.date == target_date]
        if not row.empty:
            # EoD stock
            eod = float(row[f"{p} EoD Stock (packs)"].values[0])
            if eod < 0:
                risk[dname] = True
    risk_rows.append({"Product": p, **risk})
risk_df = pd.DataFrame(risk_rows)

# -----------------------------
# Layout
# -----------------------------
st.subheader("Last Full Week Sales (Sun–Sat)")
col1, col2 = st.columns([2,3], gap="large")
with col1:
    st.write(f"**Week:** {prev_start.date()} to {prev_end.date()}")
    if prev_week_summary.empty:
        st.info("No sales for the previous full week in uploaded data.")
    else:
        show_prev = prev_week_summary.copy()
        show_prev["Packs"] = show_prev.apply(lambda r: np.ceil(r["Units"] / PACK_SIZES[r["Product"]]), axis=1)
        st.dataframe(show_prev, use_container_width=True)

with col2:
    st.write("**Weekly Sales History (Sun–Sat, Units)**")
    st.dataframe(weekly, use_container_width=True)
    st.download_button("⬇️ Download Weekly Sales History CSV", data=weekly.to_csv(index=False), file_name="weekly_sales_history.csv")

st.markdown("---")

st.subheader("Projections & Coverage")
col3, col4, col5 = st.columns([1.4, 1.2, 1.4], gap="large")
with col3:
    st.write("**Average Daily Usage (packs)** – based on lookback weeks")
    avg_tbl = pd.DataFrame({
        "Product": VALID_PRODUCTS,
        "Avg Daily Packs": [avg_daily_packs[p] for p in VALID_PRODUCTS],
        "Pack Size": [PACK_SIZES[p] for p in VALID_PRODUCTS]
    })
    st.dataframe(avg_tbl, use_container_width=True)

with col4:
    st.write("**Days of Supply On‑Hand**")
    dos_rows = []
    for p in VALID_PRODUCTS:
        avg_use = max(1e-6, avg_daily_packs.get(p, 0.0))
        dos = round(on_hand_packs.get(p, 0.0) / avg_use, 2)
        dos_rows.append({"Product": p, "On‑Hand (packs)": on_hand_packs.get(p, 0.0), "Days Remaining": dos})
    st.dataframe(pd.DataFrame(dos_rows), use_container_width=True)

with col5:
    st.write("**Stockout Risk Before Delivery (Tue/Wed/Thu)**")
    st.dataframe(risk_df, use_container_width=True)

st.markdown("---")

st.subheader("Next Order Recommendation (to reach PAR after Fri 5:00 AM delivery)")
st.write(f"**Order Date:** {order_dt.date()}  •  **Delivery:** {delivery_dt}  •  **PAR:** {par_days} days")

po_rows = []
for p in VALID_PRODUCTS:
    po_rows.append({
        "Product": p,
        "Projected On‑Hand at Delivery (packs)": proj_at_delivery[p],
        "Avg Daily (packs)": avg_daily_packs[p],
        "Target Stock (packs)": round(par_days * avg_daily_packs[p], 3),
        "Recommended PO (packs)": po_reco[p]
    })
po_df = pd.DataFrame(po_rows)
st.dataframe(po_df, use_container_width=True)

# Allow saving PO to history
if st.button("✅ Create Purchase Order and Save to History"):
    append_po_history(po_reco, order_dt, delivery_dt, avg_daily_packs, par_days)
    st.success("Purchase Order saved to po_history.csv")

# Show / download PO History
if PO_HISTORY_PATH.exists():
    st.write("**Purchase Order History**")
    po_hist = pd.read_csv(PO_HISTORY_PATH, parse_dates=["DeliveryDateTime"]) 
    st.dataframe(po_hist.sort_values(["OrderDate", "Product"]).reset_index(drop=True), use_container_width=True)
    st.download_button("⬇️ Download PO History CSV", data=po_hist.to_csv(index=False), file_name="po_history.csv")
else:
    st.info("No PO history yet. Click the button above to save the first one.")

st.markdown("---")

st.caption(
    "Notes: Projections use average weekday demand from the last N full Sun–Sat weeks. "
    "Order is planned for the next Tuesday; delivery arrives Friday @ 05:00 AM. "
    "PAR target is applied to stock level right after the delivery.")
