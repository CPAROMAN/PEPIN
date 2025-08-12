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
st.caption("Sun–Sat weekly totals • Order Tuesdays → Deliver Fridays @ 5:00 AM • PAR = 10 days")

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

def last_full_sun_sat_week(today: datetime) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the most recent full Sunday–Saturday week BEFORE 'today'."""
    today_date = pd.Timestamp(today.date())
    yday = today_date - pd.Timedelta(days=1)
    last_sunday = yday - pd.Timedelta(days=(yday.weekday() - 6) % 7)
    start = last_sunday
    end = start + pd.Timedelta(days=6)
    return (start.normalize(), end.normalize())


def next_weekday(d: datetime, target_weekday: int) -> datetime:
    days_ahead = (target_weekday - d.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return d + timedelta(days=days_ahead)


def next_order_and_delivery(now: datetime) -> tuple[datetime, datetime]:
    """Order placed on Tuesdays, delivered Fridays @ 05:00 AM local."""
    next_tue = next_weekday(now, 1)  # Tue
    next_fri = next_weekday(now, 4)  # Fri
    if next_fri <= next_tue:
        next_fri = next_fri + timedelta(days=7)
    delivery_dt = datetime.combine(next_fri.date(), time(5, 0))
    return next_tue, delivery_dt


def simulate_consumption_until_delivery(on_hand_packs: dict, avg_daily_packs: dict, start_dt: datetime, delivery_dt: datetime):
    """Day-by-day consumption using a flat daily average until delivery_dt. Returns projected on-hand at delivery and per-day usage table."""
    day = datetime.combine(start_dt.date(), time(0,0))
    per_day_rows = []
    stock = {p: float(on_hand_packs.get(p, 0)) for p in VALID_PRODUCTS}

    while day < delivery_dt:
        used_today = {}
        for p in VALID_PRODUCTS:
            daily_packs = float(avg_daily_packs.get(p, 0.0))
            stock[p] = round(stock[p] - daily_packs, 4)
            used_today[f"{p} Used (packs)"] = round(daily_packs, 3)
        per_day_rows.append({"Date": pd.Timestamp(day.date()), **used_today, **{f"{p} EoD Stock (packs)": round(stock[p], 3) for p in VALID_PRODUCTS}})
        day += timedelta(days=1)

    proj_at_delivery = {p: round(stock[p], 3) for p in VALID_PRODUCTS}
    per_day_df = pd.DataFrame(per_day_rows)
    return proj_at_delivery, per_day_df


def recommend_po(on_hand_packs_at_delivery: dict, avg_daily_packs: dict, par_days: int) -> dict:
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

# -----------------------------
# Now & Sidebar Config
# -----------------------------
now = datetime.now()

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
    st.subheader("📅 Planning Dates")
    override_dates = st.checkbox("Override auto Tue→Fri", value=True)
    if override_dates:
        order_date_input = st.date_input("Order (Tuesday)", value=datetime(2025, 8, 19).date())
        delivery_date_input = st.date_input("Delivery (Friday)", value=datetime(2025, 8, 22).date())
        delivery_time_input = st.time_input("Delivery time", value=time(5, 0))
        order_dt = datetime.combine(order_date_input, time(12, 0))  # store as midday
        delivery_dt = datetime.combine(delivery_date_input, delivery_time_input)
    else:
        order_dt, delivery_dt = next_order_and_delivery(now)

# -----------------------------
# Weekly Sales History (Manual Entry)
# -----------------------------
st.subheader("Weekly Sales (Sun–Sat) – Manual Entry")
st.caption("Enter total UNITS per week for each product. Average daily packs = (weekly units ÷ 7) ÷ pack size.")

# Load existing history or seed with last full week
if WEEKLY_HISTORY_PATH.exists():
    weekly = pd.read_csv(WEEKLY_HISTORY_PATH)
else:
    prev_start, prev_end = last_full_sun_sat_week(datetime.now())
    weekly = pd.DataFrame([
        {"WeekStart": prev_start.date(), "WeekEnd": prev_end.date(), "Hamburger Units": 0, "Hot Dog Buns Units": 0}
    ])

# Ensure proper dtypes
if not weekly.empty:
    weekly["WeekStart"] = pd.to_datetime(weekly["WeekStart"]).dt.date
    weekly["WeekEnd"] = pd.to_datetime(weekly["WeekEnd"]).dt.date

edited = st.data_editor(
    weekly,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "WeekStart": st.column_config.DateColumn(help="Sunday date"),
        "WeekEnd": st.column_config.DateColumn(help="Saturday date"),
        "Hamburger Units": st.column_config.NumberColumn(min_value=0, step=1),
        "Hot Dog Buns Units": st.column_config.NumberColumn(min_value=0, step=1),
    },
    key="weekly_editor",
)

colA, colB = st.columns(2)
with colA:
    if st.button("💾 Save Weekly History"):
        # Auto-fill WeekEnd from WeekStart if missing
        temp = edited.copy()
        temp["WeekStart"] = pd.to_datetime(temp["WeekStart"]).dt.date
        temp["WeekEnd"] = temp.apply(lambda r: (pd.to_datetime(r["WeekStart"]) + pd.Timedelta(days=6)).date() if pd.isna(r["WeekEnd"]) else r["WeekEnd"], axis=1)
        temp.to_csv(WEEKLY_HISTORY_PATH, index=False)
        st.success("Weekly sales history saved.")
with colB:
    st.download_button("⬇️ Download Weekly Sales History CSV", data=edited.to_csv(index=False), file_name="weekly_sales_history.csv")

if edited.empty:
    st.stop()

# Use last N weeks for projections
hist = edited.dropna(subset=["WeekStart"]).copy()
# Sort by WeekStart
hist = hist.sort_values("WeekStart")
recent = hist.tail(lookback_weeks)

# Compute average daily packs per product: (weekly units ÷ 7) ÷ pack size
avg_daily_packs = {}
for p in VALID_PRODUCTS:
    units_col = "Hamburger Units" if p == "Hamburger" else "Hot Dog Buns Units"
    weekly_mean_units = recent[units_col].astype(float).mean() if not recent.empty else 0.0
    avg_daily_packs[p] = round((weekly_mean_units / 7.0) / PACK_SIZES[p], 4)

# Identify previous (most recent) full week in the table
prev_row = hist.tail(1)
prev_week_label = "—"
prev_summary = None
if not prev_row.empty:
    ws = pd.to_datetime(prev_row.iloc[0]["WeekStart"]).date()
    we = pd.to_datetime(prev_row.iloc[0]["WeekEnd"]).date()
    prev_week_label = f"{ws} to {we}"
    prev_summary = {
        "Hamburger Units": int(prev_row.iloc[0].get("Hamburger Units", 0)),
        "Hot Dog Buns Units": int(prev_row.iloc[0].get("Hot Dog Buns Units", 0)),
    }

# Simulate consumption from NOW until the selected delivery
proj_at_delivery, per_day_usage = simulate_consumption_until_delivery(on_hand_packs, avg_daily_packs, now, delivery_dt)

# Recommended PO to reach PAR after delivery
po_reco = recommend_po(proj_at_delivery, avg_daily_packs, par_days)

# Stockout risk for the Tue/Wed/Thu of the DELIVERY WEEK
week_wd = delivery_dt.weekday()  # Mon=0..Sun=6; Fri=4
week_tue = (delivery_dt + timedelta(days=(1 - week_wd))).date()
week_wed = (delivery_dt + timedelta(days=(2 - week_wd))).date()
week_thu = (delivery_dt + timedelta(days=(3 - week_wd))).date()

risk_rows = []
for p in VALID_PRODUCTS:
    risk = {"Tuesday": False, "Wednesday": False, "Thursday": False}
    for dname, dval in [("Tuesday", week_tue), ("Wednesday", week_wed), ("Thursday", week_thu)]:
        row = per_day_usage[per_day_usage["Date"].dt.date == dval]
        if not row.empty:
            eod = float(row[f"{p} EoD Stock (packs)"].values[0])
            if eod < 0:
                risk[dname] = True
    risk_rows.append({"Product": p, **risk})
risk_df = pd.DataFrame(risk_rows)

# -----------------------------
# Layout
# -----------------------------
st.subheader("Last Recorded Week – Units & Packs")
col1, col2 = st.columns([2,3], gap="large")
with col1:
    st.write(f"**Week:** {prev_week_label}")
    if prev_summary is None:
        st.info("No weekly rows recorded yet.")
    else:
        show_prev = pd.DataFrame([
            {"Product": "Hamburger", "Units": prev_summary["Hamburger Units"], "Packs (ceil)": np.ceil(prev_summary["Hamburger Units"] / PACK_SIZES["Hamburger"])},
            {"Product": "Hot Dog Buns", "Units": prev_summary["Hot Dog Buns Units"], "Packs (ceil)": np.ceil(prev_summary["Hot Dog Buns Units"] / PACK_SIZES["Hot Dog Buns"])},
        ])
        st.dataframe(show_prev, use_container_width=True)

with col2:
    st.write("**Weekly Sales History (Units)**")
    st.dataframe(hist, use_container_width=True)

st.markdown("---")

st.subheader("Projections & Coverage")
col3, col4, col5 = st.columns([1.4, 1.2, 1.4], gap="large")
with col3:
    st.write("**Average Daily Usage (packs)** – from weekly totals: (units ÷ 7) ÷ pack size")
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
    st.write("**Stockout Risk (Tue/Wed/Thu of delivery week)**")
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
    "Notes: Enter weekly Sun–Sat totals in units. Average daily packs = (weekly units ÷ 7) ÷ pack size. "
    "Order is planned for Tuesday → delivery Friday @ 05:00 AM (overrideable above). "
    "PAR target is applied to stock level right after the delivery.")
