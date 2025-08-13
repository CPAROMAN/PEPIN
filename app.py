import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time, date
from pathlib import Path
from io import BytesIO
import importlib.util
import sys, subprocess

# =========================
# App Config
# =========================
st.set_page_config(page_title="Pan Pepín Orders & Inventory (PAR)", layout="wide")
st.title("🥖 Pan Pepín Orders & Inventory Dashboard")
st.caption("Sun–Sat weekly totals • Order Tuesdays → Deliver Fridays @ 5:00 AM • PAR defaults to 10 days")

# =========================
# Constants & Paths
# =========================
# Final products we order: buns (packs)
PACK_SIZES = {"Hamburger": 8, "Hot Dog Buns": 10}
VALID_PRODUCTS = list(PACK_SIZES.keys())
PAR_DAYS_DEFAULT = 7
DEFAULT_HISTORY_WEEKS = 26

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
PO_HISTORY_PATH = DATA_DIR / "po_history.csv"
WEEKLY_HISTORY_PATH = DATA_DIR / "weekly_sales_history.csv"  # still used for manual fallback/download

# ThriveMetrics → buns mapping (1 item = 1 bun)
ITEM_TO_BUN = {
    "shack burger single": "Hamburger",
    "shack double burger": "Hamburger",
    "hot dog quarter pound": "Hot Dog Buns",
}

# =========================
# Helper utilities
# =========================

def lib_info(name: str):
    ok = importlib.util.find_spec(name) is not None
    ver = None
    if ok:
        try: ver = __import__(name).__version__
        except Exception: pass
    return ok, ver

def pip_install(pkgs):
    try:
        cmd = [sys.executable, "-m", "pip", "install", *pkgs]
        res = subprocess.run(cmd, capture_output=True, text=True)
        out = (res.stdout or "") + (res.stderr or "")
        return res.returncode == 0, out
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def last_full_sun_sat_week(today: datetime):
    yday = pd.Timestamp(today.date()) - pd.Timedelta(days=1)
    last_sun = yday - pd.Timedelta(days=(yday.weekday() - 6) % 7)
    return last_sun.normalize(), (last_sun + pd.Timedelta(days=6)).normalize()

def next_weekday(d: datetime, target_weekday: int) -> datetime:
    days_ahead = (target_weekday - d.weekday() + 7) % 7
    return d + timedelta(days=days_ahead or 7)

def next_order_and_delivery(now: datetime):
    next_tue = next_weekday(now, 1); next_fri = next_weekday(now, 4)
    if next_fri <= next_tue: next_fri += timedelta(days=7)
    return next_tue, datetime.combine(next_fri.date(), time(5,0))

# ---- Demand calc (patched: prorated) ----
def simulate_consumption_until_delivery(on_hand_packs, avg_daily_packs, start_dt, delivery_dt):
    rows, stock = [], {p: float(on_hand_packs.get(p, 0.0)) for p in VALID_PRODUCTS}
    cursor = start_dt
    while cursor < delivery_dt:
        next_midnight = datetime.combine((cursor.date() + timedelta(days=1)), time(0,0))
        slice_end = min(next_midnight, delivery_dt)
        frac = (slice_end - cursor).total_seconds() / 86400.0
        disp_date = pd.Timestamp(cursor.date())
        used_today = {f"{p} Used (packs)": float(avg_daily_packs.get(p,0.0))*frac for p in VALID_PRODUCTS}
        for p in VALID_PRODUCTS: stock[p] -= used_today[f"{p} Used (packs)"]
        if rows and rows[-1]["Date"] == disp_date:
            for p in VALID_PRODUCTS:
                rows[-1][f"{p} Used (packs)"] = round(rows[-1][f"{p} Used (packs)"] + used_today[f"{p} Used (packs)"], 3)
                rows[-1][f"{p} EoD Stock (packs)"] = round(stock[p], 3)
        else:
            rows.append({"Date": disp_date,
                         **{k: round(v,3) for k,v in used_today.items()},
                         **{f"{p} EoD Stock (packs)": round(stock[p],3) for p in VALID_PRODUCTS}})
        cursor = slice_end
    return {p: round(stock[p],3) for p in VALID_PRODUCTS}, pd.DataFrame(rows)

def recommend_po(on_hand_now_packs, avg_daily_packs, par_days, start_dt, delivery_dt):
    days_until_delivery = max(0.0, (delivery_dt - start_dt).total_seconds()/86400.0)
    calc_need, reco = {}, {}
    for p in VALID_PRODUCTS:
        avg = float(avg_daily_packs.get(p,0.0)); cur = float(on_hand_now_packs.get(p,0.0))
        need = round((par_days*avg) + (days_until_delivery*avg) - cur, 3)
        calc_need[p] = need; reco[p] = int(np.ceil(max(0.0, need)))
    return calc_need, reco

# ---- ThriveMetrics CSV parsing (ROBUST) ----
def parse_thrivemetrics_csv(file) -> pd.DataFrame:
    """Return a normalized dataframe with at least: date, item, qty."""
    try:
        file.seek(0)
    except Exception:
        pass

    try:
        raw = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig",
                          skipinitialspace=True, on_bad_lines="skip")
    except Exception:
        try:
            file.seek(0)
        except Exception:
            pass
        raw = pd.read_csv(file, delimiter=",", engine="python", encoding="utf-8-sig",
                          skipinitialspace=True, on_bad_lines="skip")

    if len(raw.columns) == 1 and raw.shape[0] > 0:
        s = raw.columns[0]
        if s and isinstance(s, str) and ("," in s or ";" in s or "\t" in s or "|" in s):
            for delim in [",", ";", "\t", "|"]:
                try:
                    file.seek(0)
                except Exception:
                    pass
                raw = pd.read_csv(file, delimiter=delim, engine="python", encoding="utf-8-sig",
                                  skipinitialspace=True, on_bad_lines="skip")
                if len(raw.columns) > 1:
                    break

    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    def pick_column(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    item_col = pick_column(["item","menu item","product","name","description","modifier name"])
    qty_col  = pick_column(["qty","quantity","units","sold","count","modifier sold"])
    date_col = pick_column(["date","business date","order date","created","created at"])

    if not all([item_col, qty_col, date_col]):
        raise ValueError("CSV must include columns for item, quantity, and date.")

    out = df[[date_col, item_col, qty_col]].copy()
    out.columns = ["date", "item", "qty"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["item"] = out["item"].astype(str).str.strip().str.lower()
    out["qty"]  = pd.to_numeric(out["qty"], errors="coerce").fillna(0)
    return out.dropna(subset=["date"])

def buns_weekly_from_thrives(df_items: pd.DataFrame) -> pd.DataFrame:
    df = df_items.copy()
    df["bun_product"] = df["item"].map(ITEM_TO_BUN)
    df = df.dropna(subset=["bun_product"])
    df["bun_units"] = df["qty"].astype(float)

    dt_series = pd.to_datetime(df["date"]).dt.date
    weekday = pd.to_datetime(df["date"]).dt.weekday
    days_to_sun = (weekday + 1) % 7
    week_start = pd.to_datetime(dt_series) - pd.to_timedelta(days_to_sun, unit="D")
    week_end = week_start + pd.to_timedelta(6, unit="D")

    df["WeekStart"] = week_start.dt.date
    df["WeekEnd"] = week_end.dt.date

    piv = (df.pivot_table(index=["WeekStart","WeekEnd"],
                          columns="bun_product", values="bun_units",
                          aggfunc="sum", fill_value=0)
             .reset_index())
    if "Hamburger" not in piv.columns: piv["Hamburger"] = 0.0
    if "Hot Dog Buns" not in piv.columns: piv["Hot Dog Buns"] = 0.0

    weekly = piv.rename(columns={"Hamburger":"Hamburger Units","Hot Dog Buns":"Hot Dog Buns Units"})
    return weekly[["WeekStart","WeekEnd","Hamburger Units","Hot Dog Buns Units"]].sort_values("WeekStart")

# =========================
# Sidebar Controls
# =========================
now = datetime.now()
with st.sidebar:
    st.header("⚙️ Settings")
    par_days = st.number_input("PAR level (days)", 1, 30, PAR_DAYS_DEFAULT)
    lookback_weeks = st.slider("Weeks of history for projections", 2, 26, DEFAULT_HISTORY_WEEKS)
    st.markdown("---")
    st.subheader("📦 Current On-Hand (packs)")
    on_hand_packs = {p: st.number_input(f"{p} (packs)", min_value=0.0, step=0.001, value=0.0, format="%.3f") for p in VALID_PRODUCTS}
    st.markdown("---")
    st.subheader("📅 Planning Dates")
    override_dates = st.checkbox("Override auto Tue→Fri", value=True)
    if override_dates:
        order_date_input = st.date_input("Order (Tuesday)", value=now.date())
        delivery_date_input = st.date_input("Delivery (Friday)", value=(now + timedelta(days=3)).date())
        delivery_time_input = st.time_input("Delivery time", value=time(5,0))
        order_dt = datetime.combine(order_date_input, time(12,0))
        delivery_dt = datetime.combine(delivery_date_input, delivery_time_input)
    else:
        order_dt, delivery_dt = next_order_and_delivery(now)
    st.markdown("---")
    st.subheader("🧪 Environment Check")
    libs = ["xlsxwriter","openpyxl","reportlab"]
    missing=[]
    for lib in libs:
        ok, ver = lib_info(lib)
        st.write(("✅" if ok else "❌") + f" **{lib}**" + (f" v{ver}" if (ok and ver) else ""))
        if not ok: missing.append(lib)
    if missing:
        st.caption("Install missing: " + " ".join([f"`pip install {m}`" for m in missing]))

# =========================
# Data Source
# =========================
st.subheader("Sales Data Source")
mode = st.radio("Choose input method", ["Upload ThriveMetrics CSV (recommended)", "Manual weekly entry"], index=0)

if mode.startswith("Upload"):
    up = st.file_uploader("Upload CSV exported from ThriveMetrics", type=["csv"])
    if up is None:
        st.info("Upload a CSV to build weekly Sun–Sat bun units automatically from items.")
        st.stop()
    try:
        up.seek(0)
        items_df = parse_thrivemetrics_csv(up)
        weekly = buns_weekly_from_thrives(items_df)
        weekly.to_csv(WEEKLY_HISTORY_PATH, index=False)
        st.success("CSV parsed. Weekly bun unit totals generated.")
        st.dataframe(weekly, use_container_width=True)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        st.stop()
else:
    if WEEKLY_HISTORY_PATH.exists():
        weekly = pd.read_csv(WEEKLY_HISTORY_PATH)
    else:
        ps, pe = last_full_sun_sat_week(now)
        weekly = pd.DataFrame([{ "WeekStart": ps.date(), "WeekEnd": pe.date(), "Hamburger Units": 0, "Hot Dog Buns Units": 0 }])
    if not weekly.empty:
        weekly["WeekStart"] = pd.to_datetime(weekly["WeekStart"]).dt.date
        weekly["WeekEnd"] = pd.to_datetime(weekly["WeekEnd"]).dt.date
    edited = st.data_editor(
        weekly, num_rows="dynamic", use_container_width=True,
        column_config={
            "WeekStart": st.column_config.DateColumn(help="Sunday"),
            "WeekEnd": st.column_config.DateColumn(help="Saturday"),
            "Hamburger Units": st.column_config.NumberColumn(min_value=0, step=1),
            "Hot Dog Buns Units": st.column_config.NumberColumn(min_value=0, step=1),
        }, key="weekly_editor",
    )
    if st.button("💾 Save Weekly History"):
        temp = edited.copy()
        temp["WeekStart"] = pd.to_datetime(temp["WeekStart"]).dt.date
        temp["WeekEnd"] = temp.apply(lambda r: (pd.to_datetime(r["WeekStart"]) + pd.Timedelta(days=6)).date()
                                     if pd.isna(r["WeekEnd"]) else r["WeekEnd"], axis=1)
        temp.to_csv(WEEKLY_HISTORY_PATH, index=False)
        st.success("Saved.")
    st.download_button("⬇️ Download Weekly Sales History CSV", data=edited.to_csv(index=False),
                       file_name="weekly_sales_history.csv")
    weekly = edited
    if weekly.empty:
        st.stop()

# =========================
# Build averages
# =========================
hist = weekly.dropna(subset=["WeekStart"]).copy().sort_values("WeekStart")
recent = hist.tail(lookback_weeks)

avg_daily_packs = {}
for p in VALID_PRODUCTS:
    units_col = "Hamburger Units" if p == "Hamburger" else "Hot Dog Buns Units"
    weekly_mean_units = recent[units_col].astype(float).mean() if not recent.empty else 0.0
    avg_daily_packs[p] = round((weekly_mean_units / PACK_SIZES[p]) / 7.0, 3)

prev_row = hist.tail(1)
prev_label = "—"; prev_summary=None
if not prev_row.empty:
    ws = pd.to_datetime(prev_row.iloc[0]["WeekStart"]).date()
    we = pd.to_datetime(prev_row.iloc[0]["WeekEnd"]).date()
    prev_label = f"{ws} to {we}"
    prev_summary = {
        "Hamburger Units": int(prev_row.iloc[0].get("Hamburger Units",0)),
        "Hot Dog Buns Units": int(prev_row.iloc[0].get("Hot Dog Buns Units",0)),
    }

# =========================
# Simulation & PO
# =========================
proj_at_delivery, per_day_usage = simulate_consumption_until_delivery(on_hand_packs, avg_daily_packs, now, delivery_dt)
po_calc_need, po_reco = recommend_po(on_hand_packs, avg_daily_packs, par_days, now, delivery_dt)

# Risk Tue/Wed/Thu
ww = delivery_dt.weekday()
week_tue = (delivery_dt + timedelta(days=(1-ww))).date()
week_wed = (delivery_dt + timedelta(days=(2-ww))).date()
week_thu = (delivery_dt + timedelta(days=(3-ww))).date()
risk_rows = []
for p in VALID_PRODUCTS:
    risk = {d: False for d in ["Tuesday","Wednesday","Thursday"]}
    for dname, dval in [("Tuesday", week_tue),("Wednesday", week_wed),("Thursday", week_thu)]:
        row = per_day_usage[per_day_usage["Date"].dt.date == dval]
        if not row.empty and float(row[f"{p} EoD Stock (packs)"].values[0]) < 0:
            risk[dname] = True
    risk_rows.append({"Product": p, **risk})
risk_df = pd.DataFrame(risk_rows)

# =========================
# Layout
# =========================
st.subheader("Last Recorded Week – Units & Packs")
if prev_summary is not None:
    show_prev = pd.DataFrame([
        {"Product": "Hamburger", "Units": prev_summary["Hamburger Units"],
         "Packs (ceil)": np.ceil(prev_summary["Hamburger Units"]/PACK_SIZES["Hamburger"])},
        {"Product": "Hot Dog Buns", "Units": prev_summary["Hot Dog Buns Units"],
         "Packs (ceil)": np.ceil(prev_summary["Hot Dog Buns Units"]/PACK_SIZES["Hot Dog Buns"])},
    ])
    st.dataframe(show_prev, use_container_width=True)
st.dataframe(hist, use_container_width=True)

st.subheader("Projections & Coverage")
avg_tbl = pd.DataFrame({"Product": VALID_PRODUCTS, "Avg Daily Packs": [avg_daily_packs[p] for p in VALID_PRODUCTS],
                        "Pack Size": [PACK_SIZES[p] for p in VALID_PRODUCTS]})
avg_tbl["Avg Daily Packs"] = avg_tbl["Avg Daily Packs"].round(3)
st.dataframe(avg_tbl, use_container_width=True)

dos_rows = []
for p in VALID_PRODUCTS:
    avg_use = max(1e-6, avg_daily_packs[p])
    dos_rows.append({"Product": p, "On-Hand (packs)": round(on_hand_packs[p],3),
                     "Days Remaining": round(on_hand_packs[p]/avg_use,2)})
st.dataframe(pd.DataFrame(dos_rows), use_container_width=True)

st.dataframe(risk_df, use_container_width=True)

st.subheader("Next Order Recommendation")
po_rows = []
for p in VALID_PRODUCTS:
    packs_reco = po_reco[p]
    po_rows.append({
        "Product": p,
        "Projected On-Hand at Delivery (packs)": round(proj_at_delivery[p],3),
        "Avg Daily (packs)": round(avg_daily_packs[p],3),
        "Target Stock (packs)": round(par_days*avg_daily_packs[p],3),
        "Calc Need (packs)": po_calc_need[p],
        "Recommended PO (packs)": packs_reco,
        "Pack Size (units/pack)": PACK_SIZES[p],
        "Recommended PO (units)": int(packs_reco*PACK_SIZES[p]),
    })
po_df = pd.DataFrame(po_rows)
st.dataframe(po_df, use_container_width=True)

def build_po_excel(po_df, order_dt, delivery
