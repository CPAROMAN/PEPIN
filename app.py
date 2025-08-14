import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time, date
from pathlib import Path
from io import BytesIO
import importlib.util

# =========================
# App Config
# =========================
st.set_page_config(page_title="Pan Pepín Orders & Inventory (PAR)", layout="wide")
st.title("🥖 Pan Pepín Orders & Inventory Dashboard")
st.caption("Sun–Sat weekly totals • Order Tuesdays → Deliver Fridays @ 5:00 AM • PAR defaults to 8 days")

# =========================
# Constants & Paths
# =========================
PACK_SIZES = {"Hamburger": 8, "Hot Dog Buns": 10}   # packs → units per pack
VALID_PRODUCTS = list(PACK_SIZES.keys())
PAR_DAYS_DEFAULT = 8
DEFAULT_HISTORY_WEEKS = 26

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
PO_HISTORY_PATH = DATA_DIR / "po_history.csv"
WEEKLY_HISTORY_PATH = DATA_DIR / "weekly_sales_history.csv"

# ThriveMetrics → buns mapping (1 item = 1 bun unit)
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
        try:
            ver = __import__(name).__version__
        except Exception:
            pass
    return ok, ver

def last_full_sun_sat_week(today: datetime):
    yday = pd.Timestamp(today.date()) - pd.Timedelta(days=1)
    last_sun = yday - pd.Timedelta(days=(yday.weekday() - 6) % 7)
    return last_sun.normalize(), (last_sun + pd.Timedelta(days=6)).normalize()

def next_weekday(d: datetime, target_weekday: int) -> datetime:
    days_ahead = (target_weekday - d.weekday() + 7) % 7
    return d + timedelta(days=days_ahead or 7)

def next_order_and_delivery(now: datetime):
    next_tue = next_weekday(now, 1)  # Tuesday
    next_fri = next_weekday(now, 4)  # Friday
    if next_fri <= next_tue:
        next_fri += timedelta(days=7)
    return next_tue, datetime.combine(next_fri.date(), time(5, 0))

# =========================
# Core calcs
# =========================
def simulate_consumption_until_delivery(on_hand_packs, avg_daily_packs, start_dt, delivery_dt):
    """
    Prorates daily usage between now and the delivery moment.
    Returns: (projected_on_hand_at_delivery: dict, per_day_usage_df: DataFrame)
    """
    rows, stock = [], {p: float(on_hand_packs.get(p, 0.0)) for p in VALID_PRODUCTS}
    cursor = start_dt
    while cursor < delivery_dt:
        next_midnight = datetime.combine((cursor.date() + timedelta(days=1)), time(0, 0))
        slice_end = min(next_midnight, delivery_dt)
        frac = (slice_end - cursor).total_seconds() / 86400.0
        disp_date = pd.Timestamp(cursor.date())
        used_today = {f"{p} Used (packs)": float(avg_daily_packs.get(p, 0.0)) * frac for p in VALID_PRODUCTS}
        for p in VALID_PRODUCTS:
            stock[p] -= used_today[f"{p} Used (packs)"]
        if rows and rows[-1]["Date"] == disp_date:
            for p in VALID_PRODUCTS:
                rows[-1][f"{p} Used (packs)"] = round(rows[-1][f"{p} Used (packs)"] + used_today[f"{p} Used (packs)"], 3)
                rows[-1][f"{p} EoD Stock (packs)"] = round(stock[p], 3)
        else:
            rows.append({
                "Date": disp_date,
                **{k: round(v, 3) for k, v in used_today.items()},
                **{f"{p} EoD Stock (packs)": round(stock[p], 3) for p in VALID_PRODUCTS}
            })
        cursor = slice_end
    return {p: round(stock[p], 3) for p in VALID_PRODUCTS}, pd.DataFrame(rows)

def recommend_po(on_hand_now_packs, avg_daily_packs, par_days, start_dt, delivery_dt):
    days_until_delivery = max(0.0, (delivery_dt - start_dt).total_seconds() / 86400.0)
    calc_need, reco = {}, {}
    for p in VALID_PRODUCTS:
        avg = float(avg_daily_packs.get(p, 0.0))
        cur = float(on_hand_now_packs.get(p, 0.0))
        need = round((par_days * avg) + (days_until_delivery * avg) - cur, 3)
        calc_need[p] = need
        reco[p] = int(np.ceil(max(0.0, need)))
    return calc_need, reco

# =========================
# CSV parsing (robust + manual mapping fallback)
# =========================
DATE_SYNS = {"date","business date","order date","created","created at","sale date","date/time","timestamp"}
ITEM_SYNS = {"item","item name","menu item","product","product name","name","description","modifier name","modifier","pos item","sku","plu"}
QTY_SYNS  = {"qty","quantity","qty sold","sold qty","units","units sold","items sold","count","sold","modifier sold","# sold","quantity sold"}

def _read_csv_any(file):
    # Try sniffing; fallback to comma; then try common delimiters
    def _read(delim=None, header="infer"):
        kw = dict(engine="python", encoding="utf-8-sig", skipinitialspace=True, on_bad_lines="skip")
        if header is not None: kw["header"] = header
        if delim is None:
            return pd.read_csv(file, sep=None, **kw)
        else:
            try: file.seek(0)
            except: pass
            return pd.read_csv(file, delimiter=delim, **kw)
    try:
        file.seek(0)
    except Exception:
        pass
    try:
        raw = _read()
    except Exception:
        raw = _read(",")
    if len(raw.columns) == 1 and raw.shape[0] > 0:
        for d in [",", ";", "\t", "|"]:
            try:
                alt = _read(d)
                if len(alt.columns) > 1:
                    raw = alt
                    break
            except Exception:
                continue
    # Try to auto-detect real header row if Unnamed
    cols_lower = [str(c).strip().lower() for c in raw.columns]
    looks_like_bad_header = (len(set(cols_lower)) == 1 and ("unnamed" in cols_lower[0] or cols_lower[0] == "0"))
    if looks_like_bad_header or any(str(c).startswith("Unnamed") for c in raw.columns):
        try:
            file.seek(0)
            preview = pd.read_csv(file, header=None, engine="python", encoding="utf-8-sig",
                                  on_bad_lines="skip", nrows=20)
            best_i, best_score = 0, -1
            for i in range(min(20, len(preview))):
                row = preview.iloc[i].astype(str).str.strip()
                score = (row != "").sum()
                keywords = ["date","business","order","created","item","name","product","qty","quantity","sold","modifier"]
                score += 2 * sum(any(k in s.lower() for k in keywords) for s in row)
                if score > best_score:
                    best_i, best_score = i, score
            file.seek(0)
            raw = pd.read_csv(file, header=best_i, engine="python", encoding="utf-8-sig", on_bad_lines="skip")
        except Exception:
            pass
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    return raw

def _pick_col(df, syns):
    for c in df.columns:
        if c in syns: return c
    for c in df.columns:  # substring contains
        for s in syns:
            if s in c: return c
    return None

def _auto_detect_cols(df):
    # detect date
    scores = {}
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            scores[c] = parsed.notna().mean()
        except Exception:
            scores[c] = -1
    date_col = max(scores, key=scores.get) if scores else None
    if date_col and scores[date_col] < 0.4: date_col = None

    # detect qty
    scores = {}
    for c in df.columns:
        if c == date_col: scores[c] = -1; continue
        nums = pd.to_numeric(df[c], errors="coerce")
        scores[c] = nums.notna().mean() * 0.7 + (nums.fillna(0) >= 0).mean() * 0.3
    qty_col = max(scores, key=scores.get) if scores else None
    if qty_col and scores[qty_col] < 0.4: qty_col = None

    # detect item
    known = set(ITEM_TO_BUN.keys())
    def textiness(s): return (s.astype(str).str.strip() != "").mean()
    def matchscore(s): return s.astype(str).str.strip().str.lower().isin(known).mean()
    scores = {}
    for c in df.columns:
        if c in {date_col, qty_col}: scores[c] = -1; continue
        try:
            scores[c] = 0.6*matchscore(df[c]) + 0.4*textiness(df[c])
        except Exception:
            scores[c] = -1
    item_col = max(scores, key=scores.get) if scores else None
    if item_col and scores[item_col] < 0.2:
        scores = {k: (textiness(df[k]) if k not in {date_col, qty_col} else -1) for k in df.columns}
        item_col = max(scores, key=scores.get) if scores else None
        if item_col and scores[item_col] < 0.2: item_col = None

    return date_col, item_col, qty_col

def parse_thrivemetrics_csv_or_manual(file) -> pd.DataFrame:
    """Return normalized df with date, item, qty. If auto fails, show mapping UI and return mapped df."""
    df = _read_csv_any(file)
    if df.empty:
        raise ValueError("CSV appears to be empty.")
    # Try synonyms first
    date_col = _pick_col(df, DATE_SYNS)
    item_col = _pick_col(df, ITEM_SYNS)
    qty_col  = _pick_col(df, QTY_SYNS)
    # Heuristic fallback
    if not all([date_col, item_col, qty_col]):
        d2, i2, q2 = _auto_detect_cols(df)
        date_col = date_col or d2
        item_col = item_col or i2
        qty_col  = qty_col  or q2

    # If still missing, render mapping UI
    if not all([date_col, item_col, qty_col]):
        st.warning("Could not automatically detect the Date / Item / Quantity columns. Please map them below:")
        st.dataframe(df.head(20), use_container_width=True)
        cols = list(df.columns)
        c1,c2,c3,c4 = st.columns([1,1,1,1])
        with c1: date_col = st.selectbox("Date column", options=cols, index=0 if cols else None)
        with c2: item_col = st.selectbox("Item column", options=cols, index=1 if len(cols)>1 else 0)
        with c3: qty_col  = st.selectbox("Quantity column", options=cols, index=2 if len(cols)>2 else 0)
        with c4: date_fmt = st.text_input("Date format (optional, e.g. %m/%d/%Y)", value="")
        if not all([date_col, item_col, qty_col]):
            raise ValueError("Date / Item / Quantity must be selected.")
        # Parse with optional format
        if date_fmt.strip():
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce", format=date_fmt.strip()).dt.date
        else:
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce").dt.date
        items = df[item_col].astype(str).str.strip().str.lower()
        qty = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
        out = pd.DataFrame({"date": parsed_dates, "item": items, "qty": qty}).dropna(subset=["date"])
        st.success("Column mapping applied.")
        return out

    # Standard path
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.date
    items = df[item_col].astype(str).str.strip().str.lower()
    qty   = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    return pd.DataFrame({"date": dates, "item": items, "qty": qty}).dropna(subset=["date"])

def buns_weekly_from_thrives(df_items: pd.DataFrame) -> pd.DataFrame:
    """
    Map item rows to bun units and aggregate Sun–Sat per week.
    """
    df = df_items.copy()
    df["bun_product"] = df["item"].map(ITEM_TO_BUN)
    df = df.dropna(subset=["bun_product"])

    # 1 row = bun_units (1:1 mapping)
    df["bun_units"] = df["qty"].astype(float)

    # Week start/end (Sunday → Saturday)
    dt_series = pd.to_datetime(df["date"]).dt.date
    weekday = pd.to_datetime(df["date"]).dt.weekday  # Mon=0..Sun=6
    days_to_sun = (weekday + 1) % 7
    week_start = pd.to_datetime(dt_series) - pd.to_timedelta(days_to_sun, unit="D")
    week_end = week_start + pd.to_timedelta(6, unit="D")

    df["WeekStart"] = week_start.dt.date
    df["WeekEnd"] = week_end.dt.date

    piv = (
        df.pivot_table(
            index=["WeekStart", "WeekEnd"],
            columns="bun_product",
            values="bun_units",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    if "Hamburger" not in piv.columns:
        piv["Hamburger"] = 0.0
    if "Hot Dog Buns" not in piv.columns:
        piv["Hot Dog Buns"] = 0.0

    weekly = piv.rename(
        columns={"Hamburger": "Hamburger Units", "Hot Dog Buns": "Hot Dog Buns Units"}
    )
    weekly = weekly[["WeekStart", "WeekEnd", "Hamburger Units", "Hot Dog Buns Units"]].sort_values("WeekStart")
    return weekly

# =========================
# Sidebar Controls
# =========================
now = datetime.now()
with st.sidebar:
    st.header("⚙️ Settings")
    par_days = st.number_input("PAR level (days)", 1, 30, PAR_DAYS_DEFAULT)
    lookback_weeks = st.slider("Weeks of history for projections", 2, 26, DEFAULT_HISTORY_WEEKS)
    st.markdown("---")
    st.subheader("📦 Current On‑Hand (packs)")
    on_hand_packs = {
        p: st.number_input(f"{p} (packs)", min_value=0.0, step=0.001, value=0.0, format="%.3f")
        for p in VALID_PRODUCTS
    }
    st.markdown("---")
    st.subheader("📅 Planning Dates")
    override_dates = st.checkbox("Override auto Tue→Fri", value=True)
    if override_dates:
        order_date_input = st.date_input("Order (Tuesday)", value=now.date())
        delivery_date_input = st.date_input("Delivery (Friday)", value=(now + timedelta(days=3)).date())
        delivery_time_input = st.time_input("Delivery time", value=time(5, 0))
        order_dt = datetime.combine(order_date_input, time(12, 0))
        delivery_dt = datetime.combine(delivery_date_input, delivery_time_input)
    else:
        order_dt, delivery_dt = next_order_and_delivery(now)
    st.markdown("---")
    st.subheader("🧪 Environment Check")
    libs = ["xlsxwriter", "openpyxl", "reportlab"]
    missing = []
    for lib in libs:
        ok, ver = lib_info(lib)
        st.write(("✅" if ok else "❌") + f" **{lib}**" + (f" v{ver}" if (ok and ver) else ""))
        if not ok:
            missing.append(lib)
    if missing:
        st.caption("Install missing: " + " ".join([f"`pip install {m}`" for m in missing]))

# =========================
# Data Source: ThriveMetrics CSV or Manual
# =========================
st.subheader("Sales Data Source")
mode = st.radio("Choose input method", ["Upload ThriveMetrics CSV (recommended)", "Manual weekly entry"], index=0)

if mode.startswith("Upload"):
    up = st.file_uploader("Upload CSV exported from ThriveMetrics", type=["csv"])
    if up is None:
        st.info("Upload a CSV to build weekly Sun–Sat bun units automatically from items: Shack Burger Single, Shack Double Burger, Hot Dog Quarter Pound.")
        st.stop()
    try:
        try:
            up.seek(0)
        except Exception:
            pass
        items_df = parse_thrivemetrics_csv_or_manual(up)
        # debug/visibility: show what we detected
        st.caption("Detected/selected normalized columns (first 10 rows):")
        st.dataframe(items_df.head(10), use_container_width=True)

        weekly = buns_weekly_from_thrives(items_df)
        weekly.to_csv(WEEKLY_HISTORY_PATH, index=False)
        st.success("CSV parsed. Weekly bun unit totals generated.")
        st.dataframe(weekly, use_container_width=True)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        st.stop()
else:
    st.caption("Enter total UNITS per week for each bun product. Average daily packs = (weekly units ÷ pack size) ÷ 7.")
    if WEEKLY_HISTORY_PATH.exists():
        weekly = pd.read_csv(WEEKLY_HISTORY_PATH)
    else:
        ps, pe = last_full_sun_sat_week(now)
        weekly = pd.DataFrame([{
            "WeekStart": ps.date(),
            "WeekEnd": pe.date(),
            "Hamburger Units": 0,
            "Hot Dog Buns Units": 0
        }])
    if not weekly.empty:
        weekly["WeekStart"] = pd.to_datetime(weekly["WeekStart"]).dt.date
        weekly["WeekEnd"] = pd.to_datetime(weekly["WeekEnd"]).dt.date

    edited = st.data_editor(
        weekly,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "WeekStart": st.column_config.DateColumn(help="Sunday"),
            "WeekEnd": st.column_config.DateColumn(help="Saturday"),
            "Hamburger Units": st.column_config.NumberColumn(min_value=0, step=1),
            "Hot Dog Buns Units": st.column_config.NumberColumn(min_value=0, step=1),
        },
        key="weekly_editor",
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("💾 Save Weekly History"):
            temp = edited.copy()
            temp["WeekStart"] = pd.to_datetime(temp["WeekStart"]).dt.date
            temp["WeekEnd"] = temp.apply(
                lambda r: (pd.to_datetime(r["WeekStart"]) + pd.Timedelta(days=6)).date()
                if pd.isna(r["WeekEnd"]) else r["WeekEnd"], axis=1
            )
            temp.to_csv(WEEKLY_HISTORY_PATH, index=False)
            st.success("Saved.")
    with colB:
        st.download_button("⬇️ Download Weekly Sales History CSV",
                           data=edited.to_csv(index=False),
                           file_name="weekly_sales_history.csv")
    weekly = edited
    if weekly.empty:
        st.stop()

# =========================
# Build averages from last N weeks
# =========================
hist = weekly.dropna(subset=["WeekStart"]).copy().sort_values("WeekStart")
recent = hist.tail(lookback_weeks)

avg_daily_packs = {}
for p in VALID_PRODUCTS:
    units_col = "Hamburger Units" if p == "Hamburger" else "Hot Dog Buns Units"
    weekly_mean_units = recent[units_col].astype(float).mean() if not recent.empty else 0.0
    avg_daily_packs[p] = round((weekly_mean_units / PACK_SIZES[p]) / 7.0, 3)

# Last week summary
prev_row = hist.tail(1)
prev_label = "—"; prev_summary = None
if not prev_row.empty:
    ws = pd.to_datetime(prev_row.iloc[0]["WeekStart"]).date()
    we = pd.to_datetime(prev_row.iloc[0]["WeekEnd"]).date()
    prev_label = f"{ws} to {we}"
    prev_summary = {
        "Hamburger Units": int(prev_row.iloc[0].get("Hamburger Units", 0)),
        "Hot Dog Buns Units": int(prev_row.iloc[0].get("Hot Dog Buns Units", 0)),
    }

# =========================
# Simulation & PO
# =========================
# (order_dt / delivery_dt set earlier in sidebar)
proj_at_delivery, per_day_usage = simulate_consumption_until_delivery(
    on_hand_packs, avg_daily_packs, datetime.now(), delivery_dt
)
po_calc_need, po_reco = recommend_po(
    on_hand_packs, avg_daily_packs, par_days, datetime.now(), delivery_dt
)

# Risk Tue/Wed/Thu of delivery week
ww = delivery_dt.weekday()
week_tue = (delivery_dt + timedelta(days=(1 - ww))).date()
week_wed = (delivery_dt + timedelta(days=(2 - ww))).date()
week_thu = (delivery_dt + timedelta(days=(3 - ww))).date()
risk_rows = []
for p in VALID_PRODUCTS:
    risk = {d: False for d in ["Tuesday", "Wednesday", "Thursday"]}
    for dname, dval in [("Tuesday", week_tue), ("Wednesday", week_wed), ("Thursday", week_thu)]:
        row = per_day_usage[per_day_usage["Date"] .dt.date == dval]
        if not row.empty and float(row[f"{p} EoD Stock (packs)"].values[0]) < 0:
            risk[dname] = True
    risk_rows.append({"Product": p, **risk})
risk_df = pd.DataFrame(risk_rows)

# =========================
# Layout
# =========================
st.subheader("Last Recorded Week – Units & Packs")
col1, col2 = st.columns([2, 3], gap="large")
with col1:
    st.write(f"**Week:** {prev_label}")
    if prev_summary is None:
        st.info("No weekly rows available yet.")
    else:
        show_prev = pd.DataFrame([
            {"Product": "Hamburger", "Units": prev_summary["Hamburger Units"],
             "Packs (ceil)": np.ceil(prev_summary["Hamburger Units"] / PACK_SIZES["Hamburger"])},
            {"Product": "Hot Dog Buns", "Units": prev_summary["Hot Dog Buns Units"],
             "Packs (ceil)": np.ceil(prev_summary["Hot Dog Buns Units"] / PACK_SIZES["Hot Dog Buns"])},
        ])
        st.dataframe(show_prev, use_container_width=True)
with col2:
    st.write("**Weekly Sales History (Units)**")
    st.dataframe(hist, use_container_width=True)

st.markdown("---")

st.subheader("Projections & Coverage")
col3, col4, col5 = st.columns([1.4, 1.2, 1.4], gap="large")
with col3:
    st.write("**Average Daily Usage (packs)** – (weekly units ÷ pack size) ÷ 7")
    avg_tbl = pd.DataFrame({
        "Product": VALID_PRODUCTS,
        "Avg Daily Packs": [avg_daily_packs[p] for p in VALID_PRODUCTS],
        "Pack Size": [PACK_SIZES[p] for p in VALID_PRODUCTS],
    })
    avg_tbl["Avg Daily Packs"] = avg_tbl["Avg Daily Packs"].astype(float).round(3)
    st.dataframe(avg_tbl, use_container_width=True)
with col4:
    st.write("**Days of Supply On‑Hand**")
    dos_rows = []
    for p in VALID_PRODUCTS:
        avg_use = max(1e-6, float(avg_daily_packs.get(p, 0.0)))
        dos_rows.append({
            "Product": p,
            "On‑Hand (packs)": round(float(on_hand_packs.get(p, 0.0)), 3),
            "Days Remaining": round(float(on_hand_packs.get(p, 0.0)) / avg_use, 2),
        })
    st.dataframe(pd.DataFrame(dos_rows), use_container_width=True)
with col5:
    st.write("**Stockout Risk (Tue/Wed/Thu of delivery week)**")
    st.dataframe(risk_df, use_container_width=True)

st.markdown("---")

st.subheader("Next Order Recommendation (to reach PAR after Fri 5:00 AM delivery)")
st.write(f"**Order Date:** {order_dt.date()}  •  **Delivery:** {delivery_dt}  •  **PAR:** {par_days} days")
po_rows = []
for p in VALID_PRODUCTS:
    packs_reco = po_reco[p]
    po_rows.append({
        "Product": p,
        "Projected On‑Hand at Delivery (packs)": round(proj_at_delivery[p], 3),
        "Avg Daily (packs)": round(avg_daily_packs[p], 3),
        "Target Stock (packs)": round(par_days * avg_daily_packs[p], 3),
        "Calc Need (packs)": po_calc_need[p],
        "Recommended PO (packs)": packs_reco,
        "Pack Size (units/pack)": PACK_SIZES[p],
        "Recommended PO (units)": int(packs_reco * PACK_SIZES[p]),
    })
po_df = pd.DataFrame(po_rows)
st.dataframe(po_df, use_container_width=True)

# =========================
# Exports
# =========================
def build_po_excel(po_df: pd.DataFrame, order_dt: datetime, delivery_dt: datetime, par_days: int):
    engine = None
    try:
        import xlsxwriter; engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl; engine = "openpyxl"
        except Exception:
            return None, "Excel export requires xlsxwriter or openpyxl."
    bio = BytesIO(); df = po_df.copy()
    for c in [c for c in df.columns if "(packs)" in c and "Recommended" not in c]:
        df[c] = df[c].astype(float).round(3)
    with pd.ExcelWriter(bio, engine=engine) as w:
        pd.DataFrame({"Field":["Order Date","Delivery","PAR Days"],"Value":[order_dt.date(),delivery_dt,par_days]}).to_excel(w, index=False, sheet_name="Summary")
        df.to_excel(w, index=False, sheet_name="PO")
    return bio.getvalue(), None

def build_po_pdf(po_df: pd.DataFrame, order_dt: datetime, delivery_dt: datetime, par_days: int) -> bytes:
    try:
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as e:
        raise ImportError("reportlab not installed") from e
    bio = BytesIO(); doc = SimpleDocTemplate(bio, pagesize=LETTER, title="Pan Pepín – Purchase Order")
    styles = getSampleStyleSheet(); elems=[]
    elems += [Paragraph("Pan Pepín – Purchase Order", styles["Title"]),
              Paragraph(f"Order Date: {order_dt.date()}  •  Delivery: {delivery_dt}  •  PAR: {par_days} days", styles["Normal"]),
              Spacer(1,12)]
    headers = list(po_df.columns); data=[headers]
    for _,r in po_df.iterrows():
        row=[]
        for c in headers:
            v=r[c]; row.append(f"{v:.3f}" if isinstance(v,(float,np.floating)) else str(v))
        data.append(row)
    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    elems.append(t)
    doc.build(elems)
    return bio.getvalue()

colX, colY = st.columns(2)
with colX:
    excel_bytes, excel_err = build_po_excel(po_df, order_dt, delivery_dt, par_days)
    if excel_bytes:
        st.download_button("⬇️ Export PO to Excel (.xlsx)", data=excel_bytes, file_name=f"PO_{order_dt.date()}_{delivery_dt.date()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info(excel_err)
with colY:
    try:
        pdf_bytes = build_po_pdf(po_df, order_dt, delivery_dt, par_days)
        st.download_button("🖨️ Print PO (PDF)", data=pdf_bytes, file_name=f"PO_{order_dt.date()}_{delivery_dt.date()}.pdf", mime="application/pdf")
    except ImportError:
        st.info("PDF export requires the 'reportlab' package (pip install reportlab)")

# Save PO to history
if st.button("✅ Create Purchase Order and Save to History"):
    po_dict = {row["Product"]: int(row["Recommended PO (packs)"]) for _, row in po_df.iterrows()}
    rows = []
    for p, qty in po_dict.items():
        rows.append({
            "OrderDate": order_dt.date(),
            "DeliveryDateTime": delivery_dt,
            "Product": p,
            "PacksOrdered": qty,
            "PackSize": PACK_SIZES[p],
            "PAR_Days": par_days,
            "AvgDailyPacks": round(avg_daily_packs.get(p, 0.0), 3),
        })
    new = pd.DataFrame(rows)
    if PO_HISTORY_PATH.exists():
        old = pd.read_csv(PO_HISTORY_PATH, parse_dates=["DeliveryDateTime"])
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(PO_HISTORY_PATH, index=False)
    st.success("Purchase Order saved to po_history.csv")

# Show / download PO History
if PO_HISTORY_PATH.exists():
    st.write("**Purchase Order History**")
    po_hist = pd.read_csv(PO_HISTORY_PATH, parse_dates=["DeliveryDateTime"])
    st.dataframe(po_hist.sort_values(["OrderDate", "Product"]).reset_index(drop=True), use_container_width=True)
    st.download_button("⬇️ Download PO History CSV", data=po_hist.to_csv(index=False), file_name="po_history.csv")

st.markdown("---")
st.caption(
    "Upload your ThriveMetrics CSV and we'll convert menu items to bun demand automatically: "
    "Shack Burger Single → Hamburger bun, Shack Double Burger → Hamburger bun, Hot Dog Quarter Pound → Hot Dog bun.\n"
    "Average daily packs = (weekly bun units ÷ pack size) ÷ 7. PAR is applied to stock level right after Friday 5:00 AM delivery."
)
