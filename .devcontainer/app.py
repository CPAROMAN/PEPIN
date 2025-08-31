
import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from pathlib import Path
from io import BytesIO
import importlib.util
import re
from typing import Optional, Tuple, Dict, Any

# =========================
# App Config
# =========================
st.set_page_config(page_title="Pan Pepín Orders & Inventory (PAR)", layout="wide")
st.title("🥖 Pan Pepín Orders & Inventory Dashboard")
st.caption("Sun–Sat weekly totals • Order Tuesdays → Deliver Fridays @ 5:00 AM • PAR defaults to 9 days")

# =========================
# Constants
# =========================
# Units per pack (how many buns per pack you receive)
PACK_SIZES = {"Hamburger": 8, "Hot Dog Buns": 10}
VALID_PRODUCTS = list(PACK_SIZES.keys())

# Default PAR days (user requested 9‑day projection)
PAR_DAYS_DEFAULT = 9
DEFAULT_HISTORY_WEEKS = 8

TRAY_SIZE_PACKS = 8
HALF_TRAY_PACKS = TRAY_SIZE_PACKS // 2  # 4‑pack rounding blocks

# Item → Bun mapping (normalize incoming POS/menu names to the two bun SKUs)
ITEM_TO_BUN = {
    "shack burger single": "Hamburger",
    "shack double burger": "Hamburger",
    "hot dog quarter pound": "Hot Dog Buns",
}

# Column synonym sets for flexible import
DATE_SYNS = {"date", "business date", "order date", "created", "created at", "sale date", "date/time", "timestamp"}
ITEM_SYNS = {"item", "item name", "menu item", "product", "product name", "name", "description", "modifier name", "modifier", "pos item", "sku", "plu"}
QTY_SYNS  = {"qty", "quantity", "qty sold", "sold qty", "units", "units sold", "items sold", "count", "sold", "modifier sold", "# sold", "quantity sold"}

# =========================
# Safe writable base path
# =========================
def _pick_writable_base() -> Path:
    env_dir = os.getenv("PEPIN_DATA_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates += [
        Path("/tmp/pepin"),
        Path.home() / ".cache" / "pepin",
        Path.cwd() / ".pepin",
        Path("/mount/data/pepin"),  # Streamlit Cloud persistent volume (if available)
    ]
    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".pepin_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return base
        except Exception:
            continue
    # Last‑resort fallback
    fallback = Path(tempfile.gettempdir()) / "pepin"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

BASE_DIR = _pick_writable_base()
PO_HISTORY_PATH = BASE_DIR / "po_history.csv"
WEEKLY_HISTORY_PATH = BASE_DIR / "weekly_sales_history.csv"

# =========================
# Utility helpers
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
    """Return last full Sun–Sat week as (Sunday, Saturday) normalized to midnight."""
    yday = pd.Timestamp(today.date()) - pd.Timedelta(days=1)
    last_sun = yday - pd.Timedelta(days=(yday.weekday() - 6) % 7)
    return last_sun.normalize(), (last_sun + pd.Timedelta(days=6)).normalize()

def next_weekday(d: datetime, target_weekday: int) -> datetime:
    """0=Mon..6=Sun; returns next target weekday strictly after 'd'."""
    days_ahead = (target_weekday - d.weekday() + 7) % 7
    return d + timedelta(days=days_ahead or 7)

def next_order_and_delivery(now: datetime):
    """Auto-pick next Tue (order) and Fri 5:00 AM (delivery)."""
    next_tue = next_weekday(now, 1)  # Tuesday
    next_fri = next_weekday(now, 4)  # Friday
    if next_fri <= next_tue:
        next_fri += timedelta(days=7)
    return next_tue, datetime.combine(next_fri.date(), time(5, 0))

# =========================
# ThriveMetrics PDF parsing
# =========================
@st.cache_data(show_spinner=False)
def _read_thrivemetrics_pdf_to_rows(file) -> Tuple[pd.DataFrame, str]:
    """Return raw table-like dataframe + joined full text; be flexible if tables not detected."""
    try:
        import pdfplumber  # type: ignore
    except Exception as e:
        raise ImportError("PDF import requires pdfplumber. Try: pip install pdfplumber") from e

    # Read raw bytes (works for UploadedFile or path-like)
    try:
        file.seek(0)
        raw_bytes = file.read()
    except Exception:
        with open(file, "rb") as f:
            raw_bytes = f.read()

    full_text_parts, tables = [], []
    with pdfplumber.open(BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            full_text_parts.append(page.extract_text() or "")
            extracted = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "edge_min_length": 20,
            }) or []
            for tbl in extracted:
                if not tbl or len(tbl) < 2:
                    continue
                header = [(c or "").strip() for c in tbl[0]]
                rows = tbl[1:]
                df = pd.DataFrame(rows, columns=header)
                tables.append(df)

    full_text = "\n".join(full_text_parts)

    if tables:
        raw_df = pd.concat(tables, ignore_index=True)
        raw_df = raw_df.loc[:, raw_df.columns.map(lambda c: str(c).strip() != "")]
        raw_df.columns = [str(c).strip().lower() for c in raw_df.columns]
        return raw_df, full_text

    # Fallback: try to parse "name ... NNN" lines where the last token is an integer
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    prod, qty = [], []
    for ln in lines:
        m = re.search(r"(.+?)\s+([0-9][0-9,]*)$", ln)
        if m:
            name = m.group(1).strip().lower()
            q = m.group(2).replace(",", "")
            try:
                qv = int(q)
                prod.append(name)
                qty.append(qv)
            except Exception:
                pass
    if prod:
        raw_df = pd.DataFrame({"product": prod, "sold": qty})
        return raw_df, full_text

    raise ValueError("Could not detect any table or parse lines from PDF. Make sure you exported the correct ThriveMetrics report.")

def _extract_report_dates_from_text(text: str):
    # "from Aug 24, 2025 ... to Aug 30, 2025"
    m = re.search(r"from\s+([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}).*?to\s+([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        d1 = pd.to_datetime(m.group(1), errors="coerce")
        d2 = pd.to_datetime(m.group(2), errors="coerce")
        return (None if pd.isna(d1) else d1.date(), None if pd.isna(d2) else d2.date())
    # single date
    m2 = re.search(r"([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})", text)
    if m2:
        d = pd.to_datetime(m2.group(1), errors="coerce")
        if not pd.isna(d):
            return (d.date(), d.date())
    # or ISO date
    m3 = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m3:
        d = pd.to_datetime(m3.group(1), errors="coerce")
        if not pd.isna(d):
            return (d.date(), d.date())
    return (None, None)

def _pick_col(df: pd.DataFrame, syns) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() in syns:
            return c
    for c in df.columns:
        lc = str(c).strip().lower()
        for s in syns:
            if s in lc:
                return c
    return None

def parse_thrivemetrics_pdf(file) -> pd.DataFrame:
    raw_df, full_text = _read_thrivemetrics_pdf_to_rows(file)
    if raw_df.empty:
        raise ValueError("PDF appears to be empty.")
    item_col = _pick_col(raw_df, ITEM_SYNS.union({"product"}))
    qty_col  = _pick_col(raw_df, QTY_SYNS.union({"sold"}))
    if not all([item_col, qty_col]):
        if raw_df.shape[1] >= 2:
            item_col = item_col or raw_df.columns[0]
            qty_col  = qty_col  or raw_df.columns[1]
        else:
            raise ValueError("Could not detect Product/Sold columns in PDF.")
    items = raw_df[item_col].astype(str).str.strip().str.lower()
    qty   = pd.to_numeric(raw_df[qty_col].astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0)
    d_from, d_to = _extract_report_dates_from_text(full_text)
    chosen = d_to or d_from or datetime.now().date()
    dates = pd.Series([chosen] * len(items), dtype="object")
    out = pd.DataFrame({"date": dates, "item": items, "qty": qty})
    out = out[(out["item"] != "") & (~out["item"].str.contains("^total$", na=False))]
    return out

# =========================
# CSV (daily rows) parser
# =========================
def parse_single_csv(file) -> pd.DataFrame:
    """Accepts a CSV with at least a DATE, ITEM, QTY (synonyms OK). Returns normalized rows."""
    df = pd.read_csv(file)
    if df.empty:
        raise ValueError("CSV is empty.")
    # Normalize headers
    df.columns = [str(c).strip() for c in df.columns]
    # Pick columns
    dcol = _pick_col(df, DATE_SYNS)
    icol = _pick_col(df, ITEM_SYNS)
    qcol = _pick_col(df, QTY_SYNS)
    if not all([dcol, icol, qcol]):
        raise ValueError("Could not find DATE/ITEM/QTY columns. Try renaming headers or provide a template‑like CSV.")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[dcol], errors="coerce"),
        "item": df[icol].astype(str).str.strip().str.lower(),
        "qty":  pd.to_numeric(df[qcol], errors="coerce").fillna(0),
    }).dropna(subset=["date"])
    out["date"] = out["date"].dt.date
    out = out[(out["item"] != "") & (~out["item"].str.contains("^total$", na=False))]
    return out

# =========================
# Transform items → weekly bun units
# =========================
def buns_weekly_from_items(df_items: pd.DataFrame) -> pd.DataFrame:
    df = df_items.copy()
    df["bun_product"] = df["item"].map(ITEM_TO_BUN)
    df = df.dropna(subset=["bun_product"])
    df["bun_units"] = df["qty"].astype(float)
    # Week buckets: Sunday .. Saturday
    dt_series = pd.to_datetime(df["date"]).dt.date
    weekday = pd.to_datetime(df["date"]).dt.weekday
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
        ).reset_index()
    )
    for need in ["Hamburger", "Hot Dog Buns"]:
        if need not in piv.columns:
            piv[need] = 0.0
    weekly = piv.rename(columns={"Hamburger": "Hamburger Units", "Hot Dog Buns": "Hot Dog Buns Units"})
    weekly = weekly[["WeekStart", "WeekEnd", "Hamburger Units", "Hot Dog Buns Units"]].sort_values("WeekStart")
    return weekly

# =========================
# Inventory math
# =========================
def simulate_consumption_until_delivery(on_hand_packs, avg_daily_packs, start_dt, delivery_dt):
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
            rows.append({"Date": disp_date, **{k: round(v, 3) for k, v in used_today.items()}, **{f"{p} EoD Stock (packs)": round(stock[p], 3) for p in VALID_PRODUCTS}})
        cursor = slice_end
    return {p: round(stock[p], 3) for p in VALID_PRODUCTS}, pd.DataFrame(rows)

def _round_up_to_half_tray(packs: float) -> int:
    if packs <= 0:
        return 0
    multiple = HALF_TRAY_PACKS
    return int(np.ceil(packs / multiple) * multiple)

def recommend_po(on_hand_now_packs, avg_daily_packs, par_days, start_dt, delivery_dt, enforce_trays: bool):
    days_until_delivery = max(0.0, (delivery_dt - start_dt).total_seconds() / 86400.0)
    calc_need, reco = {}, {}
    for p in VALID_PRODUCTS:
        avg = float(avg_daily_packs.get(p, 0.0))
        cur = float(on_hand_now_packs.get(p, 0.0))
        need = round((par_days * avg) + (days_until_delivery * avg) - cur, 3)
        calc_need[p] = need
        base = max(0.0, need)
        if enforce_trays:
            reco[p] = _round_up_to_half_tray(base)
        else:
            reco[p] = int(np.ceil(base))
    return calc_need, reco

# =========================
# Sidebar Controls
# =========================
now = datetime.now()
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"Data folder: `{BASE_DIR}`")
    par_days = st.number_input("PAR level (days)", 1, 30, PAR_DAYS_DEFAULT)
    lookback_weeks = st.slider("Weeks of history for projections", 2, 26, DEFAULT_HISTORY_WEEKS)

    st.markdown("---")
    st.subheader("📦 Current On‑Hand (packs)")
    on_hand_packs = {p: st.number_input(f"{p} (packs)", min_value=0.0, step=0.001, value=0.0, format="%.3f") for p in VALID_PRODUCTS}

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
    st.subheader("📦 Tray/Case Rules")
    enforce_trays = st.checkbox(
        f"Enforce multiples of {HALF_TRAY_PACKS} packs (≥½‑tray; full tray = {TRAY_SIZE_PACKS} packs)",
        value=True,
        help="Rounds up Recommended PO (packs) to the nearest 4‑pack block (0, 4, 8, 12, ...)."
    )

    st.markdown("---")
    st.subheader("🧪 Environment Check")
    libs = ["pdfplumber", "xlsxwriter", "openpyxl", "reportlab"]
    missing = []
    for lib in libs:
        ok, ver = lib_info(lib)
        st.write(("✅" if ok else "❌") + f" **{lib}**" + (f" v{ver}" if (ok and ver) else ""))
        if not ok:
            missing.append(lib)
    if missing:
        st.caption("Install missing: " + " ".join([f"`pip install {m}`" for m in missing]))

# =========================
# Data Ingest
# =========================
st.subheader("Sales Data Source")
mode = st.radio("Choose input method", ["Upload ThriveMetrics PDF (recommended)", "Upload CSV (DATE/ITEM/QTY)", "Manual weekly entry"], index=0)

items_df: Optional[pd.DataFrame] = None

if mode.startswith("Upload ThriveMetrics PDF"):
    up = st.file_uploader("Upload ThriveMetrics Product Sales report (PDF)", type=["pdf"])
    if up is None:
        st.info("Upload a PDF to build weekly Sun–Sat bun units automatically from items: Shack Burger Single, Shack Double Burger, Hot Dog Quarter Pound.")
        st.stop()
    try:
        try:
            up.seek(0)
        except Exception:
            pass
        items_df = parse_thrivemetrics_pdf(up)
        st.caption("Detected/selected normalized rows (first 10):")
        st.dataframe(items_df.head(10), use_container_width=True)
        weekly = buns_weekly_from_items(items_df)
        weekly.to_csv(WEEKLY_HISTORY_PATH, index=False)
        st.success("PDF parsed. Weekly bun unit totals generated and saved.")
        st.dataframe(weekly, use_container_width=True)
    except Exception as e:
        st.error(f"Could not parse PDF: {e}")
        st.stop()

elif mode.startswith("Upload CSV"):
    up = st.file_uploader("Upload CSV with daily rows (DATE, ITEM, QTY). Headers can vary; we'll auto‑detect.", type=["csv"])
    if up is None:
        st.info("Upload a CSV with columns similar to DATE, ITEM, QTY (synonyms OK). We'll aggregate to Sun–Sat weeks.")
        st.stop()
    try:
        items_df = parse_single_csv(up)
        st.caption("Normalized rows (first 10):")
        st.dataframe(items_df.head(10), use_container_width=True)
        weekly = buns_weekly_from_items(items_df)
        weekly.to_csv(WEEKLY_HISTORY_PATH, index=False)
        st.success("CSV parsed. Weekly bun unit totals generated and saved.")
        st.dataframe(weekly, use_container_width=True)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        st.stop()

else:
    st.caption("Enter total UNITS per week for each bun product. Avg daily packs = (weekly units ÷ pack size) ÷ 7.")
    if WEEKLY_HISTORY_PATH.exists():
        weekly = pd.read_csv(WEEKLY_HISTORY_PATH)
    else:
        ps, pe = last_full_sun_sat_week(now)
        weekly = pd.DataFrame([{"WeekStart": ps.date(), "WeekEnd": pe.date(), "Hamburger Units": 0, "Hot Dog Buns Units": 0}])
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
                if pd.isna(r["WeekEnd"]) else r["WeekEnd"],
                axis=1,
            )
            temp.to_csv(WEEKLY_HISTORY_PATH, index=False)
            st.success("Saved.")
    with colB:
        st.download_button("⬇️ Download Weekly Sales History CSV", data=edited.to_csv(index=False), file_name="weekly_sales_history.csv")
    weekly = edited
    if weekly.empty:
        st.stop()

# =========================
# Build projections
# =========================
hist = weekly.dropna(subset=["WeekStart"]).copy().sort_values("WeekStart")
recent = hist.tail(lookback_weeks)

avg_daily_packs: Dict[str, float] = {}
for p in VALID_PRODUCTS:
    units_col = "Hamburger Units" if p == "Hamburger" else "Hot Dog Buns Units"
    weekly_mean_units = recent[units_col].astype(float).mean() if not recent.empty else 0.0
    avg_daily_packs[p] = round((weekly_mean_units / PACK_SIZES[p]) / 7.0, 3)

with st.expander("🧪 Quick QA Check (dev only)", expanded=False):
    if not hist.empty:
        ws = pd.to_datetime(hist.iloc[-1]["WeekStart"]).date()
        we = pd.to_datetime(hist.iloc[-1]["WeekEnd"]).date()
        st.markdown(f"**Detected Week (Sun–Sat):** `{ws}` → `{we}`")
    else:
        st.info("No weekly rows available to summarize.")
    if items_df is not None and not items_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Items (top 8)")
            st.dataframe(items_df.head(8), use_container_width=True)
        with c2:
            try:
                _tmp = items_df.copy()
                _tmp["bun_product"] = _tmp["item"].map(ITEM_TO_BUN)
                by_item = (
                    _tmp.groupby(["item", "bun_product"], dropna=False)["qty"]
                    .sum()
                    .reset_index()
                    .sort_values("qty", ascending=False)
                )
                st.caption("Item → Bun mapping (summed qty)")
                st.dataframe(by_item, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not build mapping summary: {e}")
    avg_tbl = pd.DataFrame({"Product": VALID_PRODUCTS, "Avg Daily Packs": [avg_daily_packs[p] for p in VALID_PRODUCTS], "Pack Size": [PACK_SIZES[p] for p in VALID_PRODUCTS]})
    avg_tbl["Avg Daily Packs"] = avg_tbl["Avg Daily Packs"].astype(float).round(3)
    st.caption("Average Daily Usage (packs) = (weekly units ÷ pack size) ÷ 7")
    st.dataframe(avg_tbl, use_container_width=True)
    try:
        _proj_at_delivery, _ = simulate_consumption_until_delivery(on_hand_packs, avg_daily_packs, datetime.now(), next_order_and_delivery(now)[1])
        _calc_need, _raw_reco = recommend_po(on_hand_packs, avg_daily_packs, PAR_DAYS_DEFAULT, datetime.now(), next_order_and_delivery(now)[1], enforce_trays=False)
        _rounded_reco = recommend_po(on_hand_packs, avg_daily_packs, PAR_DAYS_DEFAULT, datetime.now(), next_order_and_delivery(now)[1], enforce_trays=True)[1]
        prev_rows = []
        for p in VALID_PRODUCTS:
            prev_rows.append({"Product": p, "Projected @ Delivery (packs)": round(_proj_at_delivery[p], 3), "Avg Daily (packs)": round(avg_daily_packs[p], 3), "Calc Need (packs)": round(_calc_need[p], 3), "Recommended (free)": int(_raw_reco[p]), f"Recommended (½‑tray {HALF_TRAY_PACKS})": int(_rounded_reco[p])})
        st.caption("PO rounding preview (free vs. ½‑tray multiples)")
        st.dataframe(pd.DataFrame(prev_rows), use_container_width=True)
    except Exception as e:
        st.info(f"PO preview will appear after dates/inputs are set. ({e})")

# Prev week quick summary
prev_row = hist.tail(1)
prev_label = "—"
prev_summary = None
if not prev_row.empty:
    ws = pd.to_datetime(prev_row.iloc[0]["WeekStart"]).date()
    we = pd.to_datetime(prev_row.iloc[0]["WeekEnd"]).date()
    prev_label = f"{ws} to {we}"
    prev_summary = {"Hamburger Units": int(prev_row.iloc[0].get("Hamburger Units", 0)), "Hot Dog Buns Units": int(prev_row.iloc[0].get("Hot Dog Buns Units", 0))}

order_dt = locals().get("order_dt", now)
delivery_dt = locals().get("delivery_dt", next_order_and_delivery(now)[1])

proj_at_delivery, per_day_usage = simulate_consumption_until_delivery(on_hand_packs, avg_daily_packs, datetime.now(), delivery_dt)
po_calc_need, po_reco = recommend_po(on_hand_packs, avg_daily_packs, par_days, datetime.now(), delivery_dt, enforce_trays)

# Stockout risk Tue/Wed/Thu of the delivery week
ww = delivery_dt.weekday()
week_tue = (delivery_dt + timedelta(days=(1 - ww))).date()
week_wed = (delivery_dt + timedelta(days=(2 - ww))).date()
week_thu = (delivery_dt + timedelta(days=(3 - ww))).date()
risk_rows = []
for p in VALID_PRODUCTS:
    risk = {d: False for d in ["Tuesday", "Wednesday", "Thursday"]}
    for dname, dval in [("Tuesday", week_tue), ("Wednesday", week_wed), ("Thursday", week_thu)]:
        row = per_day_usage[per_day_usage["Date"].dt.date == dval]
        if not row.empty and float(row[f"{p} EoD Stock (packs)"].values[0]) < 0:
            risk[dname] = True
    risk_rows.append({"Product": p, **risk})
risk_df = pd.DataFrame(risk_rows)

# =========================
# Reporting blocks
# =========================
st.subheader("Last Recorded Week – Units & Packs")
col1, col2 = st.columns([2, 3], gap="large")
with col1:
    st.write(f"**Week:** {prev_label}")
    if prev_summary is None:
        st.info("No weekly rows available yet.")
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
    st.write("**Average Daily Usage (packs)** – (weekly units ÷ pack size) ÷ 7")
    avg_tbl = pd.DataFrame({"Product": VALID_PRODUCTS, "Avg Daily Packs": [avg_daily_packs[p] for p in VALID_PRODUCTS], "Pack Size": [PACK_SIZES[p] for p in VALID_PRODUCTS]})
    avg_tbl["Avg Daily Packs"] = avg_tbl["Avg Daily Packs"].astype(float).round(3)
    st.dataframe(avg_tbl, use_container_width=True)
with col4:
    st.write("**Days of Supply On‑Hand**")
    dos_rows = []
    for p in VALID_PRODUCTS:
        avg_use = max(1e-6, float(avg_daily_packs.get(p, 0.0)))
        dos_rows.append({"Product": p, "On‑Hand (packs)": round(float(on_hand_packs.get(p, 0.0)), 3), "Days Remaining": round(float(on_hand_packs.get(p, 0.0)) / avg_use, 2)})
    st.dataframe(pd.DataFrame(dos_rows), use_container_width=True)
with col5:
    st.write("**Stockout Risk (Tue/Wed/Thu of delivery week)**")
    st.dataframe(risk_df, use_container_width=True)

st.markdown("---")

# =========================
# PO Recommendation + Exports
# =========================
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
        "Recommended PO (packs)": int(packs_reco),
        "Pack Size (units/pack)": PACK_SIZES[p],
        "Recommended PO (units)": int(packs_reco * PACK_SIZES[p]),
    })
po_df = pd.DataFrame(po_rows)

st.info("You can edit 'Recommended PO (packs)' below. If 'Enforce multiples of 4 packs' is ON, downloads/history will re‑round to the nearest 4.")
po_df_editable = st.data_editor(
    po_df,
    use_container_width=True,
    column_config={"Recommended PO (packs)": st.column_config.NumberColumn(min_value=0, step=1)},
    key="po_editor",
)

po_df_final = po_df_editable.copy()
if enforce_trays:
    po_df_final["Recommended PO (packs)"] = po_df_final["Recommended PO (packs)"].apply(_round_up_to_half_tray)
po_df_final["Recommended PO (units)"] = po_df_final.apply(lambda r: int(r["Recommended PO (packs)"] * r["Pack Size (units/pack)"]), axis=1)

def build_po_excel(po_df: pd.DataFrame, order_dt: datetime, delivery_dt: datetime, par_days: int):
    engine = None
    try:
        import xlsxwriter
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl
            engine = "openpyxl"
        except Exception:
            return None, "Excel export requires xlsxwriter or openpyxl."
    bio = BytesIO()
    df = po_df.copy()
    for c in [c for c in df.columns if "(packs)" in c and "Recommended" not in c]:
        df[c] = df[c].astype(float).round(3)
    with pd.ExcelWriter(bio, engine=engine) as w:
        pd.DataFrame({"Field": ["Order Date", "Delivery", "PAR Days"], "Value": [order_dt.date(), delivery_dt, par_days]}).to_excel(w, index=False, sheet_name="Summary")
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
    bio = BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=LETTER, title="Pan Pepín – Purchase Order")
    styles = getSampleStyleSheet()
    elems = []
    elems += [
        Paragraph("Pan Pepín – Purchase Order", styles["Title"]),
        Paragraph(f"Order Date: {order_dt.date()}  •  Delivery: {delivery_dt}  •  PAR: {par_days} days", styles["Normal"]),
        Spacer(1, 12),
    ]
    headers = list(po_df.columns)
    data = [headers]
    for _, r in po_df.iterrows():
        row = []
        for c in headers:
            v = r[c]
            row.append(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v))
        data.append(row)
    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
    ]))
    elems.append(t)
    doc.build(elems)
    return bio.getvalue()

colX, colY = st.columns(2)
with colX:
    excel_bytes, excel_err = build_po_excel(po_df_final, order_dt, delivery_dt, par_days)
    if excel_bytes:
        st.download_button("⬇️ Export PO to Excel (.xlsx)", data=excel_bytes, file_name=f"PO_{order_dt.date()}_{delivery_dt.date()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info(excel_err)
with colY:
    try:
        pdf_bytes = build_po_pdf(po_df_final, order_dt, delivery_dt, par_days)
        st.download_button("🖨️ Print PO (PDF)", data=pdf_bytes, file_name=f"PO_{order_dt.date()}_{delivery_dt.date()}.pdf", mime="application/pdf")
    except ImportError:
        st.info("PDF export requires the 'reportlab' package (pip install reportlab)")

# =========================
# Save PO history
# =========================
if st.button("✅ Create Purchase Order and Save to History"):
    po_dict = {row["Product"]: int(row["Recommended PO (packs)"]) for _, row in po_df_final.iterrows()}
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
            "TrayRule": "half-tray multiples" if enforce_trays else "free"
        })
    new = pd.DataFrame(rows)
    if PO_HISTORY_PATH.exists():
        old = pd.read_csv(PO_HISTORY_PATH, parse_dates=["DeliveryDateTime"])
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(PO_HISTORY_PATH, index=False)
    st.success(f"Purchase Order saved to {PO_HISTORY_PATH.name}")

if PO_HISTORY_PATH.exists():
    st.write("**Purchase Order History**")
    po_hist = pd.read_csv(PO_HISTORY_PATH, parse_dates=["DeliveryDateTime"])
    st.dataframe(po_hist.sort_values(["OrderDate", "Product"]).reset_index(drop=True), use_container_width=True)
    st.download_button("⬇️ Download PO History CSV", data=po_hist.to_csv(index=False), file_name="po_history.csv")

st.markdown("---")
with st.expander("ℹ️ Deployment & Storage Tips", expanded=False):
    st.markdown("""
- **Avoid PermissionError**: this app writes to a safe data dir it auto‑detects. You can override by setting the environment variable **`PEPIN_DATA_DIR=/mount/data/pepin`** on Streamlit Cloud.
- **Data files**:
  - Weekly history → `weekly_sales_history.csv`
  - PO history → `po_history.csv`
- **Docker**: ensure system packages for `reportlab` (freetype, libjpeg) are installed. See sample Dockerfile from the bundle you built earlier.
- **PDF parsing**: use ThriveMetrics **Product Sales Report** covering a Sun–Sat range.
""")
