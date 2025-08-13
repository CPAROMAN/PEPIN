import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from io import BytesIO

# =========================
# App Config
# =========================
st.set_page_config(page_title="Pan Pepín Orders & Inventory (PAR)", layout="wide")
st.title("🥖 Pan Pepín Orders & Inventory Dashboard")
st.caption("Sun–Sat weekly totals • Order Tuesdays → Deliver Fridays @ 5:00 AM • PAR defaults to 10 days")

# =========================
# Constants & Defaults
# =========================
# Default products & pack sizes (can be edited in Admin > Products)
PACK_SIZES = {"Hamburger": 8, "Hot Dog Buns": 10}
VALID_PRODUCTS = ["Hamburger", "Hot Dog Buns"]

# Default mapping: sales items → buns required per unit (editable in Admin)
THRIVE_TO_BUNS = {
    "shack burger single": ("Hamburger", 1),
    "shack double burger": ("Hamburger", 1),
    "hot dog quarter pound": ("Hot Dog Buns", 1),
}

DEFAULT_AM_PCT = {
    "Hamburger": 0.05,    # 5% of day’s sales occur before delivery time
    "Hot Dog Buns": 0.10, # 10% of day’s sales occur before delivery time
}

# -------------------------
# Admin controls
# -------------------------
# Change this PIN to control who can edit mappings / PO values
ADMIN_PIN = "4321"
# If True, PO editing (packs_to_order) also requires PIN unlock
LOCK_PO_EDIT = True

# =========================
# Helpers
# =========================

def flexible_read_table(uploaded_file) -> pd.DataFrame:
    """Robustly read CSV or Excel from a Streamlit UploadedFile.
    - Tries Excel if the filename ends with .xlsx/.xls
    - Tries multiple CSV delimiters (",", ";", "	", "|")
    - Tries encodings (utf-8-sig, utf-8, latin1)
    - Tries skipping up to 5 header lines (common in vendor exports)
    """
    name = getattr(uploaded_file, "name", "").lower()
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()

    # Try Excel first if extension hints it
    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(BytesIO(data))
        except Exception:
            pass

    # CSV attempts
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    # (engine, sep) pairs; sep=None enables delimiter auto-detect (python engine only)
    attempts = [("c", ","), ("python", None), ("python", ";"), ("python", "	"), ("python", "|")]
    for enc in encodings:
        for engine, sep in attempts:
            for skip in [0, 1, 2, 3, 4, 5]:
                try:
                    kwargs = dict(engine=engine, encoding=enc, skiprows=skip)
                    if sep is not None:
                        kwargs["sep"] = sep
                    else:
                        # sep=None only valid with python engine
                        kwargs["sep"] = None
                    return pd.read_csv(BytesIO(data), **kwargs)
                except Exception:
                    continue

    # Last-ditch Excel attempt (if user mis-labeled extension)
    try:
        return pd.read_excel(BytesIO(data))
    except Exception as e:
        raise ValueError(f"Could not parse file as CSV or Excel: {e}")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _guess_col(columns, synonyms):
    # exact match first
    for c in columns:
        lc = str(c).strip().lower()
        if lc in synonyms:
            return c
    # then substring match
    for c in columns:
        lc = str(c).strip().lower()
        for s in synonyms:
            if s in lc:
                return c
    return None


def _suggest_mapping(df: pd.DataFrame):
    cols = list(df.columns)
    date_syn = {
        "date", "business date", "order date", "transaction date", "date/time", "datetime",
        "business_day", "fecha", "fecha de venta", "sales date"
    }
    item_syn = {
        "modifier name", "item name", "menu item", "menu item name", "item", "product", "product name",
        "plu name", "description", "nombre", "articulo", "artículo", "producto"
    }
    qty_syn = {
        "modifier sold", "quantity", "qty", "qty sold", "units", "units sold", "total quantity",
        "net qty", "net quantity", "cantidad", "cant", "count", "sold"
    }
    d = _guess_col(cols, date_syn)
    i = _guess_col(cols, item_syn)
    q = _guess_col(cols, qty_syn)
    return {"date": d, "item": i, "qty": q}


def load_sales(uploaded_file: BytesIO) -> pd.DataFrame:
    """Load a single CSV/Excel and normalize into columns: date, item, qty.
    Tries common Thrive exports and many header variants. If auto-detect fails,
    shows dropdowns so you can pick the Date / Item / Quantity columns.
    """
    try:
        df = flexible_read_table(uploaded_file)
    except Exception as e:
        raise ValueError(
            f"Could not parse file. Upload a CSV or Excel (.xlsx) with the expected headers. Original error: {e}"
        )

    df = _standardize_columns(df)

    # Candidate column mappings (exact names)
    candidates = [
        {"date": "date", "item": "modifier name", "qty": "modifier sold"},
        {"date": "date", "item": "item name", "qty": "quantity"},
        {"date": "date", "item": "item", "qty": "qty"},
    ]

    mapping = None
    for cand in candidates:
        if all(col in df.columns for col in cand.values()):
            mapping = cand
            break

    # Heuristic mapping over many variants
    if mapping is None:
        hints = _suggest_mapping(df)
        if all(hints.values()):
            mapping = hints

    # If still unknown, prompt the user to choose columns
    if mapping is None:
        st.warning(
            "Couldn't auto-detect your column names. Please select which columns contain the **Date**, **Item**, and **Quantity**.")
        options = list(df.columns)

        def _idx(name):
            try:
                return options.index(name)
            except Exception:
                return 0

        hints = _suggest_mapping(df)
        date_col = st.selectbox("Date column", options, index=_idx(hints.get("date")), key="map_date_col")
        item_col = st.selectbox("Item column", options, index=_idx(hints.get("item")), key="map_item_col")
        qty_col = st.selectbox("Quantity column", options, index=_idx(hints.get("qty")), key="map_qty_col")
        mapping = {"date": date_col, "item": item_col, "qty": qty_col}

    out = pd.DataFrame({
        "date": pd.to_datetime(df[mapping["date"]], errors="coerce").dt.date,
        "item": df[mapping["item"]].astype(str).str.strip(),
        "qty": pd.to_numeric(df[mapping["qty"]], errors="coerce").fillna(0).astype(int),
    })
    out = out.dropna(subset=["date"])  # keep only rows with valid dates
    out = out[out["qty"] > 0]
    return out



def map_sales_to_buns(sales_df: pd.DataFrame, mapping_dict: dict) -> pd.DataFrame:
    """Explode sales rows into bun requirements using mapping_dict."""
    if sales_df.empty:
        return pd.DataFrame(columns=["date", "product", "buns_required"])  

    def map_row(item_name: str):
        key = item_name.lower()
        if key in mapping_dict:
            return mapping_dict[key]
        # Fallback: partial contains match
        for k, v in mapping_dict.items():
            if k in key:
                return v
        return None

    rows = []
    for _, r in sales_df.iterrows():
        mapped = map_row(r["item"])
        if mapped is None:
            continue
        bun_kind, buns_per_unit = mapped
        rows.append({
            "date": r["date"],
            "product": bun_kind,
            "buns_required": r["qty"] * buns_per_unit,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "product", "buns_required"])  
    return df


def aggregate_daily_bun_demand(bun_df: pd.DataFrame) -> pd.DataFrame:
    if bun_df.empty:
        return pd.DataFrame(columns=["date", "product", "demand"])
    g = bun_df.groupby(["date", "product"], as_index=False)["buns_required"].sum()
    g = g.rename(columns={"buns_required": "demand"})
    return g


def make_horizon(dates: pd.Series, horizon_days: int) -> pd.DatetimeIndex:
    if dates.empty:
        start = datetime.today().date()
    else:
        start = min(dates)
    return pd.date_range(start=start, periods=horizon_days, freq="D").date


def plan_projection(daily_demand: pd.DataFrame,
                    on_hand: dict,
                    deliveries_weekdays: list,
                    delivery_time: time,
                    am_pct: dict,
                    horizon_days: int = 10,
                    pack_sizes: dict = PACK_SIZES,
                    par_days: int = 10) -> dict:
    """Return dict with projection table. (PO suggestion will be built separately.)

    - daily_demand: columns [date, product, demand]
    - on_hand: {product: units}
    - deliveries_weekdays: list of weekday ints (Mon=0..Sun=6) when deliveries occur
    - delivery_time: delivery happens before PM sales; AM sales happen first
    - am_pct: fraction of a day's demand that occurs before delivery time
    - horizon_days: days to project
    - par_days: not used here for PO (kept for compatibility)
    """
    products = list(pack_sizes.keys())

    # Prepare horizon index
    dates = make_horizon(daily_demand["date"] if not daily_demand.empty else pd.Series([], dtype="datetime64[ns]"), horizon_days)
    idx = pd.MultiIndex.from_product([dates, products], names=["date", "product"])

    # Daily demand pivot
    dd = daily_demand.set_index(["date", "product"]).reindex(idx, fill_value=0).reset_index()

    # Split demand into AM/PM using am_pct
    dd["am_demand"] = dd.apply(lambda r: int(round(r["demand"] * am_pct.get(r["product"], 0.0))), axis=1)
    dd["pm_demand"] = dd["demand"] - dd["am_demand"]

    # Projection table
    proj_rows = []
    stock = {p: int(on_hand.get(p, 0)) for p in products}

    for d in dates:
        weekday = pd.to_datetime(d).weekday()
        for p in products:
            # Start of day stock
            start_stock = stock[p]

            # AM sales
            am = int(dd[(dd["date"] == d) & (dd["product"] == p)]["am_demand"].values[0])
            after_am = start_stock - am

            # Delivery (if scheduled that weekday)
            delivered_units = 0  # quantity decided by PO; shown here as 0 in projection
            if weekday in deliveries_weekdays:
                after_delivery = after_am + delivered_units
            else:
                after_delivery = after_am

            # PM sales
            pm = int(dd[(dd["date"] == d) & (dd["product"] == p)]["pm_demand"].values[0])
            end_stock = after_delivery - pm

            proj_rows.append({
                "date": d,
                "product": p,
                "start": start_stock,
                "am_sales": am,
                "delivered": delivered_units,
                "pm_sales": pm,
                "end": end_stock,
            })

            stock[p] = end_stock

    projection = pd.DataFrame(proj_rows)

    return {"projection": projection, "daily": dd}


def build_po_excel(po_df: pd.DataFrame, order_dt: datetime.date) -> bytes:
    """Return an .xlsx file as bytes with the PO table."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Leave space for a title above the table
        po_df.to_excel(writer, sheet_name="PO", index=False, startrow=2)
        wb = writer.book
        ws = writer.sheets["PO"]
        # Title
        header_fmt = wb.add_format({"bold": True})
        ws.write(0, 0, f"Pan Pepín PO — {order_dt}", header_fmt)
        # Autofit columns (approx)
        for col_idx, col_name in enumerate(po_df.columns):
            width = max(12, len(str(col_name)) + 2)
            ws.set_column(col_idx, col_idx, width)
    return output.getvalue()


def build_po_pdf(po_df: pd.DataFrame, order_dt: datetime.date) -> bytes:
    """Return a simple PDF with the PO table."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as e:
        raise RuntimeError("reportlab is required to export PDF. Please install it: pip install reportlab") from e

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []
    title = Paragraph(f"<b>Pan Pepín Purchase Order — {order_dt}</b>", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Build table data
    cols = ["product", "avg_daily_demand", "par_days", "target_units", "on_hand", "units_to_order", "pack_size", "packs_to_order"]
    headers = [c.replace("_", " ").title() for c in cols]
    data = [headers] + po_df[cols].values.tolist()

    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
    ]))

    elements.append(t)
    doc.build(elements)
    return buf.getvalue()

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("Settings")

horizon_days = st.sidebar.number_input("Horizon (days)", min_value=7, max_value=21, value=10, step=1)
par_days = st.sidebar.number_input("PAR days (coverage target)", min_value=3, max_value=30, value=10, step=1)

st.sidebar.subheader("Starting Inventory (units)")
start_hamb = st.sidebar.number_input("Hamburger", min_value=0, value=0, step=1)
start_hotdog = st.sidebar.number_input("Hot Dog Buns", min_value=0, value=0, step=1)

st.sidebar.subheader("Delivery Schedule")
weekday_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
selected_days = st.sidebar.multiselect("Delivery Days", options=list(weekday_map.keys()), default=["Fri"])
deliveries_weekdays = [weekday_map[d] for d in selected_days]

hour = st.sidebar.slider("Delivery Hour (24h)", min_value=0, max_value=23, value=5)
minute = st.sidebar.slider("Delivery Minute", min_value=0, max_value=59, value=0)

st.sidebar.subheader("AM Sales Split (before delivery)")
am_hamb = st.sidebar.slider("Hamburger AM %", min_value=0, max_value=50, value=int(DEFAULT_AM_PCT["Hamburger"] * 100)) / 100.0
am_hot = st.sidebar.slider("Hot Dog Buns AM %", min_value=0, max_value=50, value=int(DEFAULT_AM_PCT["Hot Dog Buns"] * 100)) / 100.0

am_pct = {"Hamburger": am_hamb, "Hot Dog Buns": am_hot}

# PAR method settings
st.sidebar.subheader("PAR Method")
par_method = st.sidebar.radio(
    "Method",
    ["Simple average (all sales)", "Moving average (last N days)"],
    index=0,
)
lookback_days = (
    st.sidebar.number_input("Lookback days", min_value=3, max_value=60, value=14, step=1)
    if "Moving" in par_method
    else 14
)

on_hand = {"Hamburger": start_hamb, "Hot Dog Buns": start_hotdog}

# =========================
# Admin: Orderable Products (packs)
# =========================

st.divider()
st.subheader("Admin: Orderable Products (packs)")
st.caption("Manage products and pack sizes used for orders. This also controls options in the mapping table and PO.")

admin_unlocked = st.session_state.get("admin_unlocked", False)

DEFAULT_PRODUCTS = [
    {"product": "Hamburger", "pack_size": 8},
    {"product": "Hot Dog Buns", "pack_size": 10},
]

if "products_df" not in st.session_state:
    st.session_state["products_df"] = pd.DataFrame(DEFAULT_PRODUCTS)

prod_c1, prod_c2, prod_c3 = st.columns([1, 1, 2])
with prod_c1:
    if st.button("Reset products", disabled=not admin_unlocked):
        st.session_state["products_df"] = pd.DataFrame(DEFAULT_PRODUCTS)
        st.success("Products reset to defaults.")
with prod_c2:
    prod_csv = st.file_uploader("Import products CSV/Excel (product,pack_size)", type=["csv", "xlsx", "xls"], key="products_csv")
    if prod_csv is not None:
        try:
            imp = flexible_read_table(prod_csv)
            imp.columns = [c.strip().lower() for c in imp.columns]
            req = {"product", "pack_size"}
            if not req.issubset(set(imp.columns)):
                raise ValueError("CSV must have columns: product, pack_size")
            imp = imp.dropna(subset=["product"]).copy()
            imp["product"] = imp["product"].astype(str).str.strip()
            imp["pack_size"] = pd.to_numeric(imp["pack_size"], errors="coerce").fillna(0).astype(int)
            imp = imp[imp["pack_size"] > 0]
            if imp.empty:
                raise ValueError("No valid rows after validation.")
            st.session_state["products_df"] = imp.drop_duplicates(subset=["product"]).reset_index(drop=True)
            st.success("Products imported.")
        except Exception as e:
            st.error(f"Import failed: {e}")
with prod_c3:
    st.download_button(
        "Download products CSV",
        data=st.session_state["products_df"].to_csv(index=False),
        file_name="products_pack_sizes.csv",
        mime="text/csv",
        disabled=not admin_unlocked and False,
    )

products_editor = st.data_editor(
    st.session_state["products_df"],
    num_rows="dynamic",
    hide_index=True,
    disabled=not admin_unlocked,
    column_config={
        "product": st.column_config.TextColumn("Product"),
        "pack_size": st.column_config.NumberColumn("Pack Size", min_value=1, step=1),
    },
    key="products_editor",
)

st.session_state["products_df"] = products_editor

# Build current product list and pack size dict for downstream use
_valid_products_list = (
    st.session_state["products_df"]["product"].dropna().astype(str).str.strip().tolist()
)
_valid_products_list = [p for p in _valid_products_list if p]
_pack_sizes_dict = {}
for _, r in st.session_state["products_df"].iterrows():
    try:
        _pack_sizes_dict[str(r["product"]).strip()] = int(r["pack_size"])
    except Exception:
        pass

if not _valid_products_list or not _pack_sizes_dict:
    _valid_products_list = [p["product"] for p in DEFAULT_PRODUCTS]
    _pack_sizes_dict = {p["product"]: p["pack_size"] for p in DEFAULT_PRODUCTS}

# Override global defaults used by planner & mapping editor
VALID_PRODUCTS = _valid_products_list
PACK_SIZES = _pack_sizes_dict

# =========================
# Admin: Item → Bun Mapping (editable)
# =========================

st.divider()
st.subheader("Admin: Item → Bun Mapping")
st.caption("Add or edit how menu items consume buns. Case-insensitive; partial matches allowed.")

# --- PIN Lock ---
if "admin_unlocked" not in st.session_state:
    st.session_state["admin_unlocked"] = False

pin_col1, pin_col2 = st.columns([1, 2])
with pin_col1:
    input_pin = st.text_input("PIN", type="password", placeholder="Enter PIN")
with pin_col2:
    if st.button("Unlock / Lock"):
        if st.session_state["admin_unlocked"]:
            st.session_state["admin_unlocked"] = False
        else:
            if input_pin == ADMIN_PIN:
                st.session_state["admin_unlocked"] = True
            else:
                st.error("Incorrect PIN.")

admin_unlocked = st.session_state["admin_unlocked"]

_default_map_rows = [
    {"item": "Shack Burger Single", "product": "Hamburger", "buns_per_unit": 1},
    {"item": "Shack Double Burger", "product": "Hamburger", "buns_per_unit": 1},
    {"item": "Hot Dog Quarter Pound", "product": "Hot Dog Buns", "buns_per_unit": 1},
]

if "mapping_df" not in st.session_state:
    st.session_state["mapping_df"] = pd.DataFrame(_default_map_rows)

map_btn_c1, map_btn_c2, map_btn_c3 = st.columns([1, 1, 2])
with map_btn_c1:
    if st.button("Reset mapping", disabled=not admin_unlocked):
        st.session_state["mapping_df"] = pd.DataFrame(_default_map_rows)
        st.success("Mapping reset to defaults.")
with map_btn_c2:
    mapping_csv = st.file_uploader(
        "Import mapping CSV/Excel (item,product,buns_per_unit)", type=["csv", "xlsx", "xls"], key="mapping_csv")
    if mapping_csv is not None:
        try:
            imp = flexible_read_table(mapping_csv)
            imp.columns = [c.strip().lower() for c in imp.columns]
            required = {"item", "product", "buns_per_unit"}
            if not required.issubset(set(imp.columns)):
                raise ValueError("CSV must have columns: item, product, buns_per_unit")
            imp = imp.dropna(subset=["item", "product"]).copy()
            imp["item"] = imp["item"].astype(str)
            imp["product"] = imp["product"].astype(str)
            imp["buns_per_unit"] = pd.to_numeric(imp["buns_per_unit"], errors="coerce").fillna(0).astype(int)
            # enforce valid product names from current product list
            bad = ~imp["product"].isin(VALID_PRODUCTS)
            if bad.any():
                raise ValueError("Invalid 'product' values present. Allowed: " + ", ".join(VALID_PRODUCTS))
            st.session_state["mapping_df"] = imp
            st.success("Mapping imported.")
        except Exception as e:
            st.error(f"Import failed: {e}")
with map_btn_c3:
    st.download_button(
        "Download mapping CSV",
        data=st.session_state["mapping_df"].to_csv(index=False),
        file_name="item_to_bun_mapping.csv",
        mime="text/csv",
        disabled=not admin_unlocked and False,
    )

mapping_editor = st.data_editor(
    st.session_state["mapping_df"],
    num_rows="dynamic",
    hide_index=True,
    disabled=not admin_unlocked,
    column_config={
        "item": st.column_config.TextColumn("Item Name"),
        "product": st.column_config.SelectboxColumn("Bun Kind", options=VALID_PRODUCTS),
        "buns_per_unit": st.column_config.NumberColumn("Buns per Unit", min_value=0, step=1),
    },
    key="mapping_editor",
)

st.session_state["mapping_df"] = mapping_editor
_active_mapping = {}
for _, row in mapping_editor.dropna(subset=["item", "product"]).iterrows():
    try:
        bpu = int(row.get("buns_per_unit", 0))
    except Exception:
        bpu = 0
    if bpu > 0:
        _active_mapping[str(row["item"]).strip().lower()] = (str(row["product"]).strip(), bpu)

if not _active_mapping:
    _active_mapping = {
        "shack burger single": ("Hamburger", 1),
        "shack double burger": ("Hamburger", 1),
        "hot dog quarter pound": ("Hot Dog Buns", 1),
    }

THRIVE_TO_BUNS = _active_mapping

# =========================
# File Upload & Processing
# =========================

uploaded_file = st.file_uploader(
    "Upload a single **CSV or Excel** with all sales (must include DATE and either (Modifier Name, Modifier Sold) or (Item Name, Quantity))",
    type=["csv", "xlsx", "xls"],
)

if not uploaded_file:
    st.info("Upload your CSV to begin. If you need a quick test, export a small range from Thrive Metrics.")
    st.stop()

try:
    sales = load_sales(uploaded_file)
except Exception as e:
    st.error(str(e))
    st.stop()

bun_rows = map_sales_to_buns(sales, THRIVE_TO_BUNS)
daily = aggregate_daily_bun_demand(bun_rows)

plan = plan_projection(
    daily_demand=daily,
    on_hand={p: on_hand.get(p, 0) for p in VALID_PRODUCTS},
    deliveries_weekdays=deliveries_weekdays,
    delivery_time=time(hour, minute),
    am_pct={p: am_pct.get(p, 0.0) for p in VALID_PRODUCTS},
    horizon_days=horizon_days,
    pack_sizes=PACK_SIZES,
    par_days=par_days,
)

projection = plan["projection"]

# --- Build PO suggestion using chosen PAR method and current products ---
all_products = VALID_PRODUCTS
if daily.empty:
    avg_map = {p: 0.0 for p in all_products}
else:
    last_date = max(daily["date"]) if not daily.empty else None
    if last_date is not None and "Moving" in par_method:
        cutoff = pd.to_datetime(last_date) - pd.Timedelta(days=int(lookback_days) - 1)
        dsub = daily[pd.to_datetime(daily["date"]) >= cutoff]
    else:
        dsub = daily
    avg_series = dsub.groupby("product")["demand"].mean() if not dsub.empty else pd.Series(dtype=float)
    avg_map = {p: float(avg_series.get(p, 0.0)) for p in all_products}

po_rows = []
for p in all_products:
    avg = avg_map.get(p, 0.0)
    target_units = int(np.ceil(avg * par_days))
    current_units = int(on_hand.get(p, 0))
    need_units = max(0, target_units - current_units)
    pack = PACK_SIZES.get(p, 1)
    packs_to_order = int(np.ceil(need_units / max(1, pack)))
    po_rows.append({
        "product": p,
        "avg_daily_demand": round(avg, 2),
        "par_days": par_days,
        "target_units": target_units,
        "on_hand": current_units,
        "units_to_order": packs_to_order * pack,
        "pack_size": pack,
        "packs_to_order": packs_to_order,
    })

po_df = pd.DataFrame(po_rows)

# =========================
# Layout
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Projected Inventory by Day")
    st.dataframe(
        projection.rename(columns={
            "date": "Date",
            "product": "Product",
            "start": "Start Stock",
            "am_sales": "AM Sales",
            "delivered": "Delivered",
            "pm_sales": "PM Sales",
            "end": "End Stock",
        })
    )

with col2:
    st.subheader("PO Suggestion (Next Delivery)")
    st.caption("Edit the number of packs to order. Units update automatically based on pack size.")

    admin_unlocked = st.session_state.get("admin_unlocked", False)

    # Initialize persistent editable PO in session state
    if "po_df_final" not in st.session_state:
        st.session_state["po_df_final"] = po_df.copy()

    # Show editable PO (packs_to_order editable only when unlocked if LOCK_PO_EDIT)
    editable_po = st.data_editor(
        st.session_state["po_df_final"].assign(
            units_to_order=lambda d: d["packs_to_order"] * d["pack_size"],
        ),
        hide_index=True,
        num_rows="fixed",
        disabled=(LOCK_PO_EDIT and (not admin_unlocked)),
        column_config={
            "product": st.column_config.TextColumn("Product", disabled=True),
            "avg_daily_demand": st.column_config.NumberColumn("Avg Daily", disabled=True, format="%.2f"),
            "par_days": st.column_config.NumberColumn("PAR Days", disabled=True),
            "target_units": st.column_config.NumberColumn("Target Units", disabled=True),
            "on_hand": st.column_config.NumberColumn("On Hand", disabled=True),
            "pack_size": st.column_config.NumberColumn("Pack Size", disabled=True),
            "packs_to_order": st.column_config.NumberColumn("Packs to Order", min_value=0, step=1),
            "units_to_order": st.column_config.NumberColumn("Units (auto)", disabled=True),
        },
        key="po_editor",
    )

    # Persist edits back to session_state
    st.session_state["po_df_final"]["packs_to_order"] = editable_po["packs_to_order"].fillna(0).astype(int)
    st.session_state["po_df_final"]["units_to_order"] = (
        st.session_state["po_df_final"]["packs_to_order"] * st.session_state["po_df_final"]["pack_size"]
    )

    # Reset to suggestion button
    if st.button("Reset PO to suggestion"):
        st.session_state["po_df_final"] = po_df.copy()
        st.success("PO reset to current suggestion.")

# =========================
# Exports
# =========================

st.subheader("Export Purchase Order")
order_date = datetime.today().date()

# Use the edited PO if available; otherwise base suggestion
po_to_export = st.session_state.get("po_df_final", po_df).copy()

c1, c2 = st.columns(2)

with c1:
    if st.button("⬇️ Download PO (Excel)"):
        try:
            xlsx = build_po_excel(po_to_export, order_date)
            st.download_button(
                label="Save PO.xlsx",
                data=xlsx,
                file_name=f"PanPepin_PO_{order_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Excel export failed: {e}")

with c2:
    if st.button("⬇️ Download PO (PDF)"):
        try:
            pdf = build_po_pdf(po_to_export, order_date)
            st.download_button(
                label="Save PO.pdf",
                data=pdf,
                file_name=f"PanPepin_PO_{order_date}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"PDF export failed: {e}")

st.success("Ready. Review or edit the PO (packs), adjust PAR method if needed, and export.")
