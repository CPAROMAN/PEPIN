import streamlit as st
import pandas as pd

st.set_page_config(page_title="Pan Pepin Inventory Dashboard (10-Day)", layout="wide")
st.title("🥖 Pan Pepin Inventory Dashboard (10-Day)

# --- Config ---
VARIANTS = ["Hamburger", "Hot Dog Buns"]

# Delivery days by index in 10-day horizon (Mon=0):
# Friday(4)
DELIVERY_DAYS = [4]

# Safety stock
MIN_STOCK = 4
MAX_STOCK = 6

# AM sales split (before delivery at 09:00)
AM_PCT = {
    "Hamburger": 0.05,
    "Hot Dog Buns": 0.10
}

uploaded_file = st.file_uploader(
    "Upload a single CSV with all sales (must include DATE, Modifier Name, Modifier Sold)",
    type="csv"
)

if not uploaded_file:
    st.info("Upload your CSV to begin (use sample_sales.csv if you need a quick test).")
    st.stop()

# --- Load & normalize ---
df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip()

required = {"DATE", "Modifier Name", "Modifier Sold"}
missing = required - set(df.columns)
if missing:
    st.error(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    st.stop()

df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE"]).copy()

# Normalize variant labels from Modifier Name
mod = df["Modifier Name"].astype(str).str.lower().str.strip()

def norm_variant(x: str):
    if "Shack Burger Single" in x: return "Hamburger"
    if "Shack Double Burger" in x: return "Hamburger"
    if "Hot Dog Quarter Pound" in x: return "Hot Dog
        return None

df["Variant"] = mod.map(norm_variant)
df = df[df["Variant"].isin(VARIANTS)].copy()
df["Modifier Sold"] = pd.to_numeric(df["Modifier Sold"], errors="coerce").fillna(0).astype(int)

# Day-of-week (Mon=0..Sun=6) from actual dates
df["Dow"] = df["DATE"].dt.weekday

# Aggregate actuals by day-of-week (Mon..Sun)
daily = df.groupby(["Dow", "Variant"])["Modifier Sold"].sum().unstack(fill_value=0)
daily = daily.reindex(range(7), fill_value=0)
daily = daily.reindex(columns=VARIANTS, fill_value=0)

# Build 9-day series:
# days 0..6 = current Mon..Sun actuals
# day 7 = next Mon (copy of Monday), day 8 = next Tue (copy of Tuesday)
proj = daily.copy()
proj.loc[7] = proj.loc[0]  # Next Monday
proj.loc[8] = proj.loc[1]  # Next Tuesday
proj = proj.sort_index()

st.subheader("🔢 Starting Inventory")
cols = st.columns(len(VARIANTS))
# Defaults: your confirmed starting inventory (adjust as needed)
defaults = [16, 115, 27, 22]
start_inv = {v: cols[i].number_input(v, min_value=0, value=defaults[i]) for i, v in enumerate(VARIANTS)}

def am_split(total: int, pct: float) -> int:
    """Return integer AM portion with safe bounds."""
    am = int(round(total * pct))
    if am < 0: am = 0
    if am > total: am = total
    return am

if st.button("Generate 9-Day PO Plan"):
    records = []
    inv = {v: float(start_inv[v]) for v in VARIANTS}
    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun","Mon+1 (Proj)","Tue+1 (Proj)"]

    for d in range(9):
        row = {"Day": day_names[d]}
        for v in VARIANTS:
            sales = int(proj.loc[d, v]) if v in proj.columns else 0

            if d in DELIVERY_DAYS:
                # Split sales: AM before delivery, PM after delivery
                am = am_split(sales, AM_PCT[v])
                pm = sales - am

                # AM sales occur BEFORE the truck arrives (~09:00)
                inv[v] -= am
                pre_stockout = inv[v] < 0

                # Determine coverage window until next delivery:
                # cover PM of today + full days until next delivery + AM of next delivery day
                next_del = next((x for x in DELIVERY_DAYS if x > d), 9)

                cover = pm  # PM of today
                # full in-between days
                for dd in range(d+1, next_del):
                    cover += int(proj.loc[dd, v]) if dd in proj.index else 0
                # AM of next delivery day (if any)
                if next_del < 9:
                    next_day_sales = int(proj.loc[next_del, v]) if next_del in proj.index else 0
                    cover += am_split(next_day_sales, AM_PCT[v])

                # Size order so that after delivery:
                # inventory covers 'cover' and ends within [MIN_STOCK, MAX_STOCK]
                need_min = cover + MIN_STOCK
                need_max = cover + MAX_STOCK
                order = 0 if inv[v] >= need_min else (need_min - inv[v])
                if inv[v] + order > need_max:
                    order = max(0, need_max - inv[v])
                order = int(round(order))

                inv[v] += order  # receive at 09:00
                inv[v] -= pm     # consume PM sales
                end_stockout = inv[v] < 0

                row[f"{v} PO"] = order
                row[f"{v} EndInv"] = round(inv[v], 2)
                row[f"{v} AM_Stockout"] = "⚠️" if pre_stockout else ""
                row[f"{v} EOD_Stockout"] = "⚠️" if end_stockout else ""

            else:
                # Non-delivery day: all sales hit
                inv[v] -= sales
                stockout = inv[v] < 0
                row[f"{v} PO"] = 0
                row[f"{v} EndInv"] = round(inv[v], 2)
                row[f"{v} AM_Stockout"] = ""  # not applicable
                row[f"{v} EOD_Stockout"] = "⚠️" if stockout else ""

        records.append(row)

    out = pd.DataFrame(records)
    st.subheader("📊 9-Day PO Plan (deliveries @ 09:00, min=10 / max=15)")
    st.dataframe(out, use_container_width=True)

    # ---------- Charts ----------
    st.markdown("### 📈 Inventory Trajectory (End of Day)")
    inv_cols = [f"{v} EndInv" for v in VARIANTS]
    inv_chart_df = out[["Day"] + inv_cols].copy().set_index("Day")
    st.line_chart(inv_chart_df)

    st.markdown("### 📦 Purchase Orders by Day")
    po_cols = [f"{v} PO" for v in VARIANTS]
    po_chart_df = out[["Day"] + po_cols].copy().set_index("Day")
    st.bar_chart(po_chart_df)

    # ---------- Download ----------
    st.download_button(
        "Download PO Plan (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="PO_Plan_9Day.csv",
        mime="text/csv",
    )
