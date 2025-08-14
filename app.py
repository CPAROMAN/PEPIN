import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
from io import BytesIO

# ============================================================
# Pan Pepín Orders & Inventory Dashboard (Clean Rewrite v1.0)
# ============================================================
# Key features
# - Single CSV with all sales. Flexible column mapping.
# - Thrivemetrics exports supported automatically (Product, Sold, Date/DATE).
# - Map menu items to buns consumed (Single/Double -> Hamburger; Quarter Pound -> Hot Dog Buns).
# - Beginning inventory in PACKS (3 decimals allowed).
# - Delivery applied AFTER early AM sales (default 09:00) to respect 4:00 AM opening.
# - 9-day projection window (configurable).
# - Simple demand forecast using recent weekday averages (fallback to overall mean).
# - Generates an Excel workbook with PO and inventory schedule.
# ============================================================

st.set_page_config(page_title="Pan Pepín Orders & Inventory (Clean)", layout="wide")
st.title("🥖 Pan Pepín — Orders & Inventory (Clean Rewrite)")
st.caption("Single CSV → map columns → simulate usage in packs → suggest PO → export Excel")

# -----------------------------
# Constants and defaults
# -----------------------------
VARIANTS = ["Hamburger", "Hot Dog Buns"]
PACK_SIZES_DEFAULT = {"Hamburger": 8, "Hot Dog Buns": 10}
AM_SPLIT_DEFAULT = {"Hamburger": 0.05, "Hot Dog Buns": 0.10}  # share of daily sales before delivery time

# Known menu-item → bun variant mapping (can be extended in UI)
MENU_TO_BUN_DEFAULT = {
    # Thrivemetrics examples provided by the user
    "Shack Burger Single": "Hamburger",
    "Shack Single": "Hamburger",
    "Shack Double Burger": "Hamburger",
    "Shack Double": "Hamburger",
    "Hot Dog Quarter Pound": "Hot Dog Buns",
}

# -----------------------------
# Sidebar configuration
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    # Pack sizes
    st.subheader("Pack sizes (units per pack)")
    pack_sizes = {}
    for v in VARIANTS:
        pack_sizes[v] = st.number_input(
            f"{v}", min_value=1, max_value=60, value=int(PACK_SIZES_DEFAULT.get(v, 8)), step=1
        )
