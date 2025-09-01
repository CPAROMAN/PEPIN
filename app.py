# Create sample data files and build a new bundle that includes them.
import pandas as pd
from datetime import date, timedelta, datetime, time
from pathlib import Path
import zipfile

base = Path("/mnt/data")

# 1) Sample weekly sales history (4 recent Sun–Sat weeks)
def last_sunday(d: date) -> date:
    return d - timedelta(days=(d.weekday()+1) % 7)

today = date.today()
start_sun = last_sunday(today - timedelta(days=7*4))
rows = []
for w in range(4):
    ws = start_sun + timedelta(days=7*w)
    we = ws + timedelta(days=6)
    rows.append({
        "WeekStart": ws.isoformat(),
        "WeekEnd": we.isoformat(),
        "Hamburger Units": 160 + 5*w,   # simple ascending pattern
        "Hot Dog Buns Units": 300 + 10*w
    })
weekly_df = pd.DataFrame(rows)
weekly_path = base / "weekly_sales_history.csv"
weekly_df.to_csv(weekly_path, index=False)

# 2) Sample PO history (two entries)
po_rows = [
    {
        "OrderDate": (today - timedelta(days=10)).isoformat(),
        "DeliveryDateTime": datetime.combine(today - timedelta(days=7), time(5,0)).isoformat(),
        "Product": "Hamburger",
        "PacksOrdered": 20,
        "PackSize": 8,
        "PAR_Days": 9,
        "AvgDailyPacks": 2.5,
        "TrayRule": "half-tray multiples"
    },
    {
        "OrderDate": (today - timedelta(days=10)).isoformat(),
        "DeliveryDateTime": datetime.combine(today - timedelta(days=7), time(5,0)).isoformat(),
        "Product": "Hot Dog Buns",
        "PacksOrdered": 36,
        "PackSize": 10,
        "PAR_Days": 9,
        "AvgDailyPacks": 4.8,
        "TrayRule": "half-tray multiples"
    },
]
po_hist_df = pd.DataFrame(po_rows)
po_hist_path = base / "po_history.csv"
po_hist_df.to_csv(po_hist_path, index=False)

# 3) Sample daily CSV (DATE/ITEM/QTY)
daily_rows = []
for i in range(7):  # one week sample
    d = today - timedelta(days=7 - i)
    daily_rows.append({"DATE": d.isoformat(), "ITEM": "Shack Burger Single", "QTY": 12+i})
    daily_rows.append({"DATE": d.isoformat(), "ITEM": "Shack Double Burger", "QTY": 6+i//2})
    daily_rows.append({"DATE": d.isoformat(), "ITEM": "Hot Dog Quarter Pound", "QTY": 40+i*2})
daily_df = pd.DataFrame(daily_rows)
daily_path = base / "sample_daily_sales.csv"
daily_df.to_csv(daily_path, index=False)

# 4) Build a new bundle including samples
bundle = base / "pepin_par_app_with_samples.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in ("app.py", "requirements.txt", "Dockerfile", ".dockerignore", "README.md",
                  "weekly_sales_history.csv", "po_history.csv", "sample_daily_sales.csv"):
        fp = base / fname
        if fp.exists():
            zf.write(fp, arcname=fname)

str(weekly_path), str(po_hist_path), str(daily_path), str(bundle)
