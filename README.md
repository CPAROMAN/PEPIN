# Pan Pepín Orders & Inventory (Streamlit)

Sun–Sat weekly totals • Order **Tuesdays** → Deliver **Fridays @ 5:00 AM** • **PAR = 10 days** by default.

This app helps you enter weekly unit sales for **Hamburger** and **Hot Dog Buns**, projects demand, checks Tue/Wed/Thu stockout risk, and recommends the **Purchase Order** (in whole packs). It also exports PO to **Excel** and **PDF** and keeps **weekly sales history** and **PO history** CSVs.

---

## ✨ Features
- **Manual weekly totals (Sun–Sat)** in **units** for both products
- **Average Daily Usage (packs)** = `(weekly units ÷ pack size) ÷ 7`
  - Hamburger pack size = **8**; Hot Dog Buns pack size = **10**
  - Three‑decimal precision everywhere (e.g., `10.125` packs)
- **Cycle**: Order **Tuesday** → Deliver **Friday 5:00 AM** (overrideable in sidebar)
- **Days of Supply** & **Stockout risk (Tue/Wed/Thu of delivery week)**
- **PO math (packs)**:  
  `Calc Need = (PAR days × Avg Daily Packs) + (Days until delivery × Avg Daily Packs) − Current On‑Hand Packs`  
  `Recommended PO = ceil(max(0, Calc Need))`
- **Exports**: PO to **Excel (.xlsx)** and **PDF**
- **Histories**: weekly sales & PO saved under `data/`
- **Environment check**: shows/install helpers for `xlsxwriter`, `openpyxl`, `reportlab`

---

## 🛠️ Quick Start (local)

1. **Save the app code** as `app.py` (use the code from your canvas).
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scriptsctivate
   # macOS/Linux
   source .venv/bin/activate

   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Run**:
   ```bash
   streamlit run app.py
   ```
4. Open the browser tab (usually http://localhost:8501).

> If you don’t have `requirements.txt`, create it with the content from this repo (below).

---

## ☁️ Deploy on Streamlit Cloud

1. Put `app.py` and `requirements.txt` in a GitHub repo.
2. Go to https://share.streamlit.io → **New app**.
3. Select the repo and set **Main file path** to `app.py`.
4. **Deploy**.

**Important**: Files under `data/` are **ephemeral** on Streamlit Cloud. For long‑term history, consider syncing to cloud storage later (S3/GCS).

---

## 📂 Data & Outputs

- **Weekly Sales History**: `data/weekly_sales_history.csv` (editable in-app)
- **PO History**: `data/po_history.csv` (append-only; created when you save a PO)
- **Exports**:
  - **Excel**: `PO_YYYY-MM-DD_YYYY-MM-DD.xlsx` (Summary + PO sheets)
  - **PDF**: `PO_YYYY-MM-DD_YYYY-MM-DD.pdf` (print-friendly table)

---

## ⚙️ Configuration Tips

- **PAR days**: Sidebar control (default = 10)
- **Planning dates**: Toggle “Override auto Tue→Fri” to set specific **Order** and **Delivery** dates/time
- **On‑hand**: Enter current packs to 3 decimals (e.g., `7.003`)
- **Lookback weeks**: Choose how many past weeks drive the projection

---

## ❗ Troubleshooting

- **Excel export button disabled / error** → Install an engine:
  ```bash
  pip install xlsxwriter   # preferred
  # or
  pip install openpyxl
  ```
- **PDF export error** →
  ```bash
  pip install reportlab
  ```
- **Buttons in sidebar don’t install libs** (on Streamlit Cloud): runtime installs can be restricted. Add the package names to `requirements.txt` and redeploy.
- **Syntax errors**: Ensure your `pip_install()` function uses `"\n"` inside strings (no literal line breaks in quotes).

---

## 📜 License
Internal use. Add your license terms here if needed.
