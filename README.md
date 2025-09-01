
# Pan Pepín Orders & Inventory (PAR) — Streamlit App

Sun–Sat weekly aggregation, PAR-based order recommendations, half-tray rounding, and multi-channel sharing.

## Features
- **ThriveMetrics PDF** parser (Product Sales Report) → maps menu items to bun SKUs.
- **CSV ingest** with flexible headers (DATE/ITEM/QTY synonyms).
- **Manual weekly entry** (Sun–Sat) with in-app editing and CSV export.
- **PAR logic** (default **9 days**) and **delivery at Friday 5:00 AM**.
- **½‑tray rounding** (4-pack blocks) with optional free rounding.
- **Excel (.xlsx)** and **PDF** PO exports.
- **PO History** (`po_history.csv`) and **Weekly Sales History** (`weekly_sales_history.csv`).
- **Messaging**:
  - 📱 **SMS** preview (packs and optional units) + **Copy to Clipboard** + **Download .txt**
  - 💬 **WhatsApp via Twilio**
  - ✉️ **SMS via Twilio**
  - 📧 **Email via SendGrid** (plain text or **HTML table**)

## Quick Start (Local)
```bash
# 1) Create a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Streamlit Cloud
1. Push `app.py`, `requirements.txt`, `Dockerfile`, `.dockerignore` (optional), and `README.md` to a repo.
2. Create an app in Streamlit Cloud pointing at `app.py`.
3. **Environment variables** (Settings → Advanced → Environment variables):
   - `PEPIN_DATA_DIR=/mount/data/pepin`  *(recommended for stable storage)*
   - `TWILIO_SID=...`
   - `TWILIO_AUTH=...`
   - `SENDGRID_API_KEY=...`

> The app automatically finds a safe writable folder. If `PEPIN_DATA_DIR` is set, that path is used.

## Docker
A minimal `Dockerfile` is included. Build and run:
```bash
docker build -t pepin-par .
docker run --rm -p 8501:8501 -e PEPIN_DATA_DIR=/data -v $(pwd)/data:/data pepin-par
```
Open http://localhost:8501

## Data Files
- `weekly_sales_history.csv` — saved under the data dir.
- `po_history.csv` — saved under the data dir.

## Using the App
1. **Choose data source** (PDF / CSV / Manual weekly).
2. **Verify weekly aggregation** under “Weekly Sales History (Units)”. Adjust lookback weeks in sidebar.
3. Set **current on-hand** (packs), **PAR level**, and **delivery date/time**.
4. Review **Next Order Recommendation** table. Edit recommended packs if needed. (Rounding is re-applied if enabled.)
5. **Export** PO to Excel/PDF and/or **Save to History**.
6. **Share** the PO:
   - Copy the **SMS** text or **Download .txt**.
   - Send **SMS or WhatsApp** via Twilio.
   - Send **Email** (plain text or **HTML**) via SendGrid.

## Twilio (SMS & WhatsApp)
- Install dependency (already in requirements): `twilio`
- Environment:
  - `TWILIO_SID`, `TWILIO_AUTH`
- **SMS**:
  - *From:* Twilio phone number in E.164 format (e.g., `+17875551234`)
  - *To:* recipient in E.164 format.
- **WhatsApp**:
  - Use approved numbers or the WhatsApp Sandbox.
  - Numbers must start with `whatsapp:+` (e.g., `whatsapp:+17875551234`).

## SendGrid (Email)
- We use the HTTP API via `requests` (no SDK required).
- Environment: `SENDGRID_API_KEY`
- Choose **plain text** or **HTML** email in-app. HTML includes a simple table with Product / Packs / Units.

## Troubleshooting
- **PermissionError** at startup: set `PEPIN_DATA_DIR` to a writable path (e.g., `/mount/data/pepin` on Streamlit Cloud).
- **Missing dependencies**: the sidebar shows an environment check; install any missing libs.
- **PDF parse errors**: ensure you export **ThriveMetrics Product Sales Report** (Sun–Sat).

---

© Pan Pepín PAR Dashboard. Built with Streamlit.
