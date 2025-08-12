# Deploy to Streamlit Cloud

## 1) Prepare your repo
1. Create a new folder on your computer.
2. Save the `app.py` file (from this zip) into that folder.
3. Put this `requirements.txt` file in the same folder.

## 2) Push to GitHub
If you use the GitHub Desktop app: add the folder and publish the repository.
Or via command line:
```
git init
git add app.py requirements.txt
git commit -m "Pan Pepín Orders & Inventory app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 3) Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io and click **New app**.
2. Choose the repo you just pushed.
3. Set **Main file path** to `app.py`.
4. Click **Deploy**. Done!

### Notes
- Excel export: requires `xlsxwriter` **or** `openpyxl` (both included here).
- PDF export: requires `reportlab` (included here).
- Data saved to `data/` is ephemeral on Streamlit Cloud. For persistence, consider a cloud bucket (S3/GCS) later.
