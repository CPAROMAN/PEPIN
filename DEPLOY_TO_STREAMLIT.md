# Deploy to Streamlit Cloud

## 1) Prepare your repo
1. Create a new folder on your computer.
2. Save your Streamlit code from the canvas as **app.py** in that folder.
3. Put this **requirements.txt** file in the same folder (use the one I provided).

## 2) Push to GitHub
If you use the GitHub Desktop app: add the folder and publish the repository.
Or via command line:
```
git init
git add app.py requirements.txt
git commit -m "Pan Pepin Orders & Inventory app"
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
- The app exports Excel (xlsxwriter or openpyxl) and PDF (reportlab). All are listed in `requirements.txt`.
- If you later move the app file name, update the Streamlit Cloud "Main file path".
- You can store PO/weekly history on Streamlit Cloud but it’s ephemeral. For persistence, consider mounting a remote storage (S3, GCS) later.
