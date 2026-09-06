# IT5006 Olist E-Commerce EDA Dashboard

## Set your local data folder

The CSV files are intentionally not stored in GitHub. Each user must point the app to the folder containing all nine Olist CSV files.

Open `app.py`, find `CUSTOM_DATA_DIRECTORY` near the top, and change only:

```python
CUSTOM_DATA_DIRECTORY = ""
```

macOS example:

```python
CUSTOM_DATA_DIRECTORY = r"/Users/your-name/path/to/Olist_CSV"
```

Windows example:

```python
CUSTOM_DATA_DIRECTORY = r"C:\Users\your-name\path\to\Olist_CSV"
```

Use the folder containing the CSVs—not an individual CSV file. Keep the `r` before the path, especially on Windows. Do not commit your personal path; restore `CUSTOM_DATA_DIRECTORY = ""` before committing.

Alternatively, start the app and paste the folder path into **Data location** in the sidebar.

## First-time setup

### macOS

```bash
cd "/path/to/Team8_IT5006_Ecommerce_Analytics_AY2627Sem1/phase1_eda_dashboard"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Windows PowerShell

```powershell
cd "C:\path\to\Team8_IT5006_Ecommerce_Analytics_AY2627Sem1\phase1_eda_dashboard"
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Subsequent runs

macOS:

```bash
cd "/path/to/Team8_IT5006_Ecommerce_Analytics_AY2627Sem1/phase1_eda_dashboard"
source .venv/bin/activate
python -m streamlit run app.py
```

Windows PowerShell:

```powershell
cd "C:\path\to\Team8_IT5006_Ecommerce_Analytics_AY2627Sem1\phase1_eda_dashboard"
.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Open <http://localhost:8501> if the browser does not open automatically. Press Control+C to stop the app.
