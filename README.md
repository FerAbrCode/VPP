# ORC Knots Predictor (local web app)

Small Flask web app that trains on `USA_ORC.csv` using only these inputs:

- D, LOA, IMSL, CDL, DRAFT, BMAX, DSPL, DSPS, CREW, WSS, INDEX, DA, MAIN, GENOA, SYM, ASYM

It predicts knots at all R{TWA}{TWS} points by first predicting time allowance (s/Nm) with a multi-output RandomForest and converting with `knots = 3600 / (s/Nm)`.

## Files
- app.py — Flask app with training and prediction
- templates/index.html — UI with manual and CSV import
- static/style.css — simple styling
- requirements.txt — Python dependencies

## Data location
The app expects `USA_ORC.csv` to be in the parent folder of this app directory (i.e. under `.../VPP/USA_ORC.csv`). This matches your existing files.

## Run (Windows PowerShell)
```powershell
# From the webapp folder
cd "c:\Users\fabla\OneDrive - Universidad Politécnica de Cartagena\freetime\Coding\VPP\webapp"

# (Optional) Create and activate a venv
python -m venv .venv
. .venv\Scripts\Activate.ps1

# Install deps
pip install -r requirements.txt

# Launch
python app.py
```
Then open http://127.0.0.1:5000/ in your browser.

## CSV template
Click the "Download template.csv" button in the UI. It contains exactly the required columns and one example row constructed from your ORC data.

## Notes
- Categorical D is normalized to first uppercase letter and one-hot encoded with fixed levels [C,S,R].
- Numeric inputs are median-imputed based on USA_ORC statistics.
- For CSV uploads, predictions are returned as a downloaded CSV including your input columns plus new columns: `K{TWA}_{TWS}` with knots.
