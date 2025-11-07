## ORC Knots Predictor (VPP Local Web App)

Flask-based web application that trains a multi-output RandomForest on ORC rating data (`USA_ORC.csv`) using these input features:

D, LOA, IMSL, CDL, DRAFT, BMAX, DSPL, DSPS, CREW, WSS, INDEX, DA, MAIN, GENOA, SYM, ASYM

It predicts time allowance (s/Nm) for each performance point R{TWA}{TWS} and converts to boat speed (knots) using:

knots = 3600 / (seconds_per_nautical_mile)

### Features
- Upload or manually enter boat parameters
- Multi-output RandomForest training
- Batch polar prediction
- Simple HTML/CSS UI

### Repository Layout
webapp/
  app.py
  templates/
    index.html
  static/
    style.css
requirements.txt
USA_ORC.csv (expected one level above webapp unless you change the path)

### Data Location
By default the app expects:
VPP/
  USA_ORC.csv
  webapp/
    app.py

If you move the CSV into webapp/, change the path in app.py.

### Setup & Run (Windows PowerShell)
git clone https://github.com/FerAbrCode/VPP.git
cd VPP/webapp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r ..\requirements.txt
python app.py

Open http://127.0.0.1:5000/

### Regenerating requirements
python -m pip install <package>
python -m pip freeze | Out-File -Encoding UTF8 requirements.txt

### Attribution
Data variable descriptions adapted from:
jieter/orc-data  https://github.com/jieter/orc-data

Include LICENSE or attribution required by that repository. If MIT:
Data & variable descriptions © Original authors of jieter/orc-data (MIT License). Used under MIT terms.

### Optional CSV path env var
In app.py:
csv_path = os.getenv("ORC_DATA_PATH", "../USA_ORC.csv")

Then:
$env:ORC_DATA_PATH = "C:\full\path\USA_ORC.csv"
python app.py

### Contributing
Fork → branch → commit → PR.

### License
Add a LICENSE file (MIT recommended for code). Respect upstream data license.

### FAQ
Q: Why no absolute paths?  
A: Relative paths make the project portable.

Q: Where put USA_ORC.csv?  
A: Repo root (sibling to webapp/) unless you set ORC_DATA_PATH.

Q: How to change model?  
A: Modify the RandomForest parameters in app.py; retrain.

---

For feature requests (export polar JSON, model tuning) open an issue.
