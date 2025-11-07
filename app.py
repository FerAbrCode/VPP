from __future__ import annotations
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for, flash, jsonify

# ML import
try:
    from sklearn.ensemble import RandomForestRegressor
except Exception as e:
    raise ImportError(
        "scikit-learn is required. Please install it (pip install scikit-learn pandas numpy flask)."
    ) from e

app = Flask(__name__)
app.secret_key = "orc-webapp-secret"

# Config
TWAs = [52, 60, 75, 90, 110, 120, 135, 150]
TWSs = [4, 6, 8, 10, 12, 14, 16, 20, 24]
FEATURE_COLS = ['D','LOA','IMSL','CDL','DRAFT','BMAX','DSPL','DSPS','CREW','WSS','INDEX','DA','MAIN','GENOA','SYM','ASYM']
TARGET_COLS = [f"R{a}{w}" for a in TWAs for w in TWSs]
CAT_LEVELS = ['C','S','R']

# Descriptions for manual form boxes
FEATURE_DESCRIPTIONS = {
    'D': "IMS Division, one of 'C' (Cruiser/Racer), 'S' (Sportboat) or 'R' (Racer)",
    'LOA': 'Length over all (meters)',
    'IMSL': 'IMS Sailing Length',
    'CDL': 'Class Division Length',
    'DRAFT': 'Draft (meters)',
    'BMAX': 'Maximum beam (meters)',
    'DSPL': 'Displacement (kg)',
    'DSPS': 'Displacement in sailing trim (kg)',
    'CREW': 'Crew weight (kg)',
    'WSS': 'Wetted surface (m²)',
    'INDEX': 'Stability Index',
    'DA': 'Dynamic Allowance',
    'MAIN': 'Maximum main sail area (m²)',
    'GENOA': 'Maximum genoa area (m²)',
    'SYM': 'Maximum symmetrical spinnaker area (m²)',
    'ASYM': 'Maximum asymmetrical spinnaker area (m²)',
}

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent  # .../VPP
USA_PATH = BASE_DIR / 'USA_ORC.csv'
ESP_PATH = BASE_DIR / 'ESP_ORC.csv'

# Globals for trained artifacts
rf_model: Optional[RandomForestRegressor] = None
num_cols: List[str] = []
medians: Dict[str, float] = {}
usa_targets: List[str] = []
last_export_df: Optional[pd.DataFrame] = None


def load_csv_robust(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ['utf-8-sig', 'utf-8', 'latin-1']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return pd.read_csv(path)


def train_model() -> None:
    global rf_model, num_cols, medians, usa_targets
    df = load_csv_robust(USA_PATH)

    # Normalize D
    if 'D' in df.columns:
        df['D'] = df['D'].astype(str).str.strip().str.upper().str[0]

    # Targets present in USA_ORC
    usa_targets = [c for c in TARGET_COLS if c in df.columns]
    if not usa_targets:
        raise ValueError("No R{TWA}{TWS} target columns found in USA_ORC.csv")

    # Features
    num_cols = [c for c in FEATURE_COLS if c != 'D']
    medians = {c: pd.to_numeric(df[c], errors='coerce').median() if c in df.columns else 0.0 for c in num_cols}

    # Build X
    X_num = pd.DataFrame({c: pd.to_numeric(df[c], errors='coerce').fillna(medians[c]) for c in num_cols}, index=df.index)
    D_base = pd.Series([''] * len(df), index=df.index)
    D_ser = df.get('D', D_base).astype(str).str.strip().str.upper().str[0]
    X_cat = pd.DataFrame({f'D_{lvl}': (D_ser == lvl).astype(int) for lvl in CAT_LEVELS}, index=df.index)
    X = pd.concat([X_cat, X_num], axis=1)

    # Y
    Y = df[usa_targets].apply(pd.to_numeric, errors='coerce')

    # Drop rows with any missing Y
    valid = Y.notna().all(axis=1)
    X = X.loc[valid]
    Y = Y.loc[valid]

    rf = RandomForestRegressor(n_estimators=800, max_depth=18, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X, Y)

    rf_model = rf


def row_to_features(row: pd.Series) -> pd.DataFrame:
    """Map a row with FEATURE_COLS into the trained feature space (one-hot D + numeric with medians)."""
    # Numeric
    x_num = {c: pd.to_numeric(row[c], errors='coerce') if c in row.index else np.nan for c in num_cols}
    for c in num_cols:
        if pd.isna(x_num[c]):
            x_num[c] = medians[c]
    # Categorical D
    dval = str(row.get('D', '')).strip().upper()[:1]
    x_cat = {f'D_{lvl}': 1 if dval == lvl else 0 for lvl in CAT_LEVELS}
    # Combine in fixed order
    X_one = pd.DataFrame([{**x_cat, **x_num}], columns=[f'D_{lvl}' for lvl in CAT_LEVELS] + num_cols)
    return X_one

def predict_knots_for_row(row: pd.Series) -> Tuple[pd.DataFrame, List[List[Optional[float]]]]:
    if rf_model is None:
        raise RuntimeError("Model not trained")
    X_one = row_to_features(row)
    pred_vec = rf_model.predict(X_one)[0]
    # Series of s/Nm aligned to usa_targets
    pred_series = pd.Series(pred_vec, index=usa_targets)
    # Fill missing targets
    for c in TARGET_COLS:
        if c not in pred_series.index:
            pred_series.loc[c] = np.nan
    # Build s/Nm table
    snm = pd.DataFrame(index=TWAs, columns=TWSs, dtype=float)
    for a in TWAs:
        for w in TWSs:
            name = f"R{a}{w}"
            snm.loc[a, w] = pred_series.get(name, np.nan)
    # Convert to knots
    knots = 3600.0 / snm
    knots.replace([np.inf, -np.inf], np.nan, inplace=True)
    # List of lists for output (rows=TWA, cols=TWS)
    pred_list = [[float(knots.loc[a, w]) if pd.notna(knots.loc[a, w]) else None for w in TWSs] for a in TWAs]
    return knots, pred_list


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', twas=TWAs, twss=TWSs, feature_cols=FEATURE_COLS, feature_desc=FEATURE_DESCRIPTIONS)


@app.route('/template.csv', methods=['GET'])
def download_template():
    # Build template from ESP_ORC.csv using the boat named HISTOLAB (case-insensitive)
    df = load_csv_robust(ESP_PATH)
    # Build template with FEATURE_COLS only
    tpl = pd.DataFrame(columns=FEATURE_COLS)
    if not df.empty:
        # Try to find a name-like column and pick the row where name == HISTOLAB
        name_cols = [c for c in df.columns if any(k in str(c).lower() for k in ['name','boat','yacht'])]
        sel = df
        if name_cols:
            mask = pd.Series(False, index=df.index)
            for c in name_cols:
                mask = mask | (df[c].astype(str).str.strip().str.lower() == 'histolab')
            if mask.any():
                sel = df[mask]
        row = sel.iloc[0]
        example = {}
        for col in FEATURE_COLS:
            if col == 'D':
                val = row.get('D', 'C')
                example[col] = str(val).strip().upper()[:1]
            else:
                example[col] = pd.to_numeric(row.get(col, np.nan), errors='coerce')
        tpl = pd.DataFrame([example])
    buf = io.StringIO()
    tpl.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='template.csv')


@app.route('/predict', methods=['POST'])
def predict():
    if rf_model is None:
        # Train lazily on first predict
        train_model()
    global last_export_df

    # Case 1: CSV upload
    file = request.files.get('csv_file')
    if file and file.filename:
        try:
            df_in = pd.read_csv(file)
        except Exception as e:
            flash(f"Failed to read CSV: {e}")
            return redirect(url_for('index'))
        # Ensure all needed columns exist; fill missing numeric with NaN to impute
        for col in FEATURE_COLS:
            if col not in df_in.columns:
                df_in[col] = np.nan
        # If exactly one row, show the prediction table on the page
        if len(df_in) == 1:
            row = df_in.iloc[0]
            knots_table, _pred_list = predict_knots_for_row(row)
            # Build export frame with inputs + K columns
            out = pd.DataFrame([row.to_dict()])
            for a in TWAs:
                for w in TWSs:
                    out.loc[out.index[0], f"K{a}_{w}"] = float(knots_table.loc[a, w]) if pd.notna(knots_table.loc[a, w]) else np.nan
            last_export_df = out
            return render_template(
                'index.html',
                twas=TWAs, twss=TWSs, feature_cols=FEATURE_COLS, feature_desc=FEATURE_DESCRIPTIONS,
                knots_table=knots_table.round(2).values.tolist(),
                show_results=True,
                from_csv=True
            )
        # Otherwise predict for each row and return a CSV download
        all_knots_tables: List[pd.DataFrame] = []
        for _, row in df_in.iterrows():
            knots_table, _pred_list = predict_knots_for_row(row)
            all_knots_tables.append(knots_table)
        out = df_in.copy()
        for i, knots_table in enumerate(all_knots_tables):
            for a in TWAs:
                for w in TWSs:
                    out.loc[out.index[i], f"K{a}_{w}"] = float(knots_table.loc[a, w]) if pd.notna(knots_table.loc[a, w]) else np.nan
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(io.BytesIO(buf.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='predictions_knots.csv')

    # Case 2: manual form
    row_dict = {}
    for col in FEATURE_COLS:
        row_dict[col] = request.form.get(col, '').strip()
    # Basic validation
    if not row_dict.get('D'):
        flash("Please provide D (C/S/R) or upload a CSV.")
        return redirect(url_for('index'))
    # Convert to types
    row = {}
    row['D'] = str(row_dict['D']).strip().upper()[:1]
    for col in FEATURE_COLS:
        if col == 'D':
            continue
        try:
            row[col] = float(row_dict.get(col, '') or 'nan')
        except Exception:
            row[col] = np.nan
    row_series = pd.Series(row)

    knots_table, _pred_list = predict_knots_for_row(row_series)
    # Prepare export df for manual prediction as well
    out = pd.DataFrame([row_series.to_dict()])
    for a in TWAs:
        for w in TWSs:
            out.loc[out.index[0], f"K{a}_{w}"] = float(knots_table.loc[a, w]) if pd.notna(knots_table.loc[a, w]) else np.nan
    last_export_df = out

    # Render results page with table and list
    return render_template(
        'index.html',
        twas=TWAs, twss=TWSs, feature_cols=FEATURE_COLS, feature_desc=FEATURE_DESCRIPTIONS,
        knots_table=knots_table.round(2).values.tolist(),
        show_results=True,
        from_csv=False
    )


@app.route('/export.csv', methods=['GET'])
def export_csv():
    global last_export_df
    if last_export_df is None or last_export_df.empty:
        flash('No prediction available to export. Please run a prediction first.')
        return redirect(url_for('index'))
    buf = io.StringIO()
    last_export_df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='export.csv')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_trained': rf_model is not None,
        'usa_targets': usa_targets,
    })


if __name__ == '__main__':
    # Train upfront so first request is fast
    try:
        train_model()
    except Exception as e:
        # Defer training to first predict if something goes wrong now
        print(f"Warning: initial training failed: {e}")
    app.run(host='127.0.0.1', port=5000, debug=True)
