import io
import base64
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lasio
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)

@app.after_request
def apply_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'las'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_column(df, keywords):
    for key in keywords:
        for col in df.columns:
            if key.lower() in str(col).lower():
                return col
    return None

def build_lithology_model(df, feature_cols, lith_col):
    df_lith = df.dropna(subset=feature_cols + [lith_col])
    if df_lith[lith_col].nunique() < 2 or len(df_lith) < 50:
        return None, None
    X = df_lith[feature_cols].values
    y = df_lith[lith_col].astype(str).values
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X, y)
    return model, list(model.classes_)

def apply_lithology_model(df, feature_cols, model):
    if model is None:
        return ['Unknown'] * len(df), [0.0] * len(df)
    df_feat = df[feature_cols]
    mask = df_feat.notna().all(axis=1)
    preds = np.array(['Unknown'] * len(df), dtype=object)
    conf = np.zeros(len(df))
    if mask.sum() == 0:
        return preds.tolist(), conf.tolist()
    X = df_feat[mask].values
    proba = model.predict_proba(X)
    pred_labels = model.predict(X)
    preds[mask.values] = pred_labels
    conf[mask.values] = proba.max(axis=1)
    return preds.tolist(), conf.tolist()

def fill_missing_logs(df, log_cols):
    info = {}
    for col in log_cols:
        if col not in df.columns or df[col].isna().sum() == 0:
            continue
        other_cols = [c for c in log_cols if c != col and c in df.columns]
        if not other_cols:
            continue
        df_train = df.dropna(subset=other_cols + [col])
        if len(df_train) < 30:
            continue
        X_train, y_train = df_train[other_cols].values, df_train[col].values
        model = RandomForestRegressor(n_estimators=80, random_state=42)
        model.fit(X_train, y_train)
        mask_pred = df[col].isna() & df[other_cols].notna().all(axis=1)
        if mask_pred.sum() == 0:
            continue
        df.loc[mask_pred, col] = model.predict(df.loc[mask_pred, other_cols].values)
        info[col] = int(mask_pred.sum())
    return df, info

def classify_pay_zone(df, lith_pred, lith_conf, res_col):
    pay_flag = np.array([False] * len(df))
    pay_col = detect_column(df, ['PAY', 'PAY_FLAG', 'NETPAY', 'PAYZONE'])
    if pay_col and df[pay_col].dropna().nunique() >= 2:
        vals = df[pay_col].astype(str).str.upper().str.strip()
        pay_flag = vals.isin(['1', 'Y', 'YES', 'TRUE', 'PAY', 'P'])
        return pay_flag, 'from_label'
    high_conf = np.array(lith_conf) >= 0.6
    lith_array = np.char.lower(np.array(lith_pred, dtype=str))
    oil_like = np.isin(lith_array, ['sandstone', 'sand', 'dolomite', 'limestone'])
    pay_flag = oil_like & high_conf
    if res_col and res_col in df.columns:
        try:
            res_vals = pd.to_numeric(df[res_col], errors='coerce')
            pay_flag &= (res_vals > 10)
        except Exception:
            pass
    return pay_flag, 'rule_based'

def compute_net_pay(df, depth_col, pay_flag):
    try:
        depth = pd.to_numeric(df[depth_col], errors='coerce')
        mask = (~depth.isna()) & pay_flag
        if mask.sum() < 2:
            return 0.0
        dstep = depth.sort_values().diff().median()
        if pd.isna(dstep) or dstep <= 0:
            dstep = depth.diff().median()
        return float(dstep * mask.sum())
    except Exception:
        return 0.0

def estimate_porosity(rhob_series):
    rho_ma, rho_f = 2.65, 1.0
    rhob = pd.to_numeric(rhob_series, errors='coerce')
    phi = (rho_ma - rhob) / (rho_ma - rho_f)
    return phi.clip(0, 0.35)

def make_dynamic_log_plot(df, depth_col, lith_pred):
    """ Dynamically creates tracks based on columns present """
    track_cols = []
    possible = {
        "GR": ["GR", "Gamma"],
        "RHOB": ["RHOB", "Density"],
        "NPHI": ["NPHI", "Neutron"],
        "RES": ["RESD", "RT", "RES"],
        "DT": ["DT", "Sonic"],
        "PEF": ["PEF", "Photoelectric"]
    }

    for name, keys in possible.items():
        col = detect_column(df, keys)
        if col: track_cols.append((name, col))

    n = len(track_cols) + 1  # +1 for lithology
    fig = make_subplots(rows=1, cols=n, shared_yaxes=True, horizontal_spacing=0.05)

    depth = df[depth_col]
    for i, (name, col) in enumerate(track_cols, start=1):
        fig.add_trace(go.Scatter(x=df[col], y=depth, mode='lines', name=name), 1, i)
        fig.update_xaxes(title_text=name, row=1, col=i)

    color_map = {'sandstone': 'gold', 'shale': 'gray', 'limestone': 'lightblue',
                 'dolomite': 'orange', 'tight': 'brown', 'unknown': 'white'}
    colors = [color_map.get(str(l).lower(), 'white') for l in lith_pred]
    fig.add_trace(go.Scatter(x=[1]*len(depth), y=depth, mode='markers',
                             marker=dict(color=colors, size=5), name='Lithology'), 1, n)
    fig.update_xaxes(visible=False, row=1, col=n)
    fig.update_yaxes(title_text="Depth", autorange="reversed")
    fig.update_layout(template="plotly_dark", showlegend=False, height=800,
                      title="Dynamic Well Log Tracks")
    return fig

@app.route('/analyze_welllog', methods=['POST'])
def analyze_welllog():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        ext = file.filename.split('.')[-1].lower()
        if ext == 'csv':
            df = pd.read_csv(file)
        elif ext in ['xlsx', 'xls']:
            df = pd.read_excel(file)
        elif ext == 'las':
            las = lasio.read(file)
            df = las.df().reset_index()
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        depth_col = detect_column(df, ['DEPTH', 'MD'])
        gr_col = detect_column(df, ['GR'])
        rhob_col = detect_column(df, ['RHOB'])
        nphi_col = detect_column(df, ['NPHI'])
        res_col = detect_column(df, ['RESD', 'RT', 'RES'])
        lith_col = detect_column(df, ['LITH', 'LITHOLOGY', 'FACIES'])

        df['PHI_AI'] = estimate_porosity(df[rhob_col]) if rhob_col else 0
        lith_pred = ['Unknown'] * len(df)
        lith_conf = [0.0] * len(df)

        if lith_col:
            model, _ = build_lithology_model(df, [gr_col, rhob_col, nphi_col, res_col], lith_col)
            if model:
                lith_pred, lith_conf = apply_lithology_model(df, [gr_col, rhob_col, nphi_col, res_col], model)

        pay_flag, pay_source = classify_pay_zone(df, lith_pred, lith_conf, res_col)
        net_pay = compute_net_pay(df, depth_col, pay_flag)
        avg_phi_pay = float(pd.Series(df['PHI_AI'])[pay_flag].mean()) if pay_flag.any() else 0.0
        lith_counts = pd.Series(lith_pred).value_counts().to_dict()

        summary = f"AI Summary\n\nLithology:\n" + "\n".join([f"{k}: {v}" for k,v in lith_counts.items()]) + \
                  f"\n\nNetPay≈{net_pay:.2f}\nAvg φ={avg_phi_pay:.3f}"

        pdf_b64 = base64.b64encode(io.BytesIO().getbuffer()).decode()

        main_fig = make_dynamic_log_plot(df, depth_col, lith_pred)

        return jsonify({
            "summary": summary,
            "plots": {"main_logs": main_fig.to_json()},
            "pdf_report": {"data": "data:application/pdf;base64," + pdf_b64}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
