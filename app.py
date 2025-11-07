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
CORS(app)  # نفعّل CORS افتراضياً

# ✅ إضافة ترويسات CORS يدوياً بعد كل رد (تعمل دائماً حتى على Render)
@app.after_request
def apply_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
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

def make_log_plot(df, depth_col, gr_col, rhob_col, nphi_col, res_col, lith_pred):
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=("Gamma Ray", "Density & NPHI", "Resistivity", "Lithology"))
    depth = df[depth_col]
    fig.add_trace(go.Scatter(x=df[gr_col], y=depth, mode='lines', line=dict(color='lime', width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=df[rhob_col], y=depth, mode='lines', line=dict(color='orange')), 1, 2)
    fig.add_trace(go.Scatter(x=df[nphi_col], y=depth, mode='lines', line=dict(color='cyan')), 1, 2)
    if res_col and res_col in df.columns:
        fig.add_trace(go.Scatter(x=df[res_col], y=depth, mode='lines', line=dict(color='red')), 1, 3)
        fig.update_xaxes(title_text="Res", type='log', row=1, col=3)
    color_map = {'sandstone': '#facc15', 'shale': '#6b7280', 'limestone': '#93c5fd',
                 'dolomite': '#fdba74', 'tight': '#92400e', 'unknown': '#ffffff'}
    color_list = [color_map.get(str(l).lower(), '#ffffff') for l in lith_pred]
    fig.add_trace(go.Scatter(x=[1]*len(depth), y=depth, mode='markers',
                             marker=dict(color=color_list, size=6)), 1, 4)
    fig.update_yaxes(title_text="Depth", autorange="reversed")
    fig.update_layout(template="plotly_dark", showlegend=False, height=800,
                      title="OILNOVA Well Log AI – Tracks View")
    return fig

def generate_pdf_report(summary_text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 50, "OILNOVA Well Log AI – Report")
    c.setFont("Helvetica", 9)
    c.drawString(40, height - 65, "Powered by ChatGPT AI")
    text = c.beginText(40, height - 90)
    text.setFont("Helvetica", 10)
    for line in summary_text.splitlines():
        text.textLine(line)
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

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

        if df.empty:
            return jsonify({'error': 'Empty file'}), 400

        depth_col = detect_column(df, ['DEPTH', 'MD'])
        gr_col = detect_column(df, ['GR'])
        rhob_col = detect_column(df, ['RHOB'])
        nphi_col = detect_column(df, ['NPHI'])
        res_col = detect_column(df, ['RESD', 'RT', 'RES'])
        lith_col = detect_column(df, ['LITH', 'LITHOLOGY', 'FACIES'])
        if not all([depth_col, gr_col, rhob_col, nphi_col]):
            return jsonify({'error': 'Missing essential logs'}), 400

        df, fill_info = fill_missing_logs(df, [gr_col, rhob_col, nphi_col, res_col])
        df['PHI_AI'] = estimate_porosity(df[rhob_col])

        feature_cols = [c for c in [gr_col, rhob_col, nphi_col, res_col, 'PHI_AI'] if c]
        lith_model = None
        lith_pred, lith_conf = ['Unknown'] * len(df), [0.0] * len(df)

        if lith_col:
            lith_model, _ = build_lithology_model(df, feature_cols, lith_col)
            if lith_model:
                lith_pred, lith_conf = apply_lithology_model(df, feature_cols, lith_model)

        if lith_model is None:
            for i, row in df.iterrows():
                gr, rhob, nphi = row[gr_col], row[rhob_col], row[nphi_col]
                if pd.notna(gr) and pd.notna(rhob) and pd.notna(nphi):
                    if gr < 75 and nphi > 0.25 and rhob < 2.45:
                        lith_pred[i] = 'Sandstone'
                    elif gr > 120:
                        lith_pred[i] = 'Shale'
                    elif rhob > 2.7:
                        lith_pred[i] = 'Limestone'
                    else:
                        lith_pred[i] = 'Tight'
                    lith_conf[i] = 0.6

        pay_flag, pay_source = classify_pay_zone(df, lith_pred, lith_conf, res_col)
        net_pay = compute_net_pay(df, depth_col, pay_flag)
        avg_phi_pay = float(pd.Series(df['PHI_AI'])[pay_flag].mean()) if pay_flag.any() else 0.0
        lith_counts = pd.Series(lith_pred).value_counts().to_dict()

        summary = f"OILNOVA Well Log AI – Summary\n\nSamples: {len(df)}\n\n" \
                  f"Lithology distribution:\n" + \
                  "\n".join(f"  - {k}: {v}" for k, v in lith_counts.items()) + \
                  f"\n\nNet pay: {net_pay:.2f}\nPay source: {pay_source}\n" \
                  f"Missing logs: {fill_info}\nAvg porosity in pay: {avg_phi_pay:.3f}"
        pdf_b64 = generate_pdf_report(summary)

        return jsonify({
            "lithology_counts": lith_counts,
            "net_pay": net_pay,
            "pay_source": pay_source,
            "fill_info": fill_info,
            "avg_phi_pay": avg_phi_pay,
            "pdf_report": {
                "filename": "OILNOVA_WellLog_Report.pdf",
                "data": "data:application/pdf;base64," + pdf_b64
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
