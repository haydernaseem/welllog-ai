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

# ======================================================
# Flask + CORS
# ======================================================
app = Flask(__name__)
CORS(app)  # تفعيل CORS مبدئياً

# إضافة ترويسات CORS دائماً (مهم لـ Render + Firebase)
@app.after_request
def apply_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "las"}


# ======================================================
# Utilities
# ======================================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_column(df, keywords):
    """البحث عن عمود يحتوي اسم قريب من الكلمات المفتاحية."""
    for key in keywords:
        for col in df.columns:
            if key.lower() in str(col).lower():
                return col
    return None


def build_lithology_model(df, feature_cols, lith_col):
    """بناء نموذج RandomForest للـ Lithology إذا توجد Labels كافية."""
    df_lith = df.dropna(subset=feature_cols + [lith_col])
    if df_lith[lith_col].nunique() < 2 or len(df_lith) < 50:
        return None, None

    X = df_lith[feature_cols].values
    y = df_lith[lith_col].astype(str).values

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X, y)
    return model, list(model.classes_)


def apply_lithology_model(df, feature_cols, model):
    """تطبيق نموذج Lithology على كامل البيانات."""
    if model is None:
        return ["Unknown"] * len(df), [0.0] * len(df)

    df_feat = df[feature_cols]
    mask = df_feat.notna().all(axis=1)

    preds = np.array(["Unknown"] * len(df), dtype=object)
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
    """استخدام RandomForestRegressor لتنبؤ المجسات الناقصة."""
    info = {}
    for col in log_cols:
        if not col or col not in df.columns:
            continue
        if df[col].isna().sum() == 0:
            continue

        other_cols = [c for c in log_cols if c and c != col and c in df.columns]
        if not other_cols:
            continue

        df_train = df.dropna(subset=other_cols + [col])
        if len(df_train) < 30:
            continue

        X_train = df_train[other_cols].values
        y_train = df_train[col].values

        model = RandomForestRegressor(n_estimators=80, random_state=42)
        model.fit(X_train, y_train)

        mask_pred = df[col].isna() & df[other_cols].notna().all(axis=1)
        if mask_pred.sum() == 0:
            continue

        df.loc[mask_pred, col] = model.predict(df.loc[mask_pred, other_cols].values)
        info[col] = int(mask_pred.sum())

    return df, info


def classify_pay_zone(df, lith_pred, lith_conf, res_col):
    """
    AI Pay-zone:
    - إذا يوجد عمود PAY حقيقي → من الداتا.
    - إذا لا → Rule-based: lithology + resistivity + confidence.
    """
    pay_flag = np.array([False] * len(df))

    pay_col = detect_column(df, ["PAY", "PAY_FLAG", "NETPAY", "PAYZONE", "PAY_ZONE"])
    if pay_col and df[pay_col].dropna().nunique() >= 2:
        vals = df[pay_col].astype(str).str.upper().str.strip()
        pay_flag = vals.isin(["1", "Y", "YES", "TRUE", "PAY", "P"])
        return pay_flag, "from_label"

    high_conf = np.array(lith_conf) >= 0.6
    lith_array = np.char.lower(np.array(lith_pred, dtype=str))
    oil_like = np.isin(
        lith_array,
        ["sandstone", "sand", "dolomite", "limestone"],
    )
    pay_flag = oil_like & high_conf

    if res_col and res_col in df.columns:
        try:
            res_vals = pd.to_numeric(df[res_col], errors="coerce")
            pay_flag &= res_vals > 10
        except Exception:
            pass

    return pay_flag, "rule_based"


def compute_net_pay(df, depth_col, pay_flag):
    """حساب Net Pay بشكل تقريبي من عدد العينات و step العمق."""
    try:
        depth = pd.to_numeric(df[depth_col], errors="coerce")
        mask = (~depth.isna()) & pay_flag
        depth_pay = depth[mask]

        if len(depth_pay) < 2:
            return 0.0

        dstep = depth.sort_values().diff().median()
        if pd.isna(dstep) or dstep <= 0:
            dstep = depth_pay.diff().median()
        if pd.isna(dstep) or dstep <= 0:
            return float(depth_pay.max() - depth_pay.min())

        net_pay = dstep * mask.sum()
        return float(net_pay)
    except Exception:
        return 0.0


def estimate_porosity(rhob_series):
    """
    Φ ≈ (ρ_ma - ρ_b) / (ρ_ma - ρ_f)
    نفترض matrix density = 2.65 g/cc, fluid = 1.0 g/cc
    """
    rho_ma = 2.65
    rho_f = 1.0
    rhob = pd.to_numeric(rhob_series, errors="coerce")
    phi = (rho_ma - rhob) / (rho_ma - rho_f)
    phi = phi.clip(lower=0, upper=0.35)
    return phi


# ======================================================
# Plotly Figures (Interactive)
# ======================================================
def make_dynamic_log_plot(df, depth_col, gr_col, rhob_col, nphi_col, res_col, lith_pred):
    """
    رسم Tracks ديناميكي:
    ≤ يعرض فقط الأعمدة الموجودة فعلاً (2,3,4 Logs).
    """
    # نحدد أي من الأعمدة موجود
    tracks = []
    if gr_col:
        tracks.append(("GR", gr_col))
    if rhob_col:
        tracks.append(("RHOB", rhob_col))
    if nphi_col:
        tracks.append(("NPHI", nphi_col))
    if res_col:
        tracks.append(("RES", res_col))

    n_cols = len(tracks) + 1  # +1 لـ Lithology Track
    if n_cols < 2:
        n_cols = 2  # لو كان عندنا بس عمود واحد

    fig = make_subplots(
        rows=1,
        cols=n_cols,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=[name for name, _ in tracks] + ["Lithology"],
    )

    depth = df[depth_col]

    # رسم كل Track حسب نوعه
    for i, (name, col) in enumerate(tracks, start=1):
        color = "#22c55e"  # default
        if name == "GR":
            color = "#22c55e"
        elif name == "RHOB":
            color = "#f97316"
        elif name == "NPHI":
            color = "#06b6d4"
        elif name == "RES":
            color = "#ef4444"

        fig.add_trace(
            go.Scatter(
                x=df[col],
                y=depth,
                mode="lines",
                line=dict(color=color, width=1.8),
                name=name,
            ),
            row=1,
            col=i,
        )
        # Axis title
        fig.update_xaxes(title_text=name, row=1, col=i)
        if name == "RES":
            fig.update_xaxes(type="log", row=1, col=i)

    # Lithology Track
    color_map = {
        "sandstone": "#facc15",
        "sand": "#facc15",
        "shale": "#6b7280",
        "limestone": "#93c5fd",
        "dolomite": "#fdba74",
        "tight": "#92400e",
        "unknown": "#ffffff",
    }
    lith_lower = [str(l).lower() for l in lith_pred]
    color_list = [color_map.get(l, "#ffffff") for l in lith_lower]

    fig.add_trace(
        go.Scatter(
            x=[1] * len(depth),
            y=depth,
            mode="markers",
            marker=dict(color=color_list, size=5),
            name="Lithology",
        ),
        row=1,
        col=n_cols,
    )
    fig.update_xaxes(visible=False, row=1, col=n_cols)

    fig.update_yaxes(title_text="Depth", autorange="reversed")
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        height=800,
        title="OILNOVA – Dynamic Well Log Tracks",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_crossplots(df, gr_col, res_col, rhob_col, nphi_col, lith_pred):
    """إنشاء Crossplots: GR-RES و RHOB-NPHI."""
    figs = []

    # GR vs RES
    if gr_col and res_col and gr_col in df.columns and res_col in df.columns:
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=df[gr_col],
                y=df[res_col],
                mode="markers",
                marker=dict(
                    color=df[gr_col],
                    colorscale="Viridis",
                    size=5,
                    showscale=True,
                ),
                name="GR-RES",
            )
        )
        fig1.update_xaxes(title_text="Gamma Ray")
        fig1.update_yaxes(title_text="Resistivity", type="log")
        fig1.update_layout(
            template="plotly_dark",
            title="GR vs Resistivity",
            height=500,
        )
        figs.append(fig1)

    # RHOB vs NPHI – colored by lithology
    if rhob_col and nphi_col and rhob_col in df.columns and nphi_col in df.columns:
        color_map = {
            "sandstone": "#facc15",
            "sand": "#facc15",
            "shale": "#6b7280",
            "limestone": "#93c5fd",
            "dolomite": "#fdba74",
            "tight": "#92400e",
            "unknown": "#ffffff",
        }
        colors = [color_map.get(str(l).lower(), "#ffffff") for l in lith_pred]

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=df[rhob_col],
                y=df[nphi_col],
                mode="markers",
                marker=dict(size=6, color=colors),
                text=lith_pred,
                name="RHOB-NPHI",
            )
        )
        fig2.update_xaxes(title_text="RHOB (g/cc)")
        fig2.update_yaxes(title_text="NPHI")
        fig2.update_layout(
            template="plotly_dark",
            title="RHOB vs NPHI (AI Lithology Coloring)",
            height=500,
        )
        figs.append(fig2)

    return figs


def make_3d_cluster(df, gr_col, rhob_col, nphi_col):
    """3D KMeans Clustering على GR-RHOB-NPHI."""
    try:
        cols = [c for c in [gr_col, rhob_col, nphi_col] if c and c in df.columns]
        if len(cols) < 3:
            return None

        sub = df[cols].dropna()
        if len(sub) < 50:
            return None

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sub.values)

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=sub[cols[0]],
                    y=sub[cols[1]],
                    z=sub[cols[2]],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=labels,
                        colorscale="Viridis",
                        opacity=0.8,
                    ),
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            title="3D AI Clusters (GR – RHOB – NPHI)",
            scene=dict(
                xaxis_title=cols[0],
                yaxis_title=cols[1],
                zaxis_title=cols[2],
            ),
            height=600,
        )
        return fig
    except Exception:
        return None


def generate_pdf_report(summary_text):
    """
    توليد تقرير PDF يحتوي على ملخص التحليل.
    يرجع Base64 string.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 50, "OILNOVA Well Log AI – Report")
    c.setFont("Helvetica", 9)
    c.drawString(40, height - 65, "Powered by ChatGPT AI")

    textobject = c.beginText(40, height - 90)
    textobject.setFont("Helvetica", 10)
    for line in summary_text.splitlines():
        textobject.textLine(line)

    c.drawText(textobject)
    c.showPage()
    c.save()
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


# ======================================================
# Main Endpoint
# ======================================================
@app.route("/analyze_welllog", methods=["POST"])
def analyze_welllog():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Use csv/xlsx/xls/las"}), 400

        # قراءة الملف
        ext = file.filename.rsplit(".", 1)[1].lower()
        if ext == "csv":
            df = pd.read_csv(file)
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(file)
        elif ext == "las":
            las = lasio.read(file)
            df = las.df()
            df.reset_index(inplace=True)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400

        # اكتشاف الأعمدة
        depth_col = detect_column(df, ["DEPTH", "Depth", "MD", "Measured Depth"])
        gr_col = detect_column(df, ["GR", "Gamma", "Gamma Ray"])
        rhob_col = detect_column(df, ["RHOB", "Density", "Bulk Density"])
        nphi_col = detect_column(df, ["NPHI", "Neutron", "Neutron Porosity"])
        res_col = detect_column(df, ["RESD", "RT", "Resistivity", "LLD"])
        lith_label_col = detect_column(df, ["LITH", "LITHO", "LITHOLOGY", "FACIES"])

        if not depth_col:
            return jsonify({"error": "Depth column not found (DEPTH/MD)"}), 400

        # تحويل أعمدة رقمية
        for col in [depth_col, gr_col, rhob_col, nphi_col, res_col]:
            if col and col in df.columns and df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ML لتعبئة القيم المفقودة
        log_cols = [c for c in [gr_col, rhob_col, nphi_col, res_col] if c]
        df, fill_info = fill_missing_logs(df, log_cols)

        # تقدير المسامية
        if rhob_col:
            df["PHI_AI"] = estimate_porosity(df[rhob_col])
        else:
            df["PHI_AI"] = np.nan

        feature_cols = [c for c in [gr_col, rhob_col, nphi_col, res_col, "PHI_AI"] if c]
        lith_model, lith_classes = (None, None)
        lith_pred = ["Unknown"] * len(df)
        lith_conf = [0.0] * len(df)

        if lith_label_col:
            lith_model, lith_classes = build_lithology_model(df, feature_cols, lith_label_col)
            if lith_model:
                lith_pred, lith_conf = apply_lithology_model(df, feature_cols, lith_model)

        # Fallback rule-based lithology
        if lith_model is None:
            for i, row in df.iterrows():
                gr = row[gr_col] if gr_col else np.nan
                rhob = row[rhob_col] if rhob_col else np.nan
                nphi = row[nphi_col] if nphi_col else np.nan
                lit = "Unknown"
                if pd.notna(gr) and pd.notna(rhob) and pd.notna(nphi):
                    if gr < 75 and nphi > 0.25 and rhob < 2.45:
                        lit = "Sandstone"
                    elif gr > 120:
                        lit = "Shale"
                    elif rhob > 2.7:
                        lit = "Limestone"
                    else:
                        lit = "Tight"
                lith_pred[i] = lit
                lith_conf[i] = 0.6

        # Pay-zone AI
        pay_flag, pay_source = classify_pay_zone(df, lith_pred, lith_conf, res_col)
        net_pay = compute_net_pay(df, depth_col, pay_flag)

        lith_series = pd.Series(lith_pred)
        lith_counts = lith_series.value_counts().to_dict()

        avg_phi_pay = (
            float(pd.Series(df["PHI_AI"])[pay_flag].mean()) if pay_flag.any() else 0.0
        )

        # Summary نصي
        summary_lines = [
            "OILNOVA Well Log AI – Smart Interpretation Summary",
            "",
            f"Total samples: {len(df)}",
            "",
            "Lithology distribution:",
        ]
        for k, v in lith_counts.items():
            summary_lines.append(f"  - {k}: {v}")

        summary_lines.append("")
        summary_lines.append(f"Net Pay (approx): {net_pay:.2f} depth units")
        summary_lines.append(f"Pay detection source: {pay_source}")

        if fill_info:
            summary_lines.append("")
            summary_lines.append("Missing logs predicted via ML:")
            for col, cnt in fill_info.items():
                summary_lines.append(f"  - {col}: {cnt} points")

        summary_lines.append("")
        summary_lines.append(
            f"Average AI porosity in pay zones: {avg_phi_pay:.3f}"
        )

        summary_text = "\n".join(summary_lines)
        pdf_b64 = generate_pdf_report(summary_text)

        # Create plots
        main_fig = make_dynamic_log_plot(df, depth_col, gr_col, rhob_col, nphi_col, res_col, lith_pred)
        cross_figs = make_crossplots(df, gr_col, res_col, rhob_col, nphi_col, lith_pred)
        cluster_fig = make_3d_cluster(df, gr_col, rhob_col, nphi_col)

        plots = {"main_logs": main_fig.to_json()}
        if cross_figs:
            if len(cross_figs) > 0:
                plots["crossplot1"] = cross_figs[0].to_json()
            if len(cross_figs) > 1:
                plots["crossplot2"] = cross_figs[1].to_json()
        if cluster_fig:
            plots["cluster3d"] = cluster_fig.to_json()

        return jsonify(
            {
                "lithology_counts": lith_counts,
                "net_pay": net_pay,
                "pay_source": pay_source,
                "fill_info": fill_info,
                "avg_phi_pay": avg_phi_pay,
                "ai_summary": summary_text,
                "plots": plots,
                "pdf_report": {
                    "filename": "OILNOVA_WellLog_Report.pdf",
                    "data": "data:application/pdf;base64," + pdf_b64,
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # للـ local فقط – على Render استخدم gunicorn
    app.run(host="0.0.0.0", port=5001, debug=False)
