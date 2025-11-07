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

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'las'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_column(df, keywords):
    """البحث عن عمود يحتوي على أي من الكلمات المفتاحية."""
    for key in keywords:
        for col in df.columns:
            if key.lower() in str(col).lower():
                return col
    return None


def build_lithology_model(df, feature_cols, lith_col):
    """بناء نموذج RandomForest للـ Lithology إذا توفرت Labels حقيقية في الداتا."""
    df_lith = df.dropna(subset=feature_cols + [lith_col])
    if df_lith[lith_col].nunique() < 2 or len(df_lith) < 50:
        return None, None

    X = df_lith[feature_cols].values
    y = df_lith[lith_col].astype(str).values

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X, y)
    classes = list(model.classes_)
    return model, classes


def apply_lithology_model(df, feature_cols, model, classes):
    """تطبيق نموذج Lithology على كامل البيانات."""
    if model is None:
        return None, None

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
    """
    استخدام RandomForestRegressor لتنبؤ المجسات الناقصة.
    يرجع df محدث + قاموس بعدد القيم التي تم تعويضها لكل Log.
    """
    info = {}
    for col in log_cols:
        if col not in df.columns:
            continue
        if df[col].isna().sum() == 0:
            continue

        other_cols = [c for c in log_cols if c != col and c in df.columns]
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

        X_pred = df.loc[mask_pred, other_cols].values
        y_pred = model.predict(X_pred)
        df.loc[mask_pred, col] = y_pred

        info[col] = int(mask_pred.sum())

    return df, info


def classify_pay_zone(df, lith_pred, lith_conf, res_col):
    """
    تحديد Pay Zone:
    - إذا يوجد عمود PAY حقيقي → نستخدمه.
    - إذا لا → Rule-based يعتمد على Lithology + Resistivity + الثقة.
    """
    pay_flag = np.array([False] * len(df))

    pay_col = detect_column(df, ['PAY', 'PAY_FLAG', 'NETPAY', 'PAYZONE'])
    if pay_col and df[pay_col].dropna().nunique() >= 2:
        vals = df[pay_col]
        pay_flag = vals.astype(str).str.strip().str.upper().isin(
            ['1', 'Y', 'YES', 'TRUE', 'PAY', 'P']
        )
        return pay_flag, 'from_label'

    high_conf = np.array(lith_conf) >= 0.6
    lith_array = np.array(lith_pred, dtype=str)
    oil_like = np.isin(
        np.char.lower(lith_array),
        ['sandstone', 'sand', 'dolomite', 'limestone']
    )
    pay_flag = oil_like & high_conf

    if res_col and res_col in df.columns:
        try:
            res_vals = pd.to_numeric(df[res_col], errors='coerce')
            pay_flag = pay_flag & (res_vals > 10)
        except Exception:
            pass

    return pay_flag, 'rule_based'


def compute_net_pay(df, depth_col, pay_flag):
    """حساب Net Pay تقريبي من عدد العينات و Step العمق."""
    try:
        depth = pd.to_numeric(df[depth_col], errors='coerce')
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
    تقدير بسيط للـ Porosity من الـ RHOB:
    Φ ≈ (ρ_ma - ρ_b) / (ρ_ma - ρ_f)
    نفترض matrix density = 2.65 g/cc, fluid = 1.0 g/cc
    """
    rho_ma = 2.65
    rho_f = 1.0
    rhob = pd.to_numeric(rhob_series, errors='coerce')
    phi = (rho_ma - rhob) / (rho_ma - rho_f)
    phi = phi.clip(lower=0, upper=0.35)  # قص القيم غير واقعية
    return phi


def make_log_plot(df, depth_col, gr_col, rhob_col, nphi_col, res_col, lith_pred):
    """رسم Tracks: GR, RHOB/NPHI, RES, Lithology."""
    fig = make_subplots(
        rows=1,
        cols=4,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=("Gamma Ray", "Density & NPHI",
                        "Resistivity", "Lithology")
    )

    depth = df[depth_col]

    # GR Track
    fig.add_trace(
        go.Scatter(
            x=df[gr_col],
            y=depth,
            mode='lines',
            line=dict(color='lime', width=2),
            name='GR'
        ),
        row=1,
        col=1
    )
    fig.update_xaxes(title_text="GR", row=1, col=1)

    # Density & NPHI Track
    fig.add_trace(
        go.Scatter(
            x=df[rhob_col],
            y=depth,
            mode='lines',
            line=dict(color='orange'),
            name='RHOB'
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df[nphi_col],
            y=depth,
            mode='lines',
            line=dict(color='cyan'),
            name='NPHI'
        ),
        row=1,
        col=2
    )
    fig.update_xaxes(title_text="RHOB / NPHI", row=1, col=2)

    # Resistivity Track
    if res_col and res_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[res_col],
                y=depth,
                mode='lines',
                line=dict(color='red'),
                name='Resistivity'
            ),
            row=1,
            col=3
        )
        fig.update_xaxes(title_text="Res", type='log', row=1, col=3)

    # Lithology Track
    colors_map = {
        'sandstone': 'yellow',
        'sand': 'yellow',
        'limestone': 'green',
        'dolomite': 'orange',
        'shale': 'gray',
        'tight': 'brown',
        'unknown': 'white'
    }
    lith_lower = [str(l).lower() for l in lith_pred]
    color_list = [colors_map.get(l, 'white') for l in lith_lower]

    fig.add_trace(
        go.Scatter(
            x=[1] * len(depth),
            y=depth,
            mode='markers',
            marker=dict(color=color_list, size=6),
            name='Lithology'
        ),
        row=1,
        col=4
    )
    fig.update_xaxes(visible=False, row=1, col=4)

    fig.update_yaxes(title_text="Depth", autorange="reversed")
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        height=800,
        title="OILNOVA Well Log AI – Tracks View",
        margin=dict(l=40, r=20, t=60, b=40)
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
                mode='markers',
                marker=dict(
                    color=df[gr_col],
                    colorscale='Viridis',
                    size=5
                ),
                name='GR-RES'
            )
        )
        fig1.update_xaxes(title_text="GR")
        fig1.update_yaxes(title_text="RES", type='log')
        fig1.update_layout(
            template="plotly_dark",
            title="GR vs Resistivity"
        )
        figs.append(fig1)

    # RHOB vs NPHI colored by lithology
    if rhob_col and nphi_col and rhob_col in df.columns and nphi_col in df.columns:
        lith_lower = [str(l).lower() for l in lith_pred]
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=df[rhob_col],
                y=df[nphi_col],
                mode='markers',
                marker=dict(
                    size=6,
                    color=lith_lower,
                    colorscale='Turbo'
                ),
                text=lith_pred,
                name='RHOB-NPHI'
            )
        )
        fig2.update_xaxes(title_text="RHOB")
        fig2.update_yaxes(title_text="NPHI")
        fig2.update_layout(
            template="plotly_dark",
            title="RHOB vs NPHI (AI Lithology Coloring)"
        )
        figs.append(fig2)

    return figs


def make_3d_cluster(df, gr_col, rhob_col, nphi_col):
    """3D KMeans Clustering على GR-RHOB-NPHI."""
    try:
        cols = [c for c in [gr_col, rhob_col, nphi_col]
                if c and c in df.columns]
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
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=labels,
                        colorscale='Viridis',
                        opacity=0.8
                    )
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            title="3D AI Clusters (GR–RHOB–NPHI)",
            scene=dict(
                xaxis_title=cols[0],
                yaxis_title=cols[1],
                zaxis_title=cols[2],
            )
        )
        return fig
    except Exception:
        return None


def generate_pdf_report(summary_text):
    """
    توليد تقرير PDF بسيط يحتوي على ملخص ذكي.
    الملف يرجع Base64 حتى الواجهة تقدر تعرض زر Download.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # عنوان
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
    pdf_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    return pdf_b64


@app.route('/analyze_welllog', methods=['POST'])
def analyze_welllog():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload .csv, .xlsx or .las'}), 400

        # قراءة الملف
        ext = file.filename.lower().split('.')[-1]
        if ext == 'csv':
            df = pd.read_csv(file)
        elif ext in ['xlsx', 'xls']:
            df = pd.read_excel(file)
        elif ext == 'las':
            las = lasio.read(file)
            df = las.df()
            df.reset_index(inplace=True)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        if df.empty:
            return jsonify({'error': 'The uploaded file is empty'}), 400

        # اكتشاف الأعمدة
        depth_col = detect_column(
            df, ['DEPTH', 'Depth', 'MD', 'Measured Depth'])
        gr_col = detect_column(df, ['GR', 'Gamma', 'Gamma Ray'])
        rhob_col = detect_column(df, ['RHOB', 'Density', 'Bulk Density'])
        nphi_col = detect_column(df, ['NPHI', 'Neutron', 'Neutron Porosity'])
        res_col = detect_column(df, ['RESD', 'RT', 'Resistivity', 'LLD'])
        lith_label_col = detect_column(
            df, ['LITH', 'LITHO', 'LITHOLOGY', 'FACIES'])

        if not all([depth_col, gr_col, rhob_col, nphi_col]):
            return jsonify({'error': 'Missing essential logs (Depth, GR, RHOB, NPHI)'}), 400

        # تحويل الأعمدة الرقمية
        for col in [depth_col, gr_col, rhob_col, nphi_col, res_col, lith_label_col]:
            if col and col in df.columns and df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # ML لتنبؤ المجسات الناقصة
        log_cols = [c for c in [gr_col, rhob_col, nphi_col, res_col] if c]
        df, fill_info = fill_missing_logs(df, log_cols)

        # تقدير Porosity من RHOB
        porosity = estimate_porosity(df[rhob_col])
        df['PHI_AI'] = porosity

        # نموذج ML للـ Lithology إذا توجد Labels
        feature_cols = [c for c in [gr_col, rhob_col,
                                    nphi_col, res_col, 'PHI_AI'] if c]
        lith_model, lith_classes = (None, None)
        lith_pred = ['Unknown'] * len(df)
        lith_conf = [0.0] * len(df)

        if lith_label_col:
            lith_model, lith_classes = build_lithology_model(
                df, feature_cols, lith_label_col)
            if lith_model:
                lith_pred, lith_conf = apply_lithology_model(
                    df, feature_cols, lith_model, lith_classes)

        # Fallback rule-based إذا لا يوجد نموذج
        if lith_model is None:
            for i, row in df.iterrows():
                gr = row[gr_col]
                rhob = row[rhob_col]
                nphi = row[nphi_col]
                lit = 'Unknown'

                if pd.notna(gr) and pd.notna(rhob) and pd.notna(nphi):
                    if gr < 75 and nphi > 0.25 and rhob < 2.45:
                        lit = 'Sandstone'
                    elif gr > 120:
                        lit = 'Shale'
                    elif rhob > 2.7:
                        lit = 'Limestone'
                    else:
                        lit = 'Tight'

                lith_pred[i] = lit
                lith_conf[i] = 0.6

        # Pay-zone AI
        pay_flag, pay_source = classify_pay_zone(
            df, lith_pred, lith_conf, res_col)
        net_pay = compute_net_pay(df, depth_col, pay_flag)

        # ملخص نصي للتقرير
        lith_series = pd.Series(lith_pred)
        lith_counts = lith_series.value_counts().to_dict()

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
        summary_lines.append(f"Net pay (approx): {net_pay:.2f} depth units")
        summary_lines.append(f"Pay detection source: {pay_source}")

        if fill_info:
            summary_lines.append("")
            summary_lines.append("Missing logs predicted via ML:")
            for col, cnt in fill_info.items():
                summary_lines.append(f"  - {col}: {cnt} points")

        avg_phi_pay = float(pd.Series(df['PHI_AI'])[
                            pay_flag].mean()) if pay_flag.any() else 0.0
        summary_lines.append("")
        summary_lines.append(
            f"Average AI porosity in pay zones: {avg_phi_pay:.3f}")

        summary_text = "\n".join(summary_lines)
        pdf_b64 = generate_pdf_report(summary_text)

        # الرسوم
        main_fig = make_log_plot(
            df, depth_col, gr_col, rhob_col, nphi_col, res_col, lith_pred)
        cross_figs = make_crossplots(
            df, gr_col, res_col, rhob_col, nphi_col, lith_pred)
        cluster_fig = make_3d_cluster(df, gr_col, rhob_col, nphi_col)

        def fig_to_b64(fig):
            img_bytes = fig.to_image(
                format="png", width=900, height=600, scale=2)
            return "data:image/png;base64," + base64.b64encode(img_bytes).decode('utf-8')

        images = {
            'main_logs': fig_to_b64(main_fig)
        }
        if cross_figs:
            if len(cross_figs) > 0:
                images['crossplot1'] = fig_to_b64(cross_figs[0])
            if len(cross_figs) > 1:
                images['crossplot2'] = fig_to_b64(cross_figs[1])
        if cluster_fig:
            images['cluster3d'] = fig_to_b64(cluster_fig)

        return jsonify({
            'lithology_counts': lith_counts,
            'net_pay': net_pay,
            'pay_source': pay_source,
            'fill_info': fill_info,
            'avg_phi_pay': avg_phi_pay,
            'images': images,
            'pdf_report': {
                'filename': 'OILNOVA_WellLog_Report.pdf',
                'data': 'data:application/pdf;base64,' + pdf_b64
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # للـ local فقط؛ على Render سيتم استخدام gunicorn
    app.run(host='0.0.0.0', port=5001, debug=False)
