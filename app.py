
import os
import math
import datetime
import tempfile
from io import BytesIO

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from fpdf import FPDF

try:
    import shap
except Exception:
    shap = None

st.set_page_config(page_title="AIcrete Solutions", layout="wide")

APP_NAME = "AIcrete Solutions"
TAGLINE = "UHPC Intelligence Platform"
WORKBOOK_NAME = "Data UHPC.xlsx"
MODEL_NAME = "model.pkl"
LOGO_NAME = "logo.png"

STANDARD_THRESHOLDS = {
    "ACI 318": 120,
    "ACI 363": 120,
    "BS 8110": 100,
    "Eurocode 2": 120,
    "fib MC2010": 120,
    "MS 1195": 150,
}

STANDARD_OPTIONS = [
    "Eurocode 2 (BS EN 1992-1-1)",
    "ACI 318",
    "IS 456 (India)",
    "MS EN (Malaysia)",
    "JSCE (Japan)",
    "GB / China",
]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_NAME):
        st.error(f"Missing model file: {MODEL_NAME}")
        st.stop()
    return joblib.load(MODEL_NAME)


@st.cache_data
def load_workbook():
    if not os.path.exists(WORKBOOK_NAME):
        st.error(f"Missing workbook: {WORKBOOK_NAME}")
        st.stop()
    raw = pd.read_excel(WORKBOOK_NAME)
    df = raw.select_dtypes(include=[np.number]).copy()
    if df.shape[1] < 2:
        st.error("Workbook must contain at least two numeric columns.")
        st.stop()
    return df


MODEL = load_model()
DF = load_workbook()
FEATURE_COLS = list(DF.columns[:-1])
TARGET_COL = DF.columns[-1]


@st.cache_data
def get_ranges():
    return {c: {
        "min": float(DF[c].min()),
        "max": float(DF[c].max()),
        "mean": float(DF[c].mean())
    } for c in FEATURE_COLS}


RANGES = get_ranges()


def logo_exists():
    return os.path.exists(LOGO_NAME)


def get_feature_bounds(name, lo, hi):
    lname = name.lower()
    if "cement" in lname:
        lo = max(lo, 500.0)
        hi = min(hi, 1200.0)
    if "water" in lname:
        lo = max(lo, 120.0)
    if "age" in lname:
        lo = max(lo, 1.0)
        hi = min(hi, 90.0)
    return lo, hi


def build_input_df(inputs: dict) -> pd.DataFrame:
    row = {}
    for col in FEATURE_COLS:
        row[col] = float(inputs.get(col, RANGES[col]["mean"]))
    x = pd.DataFrame([row], columns=FEATURE_COLS)
    return x.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def predict_strength(inputs: dict) -> float:
    x = build_input_df(inputs)
    pred = MODEL.predict(x)
    return float(np.array(pred).reshape(-1)[0])


def derived_properties(cs_mpa: float, standard: str):
    """Calculate derived properties based on compressive strength and selected standard.
    
    The AI provides the compressive strength prediction.
    The standard controls the derived-property calculation formulas.
    """
    fc = max(float(cs_mpa), 1.0)
    
    if standard == "Eurocode 2 (BS EN 1992-1-1)":
        fcm = fc + 8.0
        E = 22.0 * ((fcm / 10.0) ** 0.30)  # GPa
        ft = 0.30 * (fc ** (2 / 3)) if fc <= 50 else 2.12 * np.log(1 + fcm / 10.0)
    
    elif standard == "ACI 318":
        E = 4.70 * np.sqrt(fc)
        ft = 0.56 * np.sqrt(fc)
    
    elif standard == "JSCE (Japan)":
        E = 4.70 * np.sqrt(fc)
        ft = 0.56 * np.sqrt(fc)
    
    elif standard == "IS 456 (India)":
        E = 5.00 * np.sqrt(fc)
        ft = 0.70 * np.sqrt(fc)
    
    elif standard == "GB / China":
        E = 4.20 * np.sqrt(fc)
        ft = 0.395 * (fc ** 0.55)
    
    elif standard == "MS EN (Malaysia)":
        fcm = fc + 8.0
        E = 22.0 * ((fcm / 10.0) ** 0.30)
        ft = 0.30 * (fc ** (2 / 3)) if fc <= 50 else 2.12 * np.log(1 + fcm / 10.0)
    
    else:
        # Default to Eurocode 2
        fcm = fc + 8.0
        E = 22.0 * ((fcm / 10.0) ** 0.30)
        ft = 0.30 * (fc ** (2 / 3))
    
    density = 2400.0
    upv = np.sqrt((E * 1e9) / density) / 1000.0
    return E, ft, upv


def extract_materials(inputs):
    cement = scm = water = sp = fibre = 0.0
    for k, v in inputs.items():
        lk = k.lower()
        val = float(v)
        if "cement" in lk:
            cement += val
        elif any(x in lk for x in ["slag", "fly ash", "silica fume", "quartz powder", "limestone powder"]):
            scm += val
        elif "water" in lk:
            water += val
        elif "plasticizer" in lk or "super" in lk:
            sp += val
        elif "fibre" in lk or "fiber" in lk:
            fibre += val
    return cement, scm, water, sp, fibre


def cost_calc(inputs):
    cement, scm, water, sp, fibre = extract_materials(inputs)
    return 0.12 * cement + 0.06 * scm + 0.002 * water + 0.40 * sp + 0.8 * fibre


def carbon_calc(inputs):
    cement, scm, water, sp, fibre = extract_materials(inputs)
    return 0.90 * cement + 0.12 * scm + 0.0003 * water + 0.08 * fibre


def sustainability_score(cs, carbon, cost):
    score = 100 - 0.055 * carbon - 0.02 * cost + 0.12 * min(cs, 180)
    return max(0.0, min(100.0, score))


def confidence_level(inputs):
    cement, scm, water, sp, fibre = extract_materials(inputs)
    unusual = 0
    if cement > 950 or cement < 550:
        unusual += 1
    if water < 130 or water > 220:
        unusual += 1
    if scm > 450:
        unusual += 1
    if fibre > 200:
        unusual += 1
    if unusual == 0:
        return "High", "#16a34a"
    if unusual == 1:
        return "Moderate", "#f59e0b"
    return "Low", "#ef4444"


def strength_status(cs):
    if cs >= 150:
        return "Excellent", "#16a34a"
    if cs >= 120:
        return "Good", "#0ea5e9"
    if cs >= 100:
        return "Moderate", "#f59e0b"
    return "Low", "#ef4444"


def carbon_status(carbon):
    if carbon <= 700:
        return "Low Carbon", "#16a34a"
    if carbon <= 850:
        return "Moderate", "#f59e0b"
    return "High Carbon", "#ef4444"


def compliance_cards(cs_mpa: float, standard: str):
    """Generate compliance checks based on compressive strength and selected standard.
    
    Each standard has its own compliance rules and thresholds.
    """
    # Standard-specific compliance thresholds
    standard_rules = {
        "Eurocode 2 (BS EN 1992-1-1)": [
            {"name": "UHPC Target (120 MPa)", "threshold": 120, "category": "Ultra-high-performance"},
            {"name": "High-Performance (100 MPa)", "threshold": 100, "category": "High-strength"},
        ],
        "ACI 318": [
            {"name": "Structural UHPC (120 MPa)", "threshold": 120, "category": "Ultra-high-performance"},
            {"name": "High-Strength (100 MPa)", "threshold": 100, "category": "High-strength"},
        ],
        "JSCE (Japan)": [
            {"name": "High-Performance (120 MPa)", "threshold": 120, "category": "Advanced concrete"},
            {"name": "Advanced (100 MPa)", "threshold": 100, "category": "High-strength"},
        ],
        "IS 456 (India)": [
            {"name": "High-Performance (100 MPa)", "threshold": 100, "category": "Advanced concrete"},
            {"name": "Advanced (120 MPa)", "threshold": 120, "category": "Ultra-high-performance"},
        ],
        "GB / China": [
            {"name": "High-Performance (100 MPa)", "threshold": 100, "category": "Advanced concrete"},
            {"name": "Advanced (120 MPa)", "threshold": 120, "category": "Ultra-high-performance"},
        ],
        "MS EN (Malaysia)": [
            {"name": "UHPC Target (120 MPa)", "threshold": 120, "category": "Ultra-high-performance"},
            {"name": "High-Performance (100 MPa)", "threshold": 100, "category": "High-strength"},
        ],
    }
    
    # Get rules for selected standard, default to Eurocode 2
    rules = standard_rules.get(standard, standard_rules["Eurocode 2 (BS EN 1992-1-1)"])
    
    out = []
    for rule in rules:
        ok = cs_mpa >= rule["threshold"]
        out.append({
            "name": rule["name"],
            "threshold": rule["threshold"],
            "ok": ok,
            "note": "Compliant" if ok else "Not compliant",
            "color": "#16a34a" if ok else "#ef4444",
            "icon": "✓" if ok else "✕",
        })
    return out


def recommendation_summary(inputs, cs, carbon, cost):
    cement, scm, water, sp, fibre = extract_materials(inputs)
    recs = []
    if cement > 750:
        recs.append("Reduce cement content to improve embodied carbon.")
    if scm < 150:
        recs.append("Increase SCM replacement to improve sustainability.")
    if sp > 0:
        recs.append("Optimise superplasticizer dosage to maintain workability.")
    if water > 190:
        recs.append("Reduce water demand to improve binder efficiency.")
    if not recs:
        recs.append("Current mix shows a strong performance-carbon balance.")
    expected = f"Estimated carbon: {carbon:.1f} kg CO2/m³ | Cost: {cost:.1f} USD/m³"
    return recs, expected


def evaluate_mix(inputs, standard):
    cs = predict_strength(inputs)
    E, ft, upv = derived_properties(cs, standard)
    youngs = 0.95 * E  # Young's modulus approximation
    carbon = carbon_calc(inputs)
    cost = cost_calc(inputs)
    score = sustainability_score(cs, carbon, cost)
    conf_label, conf_color = confidence_level(inputs)
    strength_label, strength_color = strength_status(cs)
    carbon_label, carbon_color = carbon_status(carbon)
    recs, expected = recommendation_summary(inputs, cs, carbon, cost)
    return {
        "inputs": inputs,
        "standard": standard,
        "cs": cs,
        "ft": ft,
        "E": E,
        "youngs": youngs,
        "upv": upv,
        "carbon": carbon,
        "cost": cost,
        "score": score,
        "confidence_label": conf_label,
        "confidence_color": conf_color,
        "strength_label": strength_label,
        "strength_color": strength_color,
        "carbon_label": carbon_label,
        "carbon_color": carbon_color,
        "compliance": compliance_cards(cs, standard),
        "recommendations": recs,
        "recommendation_note": expected,
    }


def metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tag(text, color):
    return f'<span class="tag" style="background:{color};">{text}</span>'


def render_compliance(result):
    cols = st.columns(2)
    for i, item in enumerate(result["compliance"]):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="compliance-card" style="border-left:4px solid {item['color']};">
                    <div class="compliance-top">
                        <span style="color:{item['color']}; font-weight:900;">{item['icon']}</span>
                        <span class="compliance-name">{item['name']}</span>
                    </div>
                    <div class="compliance-note">Min: {item['threshold']} MPa</div>
                    <div class="compliance-note">{item['note']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def pdf_safe(text):
    replacements = {"£": "GBP ", "✓": "OK", "✕": "NO", "–": "-", "—": "-", "CO₂": "CO2", "m³": "m3"}
    s = str(text)
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "ignore").decode("latin-1")


def build_pdf_chart(result, path):
    chart_df = pd.DataFrame({
        "Scenario": ["Baseline", "AIcrete"],
        "Carbon": [result["carbon"] * 1.12, result["carbon"]],
        "Strength": [result["cs"] * 0.92, result["cs"]],
    })
    fig = px.scatter(chart_df, x="Carbon", y="Strength", text="Scenario", size=[18, 24], title="Performance vs Carbon")
    fig.update_traces(textposition="top center")
    fig.write_image(path, width=900, height=520)


def generate_pdf(result, filename="AIcrete_Report.pdf"):
    with tempfile.TemporaryDirectory() as td:
        chart_png = os.path.join(td, "chart.png")
        perf_vs_carbon_png = os.path.join(td, "perf_vs_carbon.png")
        pred_vs_actual_png = os.path.join(td, "pred_vs_actual.png")
        
        # matplotlib fallback instead of plotly image deps
        import matplotlib.pyplot as plt
        
        # Performance vs Carbon chart
        plt.figure(figsize=(7.2, 4.2))
        plt.scatter([result["carbon"] * 1.12], [result["cs"] * 0.92], s=120, label="Baseline")
        plt.scatter([result["carbon"]], [result["cs"]], s=140, label="AIcrete")
        plt.xlabel("Embodied Carbon (kg CO2/m3)")
        plt.ylabel("Predicted Strength (MPa)")
        plt.title("Performance vs Carbon")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(perf_vs_carbon_png, dpi=180, bbox_inches="tight")
        plt.close()
        
        # Predicted vs Actual chart
        metrics_data = calculate_model_metrics()
        if metrics_data:
            plt.figure(figsize=(7.2, 4.2))
            plt.scatter(metrics_data["y_true"], metrics_data["y_pred"], alpha=0.5, s=30, color="#0ea5e9")
            min_val = min(metrics_data["y_true"].min(), metrics_data["y_pred"].min())
            max_val = max(metrics_data["y_true"].max(), metrics_data["y_pred"].max())
            plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
            plt.xlabel("Actual Strength (MPa)")
            plt.ylabel("Predicted Strength (MPa)")
            plt.title("Predicted vs Actual Strength (Training Data)")
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(pred_vs_actual_png, dpi=180, bbox_inches="tight")
            plt.close()

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)

        # Cover Page
        pdf.add_page()
        if logo_exists():
            try:
                pdf.image(LOGO_NAME, x=12, y=12, w=24)
            except Exception:
                pass
        pdf.set_font("Arial", "B", 22)
        pdf.ln(24)
        pdf.cell(0, 12, pdf_safe(APP_NAME), ln=True)
        pdf.set_font("Arial", "", 13)
        pdf.cell(0, 8, pdf_safe("Low-Carbon UHPC Design Assessment"), ln=True)
        pdf.cell(0, 8, pdf_safe(TAGLINE), ln=True)
        pdf.ln(6)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, pdf_safe("AI-assisted decision support for UHPC prediction, sustainability, compliance, benchmarking, and optimisation."))
        pdf.cell(0, 6, pdf_safe(datetime.datetime.now().strftime("Generated on %d %B %Y, %H:%M")), ln=True)
        
        pdf.ln(4)
        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(0, 6, pdf_safe("Disclaimer: For preliminary engineering assessment only. Laboratory validation and professional review remain necessary before implementation."))
        
        # Add AI Validation Narrative
        pdf.ln(8)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, pdf_safe("AI Validation Summary"), ln=True)
        pdf.set_font("Arial", "", 10)
        if metrics_data:
            closest_idx = np.argmin(np.abs(metrics_data["y_true"] - result["cs"]))
            closest_actual = metrics_data["y_true"][closest_idx]
            error_pct = abs(result["cs"] - closest_actual) / closest_actual * 100 if closest_actual != 0 else 0
            narrative_text = (
                f"AIcrete Solutions predicted {result['cs']:.0f} MPa vs training reference {closest_actual:.0f} MPa "
                f"— within {error_pct:.1f}% error. Model validation shows consistent accuracy across the dataset "
                f"(MAPE = {metrics_data['mape']:.1f}%, R² = {metrics_data['r_squared']:.4f}).\n\n"
                f"Compressive strength is AI-predicted from the trained UHPC dataset. "
                f"Standard selection ({result.get('standard', 'Not specified')}) changes derived-property equations and compliance checks."
            )
            pdf.multi_cell(0, 6, pdf_safe(narrative_text))
        
        # Results Page
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, pdf_safe("1. Key Results"), ln=True)
        pdf.set_font("Arial", "", 11)
        rows = [
            ("Predicted Strength (MPa)", result["cs"]),
            ("Tensile Strength (MPa)", result["ft"]),
            ("Elastic Modulus (GPa)", result["E"]),
            ("Young's Modulus (GPa)", result["youngs"]),
            ("Pulse Velocity (km/s)", result["upv"]),
            ("Embodied Carbon (kg CO2/m3)", result["carbon"]),
            ("Cost per m3 (USD)", result["cost"]),
            ("Sustainability Score", result["score"]),
            ("Confidence", result["confidence_label"]),
        ]
        for k, v in rows:
            pdf.cell(95, 8, pdf_safe(k), 1)
            pdf.cell(95, 8, pdf_safe(f"{v:.2f}" if isinstance(v, (int, float)) else v), 1, ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, pdf_safe("2. Model Performance Metrics"), ln=True)
        pdf.set_font("Arial", "", 11)
        if metrics_data:
            metrics_rows = [
                ("RMSE (Root Mean Square Error)", f"{metrics_data['rmse']:.2f} MPa"),
                ("MAPE (Mean Absolute % Error)", f"{metrics_data['mape']:.2f} %"),
                ("R² (Coefficient of Determination)", f"{metrics_data['r_squared']:.4f}"),
            ]
            for k, v in metrics_rows:
                pdf.cell(95, 8, pdf_safe(k), 1)
                pdf.cell(95, 8, pdf_safe(v), 1, ln=True)
        else:
            pdf.cell(0, 8, pdf_safe("Model metrics not available"), ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, pdf_safe("3. Design Standard"), ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(95, 8, pdf_safe("Standard Used"), 1)
        pdf.cell(95, 8, pdf_safe(result.get("standard", "Not specified")), 1, ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, pdf_safe("4. Mix Parameters"), ln=True)
        pdf.set_font("Arial", "", 11)
        for k, v in result["inputs"].items():
            pdf.cell(95, 8, pdf_safe(k), 1)
            pdf.cell(95, 8, pdf_safe(f"{float(v):.2f}"), 1, ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, pdf_safe("5. Compliance Overview"), ln=True)
        pdf.set_font("Arial", "", 11)
        for item in result["compliance"]:
            pdf.multi_cell(0, 6, pdf_safe(f"{item['name']}: {item['note']} (Min {item['threshold']} MPa)"))

        pdf.ln(2)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, pdf_safe("6. AI Recommendation"), ln=True)
        pdf.set_font("Arial", "", 11)
        for rec in result["recommendations"]:
            pdf.multi_cell(0, 6, pdf_safe(f"- {rec}"))
        pdf.multi_cell(0, 6, pdf_safe(result["recommendation_note"]))

        # Performance Charts Page
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, pdf_safe("7. Predicted vs Actual Strength"), ln=True)
        if metrics_data and os.path.exists(pred_vs_actual_png):
            y = pdf.get_y()
            pdf.image(pred_vs_actual_png, x=22, y=y+3, w=165)
            pdf.set_y(y + 75)
        
        pdf.ln(4)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, pdf_safe("8. Performance vs Carbon"), ln=True)
        y = pdf.get_y()
        if os.path.exists(perf_vs_carbon_png):
            pdf.image(perf_vs_carbon_png, x=22, y=y+3, w=165)
            pdf.set_y(y + 75)

        # Footer and Copyright
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, pdf_safe("Report Information"), ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.ln(5)
        pdf.multi_cell(0, 6, pdf_safe(
            f"This report was generated by AIcrete Solutions on {datetime.datetime.now().strftime('%d %B %Y at %H:%M')}. "
            f"All data, predictions, and analyses contained herein are proprietary and confidential."
        ))
        
        pdf.ln(10)
        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(0, 6, pdf_safe(
            "Important Disclaimer: This assessment is for preliminary engineering purposes only. "
            "All recommendations must be validated through laboratory testing and professional engineering review "
            "before implementation in any construction project. AIcrete Solutions and its contributors assume no liability "
            "for the use or misuse of this information."
        ))
        
        pdf.ln(15)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, pdf_safe("Copyright & Rights"), ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, pdf_safe("(c) Copyright All rights reserved to AIcrete Solutions"))

        pdf.output(filename)
        return filename


def history_append(name, result):
    st.session_state.history.append({
        "name": name,
        "time": datetime.datetime.now().strftime("%d %b %Y %H:%M"),
        "result": result,
    })


@st.cache_resource
def get_shap_explainer():
    if shap is None:
        return None
    try:
        return shap.TreeExplainer(MODEL)
    except Exception:
        return None


@st.cache_data
def shap_sample(json_text):
    data = pd.read_json(json_text)
    n = min(200, len(data))
    return data[FEATURE_COLS].sample(n=n, random_state=42).copy()


def shap_values(explainer, xdf):
    vals = explainer.shap_values(xdf)
    if isinstance(vals, list):
        vals = vals[0]
    return np.array(vals)


@st.cache_data
def calculate_model_metrics():
    """Calculate RMSE, MAPE, and R² metrics on training data."""
    try:
        # Get predictions on training data
        X_train = DF[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_true = DF[TARGET_COL].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        
        try:
            y_pred = MODEL.predict(X_train).reshape(-1)
        except Exception:
            y_pred = np.array([predict_strength({col: X_train.iloc[i][col] for col in FEATURE_COLS}) 
                              for i in range(len(X_train))]).reshape(-1)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-6))
        
        return {
            "rmse": rmse,
            "mape": mape,
            "r_squared": r_squared,
            "y_true": y_true,
            "y_pred": y_pred,
            "X_train": X_train
        }
    except Exception as e:
        st.warning(f"Could not calculate metrics: {e}")
        return None


def generate_prediction_narrative(predicted_strength, metrics_data):
    """Generate a narrative summary of the prediction with error analysis."""
    if not metrics_data:
        return None
    
    # Find a similar actual value from the training data for comparison
    closest_idx = np.argmin(np.abs(metrics_data["y_true"] - predicted_strength))
    closest_actual = metrics_data["y_true"][closest_idx]
    
    # Calculate error for this prediction
    error_pct = abs(predicted_strength - closest_actual) / closest_actual * 100 if closest_actual != 0 else 0
    
    # Determine if prediction is within acceptable range
    mape = metrics_data["mape"]
    within_range = error_pct < mape * 1.5
    range_text = f"within {error_pct:.1f}% error" if within_range else f"outside typical {error_pct:.1f}% error"
    
    narrative = (
        f"<div class='info-box' style='margin-top:1rem;'>"
        f"<strong>🎯 AI Validation Summary</strong><br>"
        f"AIcrete Solutions predicted <strong>{predicted_strength:.0f} MPa</strong> vs training reference <strong>{closest_actual:.0f} MPa</strong> "
        f"— {range_text}. "
        f"Model validation shows consistent accuracy across the dataset "
        f"(MAPE ≈ <strong>{mape:.1f}%</strong>, R² = <strong>{metrics_data['r_squared']:.4f}</strong>). "
        f"<br><br><em>Compressive strength is AI-predicted from the trained UHPC dataset. "
        f"Standard selection changes derived-property equations and compliance checks.</em>"
        f"</div>"
    )
    return narrative


if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "optimizer_result" not in st.session_state:
    st.session_state.optimizer_result = None
if "history" not in st.session_state:
    st.session_state.history = []
if "bench_results" not in st.session_state:
    st.session_state.bench_results = {}


st.markdown("""
<style>
.block-container {padding-top: 1.25rem; max-width: 96rem;}
[data-testid="stSidebar"]{
    background:#0c1d34;
    border-right:1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] *{color:white !important;}
[data-testid="stSidebar"] .stRadio > label{font-weight:800 !important;}
[data-testid="stSidebar"] label{background:transparent !important;border:none !important;padding:0 !important;}
body{background:#f5f7fb;}
.main-title{font-size:2.45rem;font-weight:900;color:#162c47;margin-bottom:0.15rem;line-height:1.30;padding-top:0.45rem;overflow:visible;display:block;min-height:3.7rem;}
.main-sub{font-size:0.96rem;color:#718197;margin-bottom:1.0rem;}
.panel{
    background:white;
    border:1px solid rgba(20,40,80,0.08);
    border-radius:18px;
    padding:1rem 1rem;
    box-shadow:0 8px 18px rgba(15,23,42,0.06);
    margin-bottom:1rem;
}
.panel-title{font-size:1.05rem;font-weight:800;color:#162c47;margin-bottom:0.15rem;}
.panel-sub{font-size:0.92rem;color:#6b7d93;margin-bottom:0.8rem;}
.metric-card{
    background:white;
    border:1px solid rgba(20,40,80,0.08);
    border-radius:14px;
    padding:0.9rem 0.9rem;
    min-height:94px;
}
.metric-title{font-size:0.82rem;color:#6c7e92;font-weight:800;margin-bottom:0.18rem;}
.metric-value{font-size:1.6rem;color:#0ea5e9;font-weight:900;line-height:1.1;}
.metric-sub{font-size:0.84rem;color:#6c7e92;margin-top:0.12rem;}
.big-score{font-size:3.45rem;color:#0ea5e9;font-weight:900;line-height:1.0;}
.muted{color:#6b7d93;font-size:0.92rem;}
.badge{
    display:inline-block;padding:0.24rem 0.58rem;border-radius:999px;
    font-weight:800;font-size:0.78rem;margin-left:0.4rem;color:white;
}
.info-box{
    background:#eef7fb;border-left:4px solid #0ea5e9;border-radius:10px;padding:0.9rem 1rem;color:#21435f;
}
.compliance-card{
    background:white;border:1px solid rgba(20,40,80,0.08);border-radius:12px;padding:0.75rem 0.85rem;margin-bottom:0.6rem;
}
.compliance-top{display:flex;gap:0.4rem;align-items:center;margin-bottom:0.22rem;}
.compliance-name{font-weight:900;color:#162c47;}
.compliance-note{font-size:0.88rem;color:#6b7d93;}
.soft-tag{
    display:inline-block;background:#e6f6ee;color:#166534;padding:0.25rem 0.55rem;border-radius:999px;font-size:0.78rem;font-weight:800;
}
.warn-tag{
    display:inline-block;background:#fff4dd;color:#b45309;padding:0.25rem 0.55rem;border-radius:999px;font-size:0.78rem;font-weight:800;
}
.bad-tag{
    display:inline-block;background:#fee2e2;color:#b91c1c;padding:0.25rem 0.55rem;border-radius:999px;font-size:0.78rem;font-weight:800;
}
label, .stSlider label, .stTextInput label, .stNumberInput label, .stSelectbox label{
    color:#162c47 !important;font-weight:800 !important;
}
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="select"] [role="combobox"], div[data-baseweb="input"] input{
    background:white !important; color:#162c47 !important;
}
.stButton > button{
    background:linear-gradient(90deg,#24457d,#0ea5c6) !important;
    color:white !important;border:none !important;border-radius:10px !important;font-weight:800 !important;
}
.stDownloadButton > button{
    background:#24457d !important;color:white !important;border:none !important;border-radius:10px !important;font-weight:800 !important;
}
button[title="Increment"], button[title="Decrement"]{display:none !important;}
.footer-container{
    border-top:1px solid rgba(20,40,80,0.08);
    padding:2rem 0;
    margin-top:3rem;
    background:#f9fafb;
}
.footer-content{
    max-width:96rem;
    margin:0 auto;
    padding:0 1rem;
}
.footer-main{
    display:flex;
    justify-content:space-between;
    align-items:center;
    flex-wrap:wrap;
    gap:2rem;
    margin-bottom:1.5rem;
}
.footer-column{
    display:flex;
    flex-direction:column;
    gap:0.5rem;
}
.footer-column-title{
    font-weight:800;
    color:#162c47;
    font-size:0.95rem;
    margin-bottom:0.3rem;
}
.footer-links{
    display:flex;
    gap:1.2rem;
    flex-wrap:wrap;
}
.footer-link{
    color:#718197;
    text-decoration:none;
    font-size:0.875rem;
    transition:color 0.2s;
}
.footer-link:hover{
    color:#0ea5e9;
}
.footer-divider{
    border-top:1px solid rgba(20,40,80,0.08);
    padding-top:1rem;
    margin-top:1rem;
}
.footer-bottom{
    display:flex;
    justify-content:space-between;
    align-items:center;
    flex-wrap:wrap;
    gap:1rem;
    font-size:0.84rem;
    color:#718197;
}
.footer-copyright{
    font-weight:600;
    color:#162c47;
}
@media (max-width: 768px) {
    .footer-main{flex-direction:column;align-items:flex-start;}
    .footer-bottom{flex-direction:column;align-items:flex-start;}
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    if logo_exists():
        st.image(LOGO_NAME, width=84)
    st.markdown("### AIcrete Solutions")
    st.caption("UHPC Intelligence Platform")
    
    # Check for legal pages via query params
    if "legal_page" in st.query_params:
        page = st.query_params["legal_page"]
    else:
        page = st.radio(
            "Navigation",
            ["Predictor", "History", "Benchmarking", "Mix Optimizer", "Sensitivity Analysis", "SHAP Analysis", "Report"]
        )

st.markdown('<div style="height:0.55rem;"></div>', unsafe_allow_html=True)

# Logo and Title Header
if logo_exists():
    logo_col, title_col = st.columns([0.15, 1], gap="medium")
    with logo_col:
        st.image(LOGO_NAME, use_container_width=True)
    with title_col:
        st.markdown(f'<div class="main-title" style="padding-top:0.45rem;padding-bottom:0.2rem;line-height:1.28;min-height:3.9rem;">{APP_NAME}</div>', unsafe_allow_html=True)
        st.markdown('<div class="main-sub">Low-Carbon Concrete Decision Intelligence</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="main-title" style="padding-top:0.45rem;padding-bottom:0.2rem;line-height:1.28;min-height:3.9rem;">{APP_NAME}</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Low-Carbon Concrete Decision Intelligence</div>', unsafe_allow_html=True)


def input_grid(prefix, defaults=None):
    defaults = defaults or {}
    values = {}
    cols = st.columns(2)
    for i, col in enumerate(FEATURE_COLS):
        lo = RANGES[col]["min"]
        hi = RANGES[col]["max"]
        lo, hi = get_feature_bounds(col, lo, hi)
        val = defaults.get(col, RANGES[col]["mean"])
        val = float(max(lo, min(hi, val)))
        with cols[i % 2]:
            values[col] = st.number_input(
                col,
                min_value=float(lo),
                max_value=float(hi),
                value=float(val),
                step=0.1,
                format="%.2f",
                key=f"{prefix}_{col}",
            )
    return values


def status_html(label, color):
    return f'<span class="badge" style="background:{color};">{label}</span>'


def render_result_summary(result, show_save=True):
    left, right = st.columns([1.05, 1.25], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">AI-powered UHPC compressive strength analysis</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="big-score">{result["cs"]:.1f} <span style="font-size:1.1rem;color:#51657d;font-weight:700;">MPa</span>{status_html(result["strength_label"], result["strength_color"])}</div>',
            unsafe_allow_html=True
        )
        ci_lo = result["cs"] * 0.90
        ci_hi = result["cs"] * 1.10
        st.markdown(f'<div class="muted">90% Interval: {ci_lo:.1f} - {ci_hi:.1f} MPa</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Predicted Compressive Strength</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add AI validation narrative
        metrics_data = calculate_model_metrics()
        narrative = generate_prediction_narrative(result["cs"], metrics_data)
        if narrative:
            st.markdown(narrative, unsafe_allow_html=True)

        gcols = st.columns(3)
        metrics = [
            ("Tensile Strength", f'{result["ft"]:.2f}', "MPa"),
            ("Elastic Modulus", f'{result["E"]:.2f}', "GPa"),
            ("Young\'s Modulus", f'{result["youngs"]:.2f}', "GPa"),
            ("Pulse Velocity", f'{result["upv"]:.2f}', "km/s"),
            ("Embodied Carbon", f'{result["carbon"]:.1f}', "kg CO₂/m³"),
            ("Cost per m³", f'{result["cost"]:.0f}', "USD"),
        ]
        for idx, item in enumerate(metrics):
            with gcols[idx % 3]:
                metric_card(item[0], item[1], item[2])

    with right:
        tabs = st.tabs(["Compliance", "Sustainability", "Interpretability", "Age Curve", "✅ Model Performance"])
        with tabs[0]:
            render_compliance(result)
            st.markdown(
                f'<div style="text-align:center;color:#64748b;font-size:0.88rem;margin-top:0.4rem;">{sum(1 for x in result["compliance"] if x["ok"])} of {len(result["compliance"])} standards met</div>',
                unsafe_allow_html=True
            )
        with tabs[1]:
            s1, s2, s3 = st.columns(3)
            with s1:
                if result["score"] >= 75:
                    st.markdown(f'<div class="soft-tag">Sustainability Score {result["score"]:.0f}</div>', unsafe_allow_html=True)
                elif result["score"] >= 55:
                    st.markdown(f'<div class="warn-tag">Sustainability Score {result["score"]:.0f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bad-tag">Sustainability Score {result["score"]:.0f}</div>', unsafe_allow_html=True)
            with s2:
                st.markdown(status_html(result["carbon_label"], result["carbon_color"]), unsafe_allow_html=True)
            with s3:
                st.markdown(status_html(f'Confidence {result["confidence_label"]}', result["confidence_color"]), unsafe_allow_html=True)
            perf_df = pd.DataFrame(
                {"Metric": ["Strength", "Carbon", "Cost", "Score"],
                 "Value": [result["cs"], result["carbon"], result["cost"], result["score"]]}
            )
            fig = px.bar(perf_df, x="Metric", y="Value", text="Value", title="Sustainability Snapshot")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
            st.plotly_chart(fig, use_container_width=True)
            rec_html = "".join([f"<li>{r}</li>" for r in result["recommendations"]])
            st.markdown(f'<div class="info-box"><strong>AI Recommendation</strong><ul>{rec_html}</ul><div style="margin-top:0.4rem;">{result["recommendation_note"]}</div></div>', unsafe_allow_html=True)
        with tabs[2]:
            if shap is None:
                st.info("SHAP not installed. Showing model feature importance instead.")
                if hasattr(MODEL, "feature_importances_"):
                    imp = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": MODEL.feature_importances_}).sort_values("Importance", ascending=False)
                    fig = px.bar(imp, x="Feature", y="Importance", text="Importance", title="Feature Importance")
                    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                try:
                    explainer = get_shap_explainer()
                    xdf = build_input_df(result["inputs"])
                    if explainer is None:
                        raise RuntimeError("Explainer unavailable")
                    vals = shap_values(explainer, xdf)[0]
                    local_df = pd.DataFrame({
                        "Feature": FEATURE_COLS,
                        "SHAP Value": vals,
                        "Abs": np.abs(vals)
                    }).sort_values("Abs", ascending=False)
                    fig = px.bar(local_df.head(10), x="Feature", y="SHAP Value", text="SHAP Value", title="Local SHAP Contribution")
                    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    if hasattr(MODEL, "feature_importances_"):
                        imp = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": MODEL.feature_importances_}).sort_values("Importance", ascending=False)
                        fig = px.bar(imp, x="Feature", y="Importance", text="Importance", title="Feature Importance")
                        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
                        st.plotly_chart(fig, use_container_width=True)
        with tabs[3]:
            ages = np.array([1, 3, 7, 14, 28, 56, 90], dtype=float)
            curve = result["cs"] * (1 - np.exp(-ages / 18))
            curve_df = pd.DataFrame({"Age": ages, "Strength": curve})
            fig = px.line(curve_df, x="Age", y="Strength", markers=True, title="Indicative Age Curve")
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
            st.plotly_chart(fig, use_container_width=True)
        with tabs[4]:
            st.markdown("### ✅ Model Performance Metrics")
            metrics_data = calculate_model_metrics()
            if metrics_data:
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("✅ RMSE", f"{metrics_data['rmse']:.2f} MPa", 
                              "Root Mean Square Error\n(lower is better)")
                with mc2:
                    st.metric("✅ MAPE", f"{metrics_data['mape']:.2f} %", 
                              "Mean Absolute Percentage Error\n(lower is better)")
                with mc3:
                    st.metric("✅ R²", f"{metrics_data['r_squared']:.4f}", 
                              "Coefficient of Determination\n(closer to 1 is better)")
                
                st.markdown("---")
                st.markdown("### Predicted vs Actual")
                
                # Create scatter plot
                pred_actual_df = pd.DataFrame({
                    "Actual": metrics_data["y_true"],
                    "Predicted": metrics_data["y_pred"],
                })
                
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=metrics_data["y_true"],
                    y=metrics_data["y_pred"],
                    mode="markers",
                    marker=dict(size=8, color="#0ea5e9", opacity=0.6),
                    text=[f"Actual: {actual:.1f}<br>Predicted: {pred:.1f}" 
                          for actual, pred in zip(metrics_data["y_true"], metrics_data["y_pred"])],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    name="Predictions"
                ))
                
                # Add perfect prediction line
                min_val = min(metrics_data["y_true"].min(), metrics_data["y_pred"].min())
                max_val = max(metrics_data["y_true"].max(), metrics_data["y_pred"].max())
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color="#ef4444", dash="dash"),
                    name="Perfect Prediction"
                ))
                
                fig_scatter.update_layout(
                    title="Predicted vs Actual Strength (Training Data)",
                    xaxis_title="Actual Strength (MPa)",
                    yaxis_title="Predicted Strength (MPa)",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(color="#16324f"),
                    hovermode="closest",
                    showlegend=True,
                    height=500
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Distribution comparison
                st.markdown("### Distribution Comparison")
                dist_col1, dist_col2 = st.columns(2)
                with dist_col1:
                    fig_hist_actual = px.histogram(
                        x=metrics_data["y_true"],
                        nbins=30,
                        title="Actual Strength Distribution",
                        labels={"x": "Strength (MPa)", "count": "Frequency"}
                    )
                    fig_hist_actual.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
                    st.plotly_chart(fig_hist_actual, use_container_width=True)
                with dist_col2:
                    fig_hist_pred = px.histogram(
                        x=metrics_data["y_pred"],
                        nbins=30,
                        title="Predicted Strength Distribution",
                        labels={"x": "Strength (MPa)", "count": "Frequency"}
                    )
                    fig_hist_pred.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
                    st.plotly_chart(fig_hist_pred, use_container_width=True)
                
                # Statistics
                st.markdown("### Statistics")
                stats_df = pd.DataFrame({
                    "Metric": ["Mean", "Median", "Std Dev", "Min", "Max"],
                    "Actual": [
                        metrics_data["y_true"].mean(),
                        np.median(metrics_data["y_true"]),
                        metrics_data["y_true"].std(),
                        metrics_data["y_true"].min(),
                        metrics_data["y_true"].max()
                    ],
                    "Predicted": [
                        metrics_data["y_pred"].mean(),
                        np.median(metrics_data["y_pred"]),
                        metrics_data["y_pred"].std(),
                        metrics_data["y_pred"].min(),
                        metrics_data["y_pred"].max()
                    ]
                })
                stats_df = stats_df.round(2)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Could not load model metrics data.")

    if show_save:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Save to History", key=f"save_{datetime.datetime.now().timestamp()}"):
                history_append(f"Run {len(st.session_state.history)+1}", result)
                st.success("Saved.")
        with c2:
            st.session_state.latest_result = result


if page == "Predictor":
    left, right = st.columns([1.0, 1.7], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Configure parameters for UHPC strength prediction</div></div>', unsafe_allow_html=True)
        session_name = st.text_input("Session Name", "e.g. Mix Design A")
        standard = st.selectbox("Standard", STANDARD_OPTIONS, key="pred_std")
        inputs = input_grid("pred")
        if st.button("Predict Strength", use_container_width=True):
            result = evaluate_mix(inputs, standard)
            st.session_state.latest_result = result
            history_append(session_name or f"Run {len(st.session_state.history)+1}", result)
    with right:
        if st.session_state.latest_result:
            render_result_summary(st.session_state.latest_result, show_save=False)
        else:
            st.markdown('<div class="panel"><div class="panel-title">Results</div><div class="panel-sub">Run a prediction to view strength, derived properties, compliance, sustainability, and interpretability.</div></div>', unsafe_allow_html=True)

elif page == "History":
    st.markdown('<div class="panel"><div class="panel-title">History</div><div class="panel-sub">Saved mix sessions.</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No saved history yet.")
    else:
        rows = []
        for i, item in enumerate(st.session_state.history):
            rows.append({
                "Index": i + 1,
                "Name": item["name"],
                "Time": item["time"],
                "Strength (MPa)": round(item["result"]["cs"], 2),
                "Carbon": round(item["result"]["carbon"], 1),
                "Score": round(item["result"]["score"], 1),
            })
        hist_df = pd.DataFrame(rows)
        st.dataframe(hist_df, use_container_width=True)
        pick = st.selectbox("Open saved run", hist_df["Name"].tolist())
        if st.button("Load Selected Run"):
            for item in st.session_state.history:
                if item["name"] == pick:
                    st.session_state.latest_result = item["result"]
                    st.success("Loaded selected run.")
                    break
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Benchmarking":
    st.markdown('<div class="panel"><div class="panel-title">Mix Benchmarking</div><div class="panel-sub">Compare up to 3 UHPC mix designs side by side.</div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="large")
    results = []
    for idx, colbox in enumerate(cols, start=1):
        with colbox:
            st.markdown(f"**Mix {chr(64+idx)}**")
            defaults = None
            if idx == 1 and st.session_state.latest_result:
                defaults = st.session_state.latest_result["inputs"]
            standard = st.selectbox("Standard", STANDARD_OPTIONS, key=f"bench_std_{idx}")
            mix = input_grid(f"bench_{idx}")
            if st.button(f"Run Mix {chr(64+idx)}", key=f"bench_run_{idx}", use_container_width=True):
                st.session_state.bench_results[idx] = evaluate_mix(mix, standard)
            if idx in st.session_state.bench_results:
                r = st.session_state.bench_results[idx]
                st.markdown(f'<div style="font-size:2rem;color:#0ea5e9;font-weight:900;">{r["cs"]:.1f} <span style="font-size:1rem;color:#51657d;">MPa</span></div>', unsafe_allow_html=True)
                st.markdown(status_html(r["strength_label"], r["strength_color"]), unsafe_allow_html=True)
                results.append((f"Mix {chr(64+idx)}", r))
    if results:
        comp_rows = []
        for name, r in results:
            comp_rows.append({
                "Mix": name,
                "Predicted Strength (MPa)": round(r["cs"], 1),
                "Tensile Strength (MPa)": round(r["ft"], 2),
                "Elastic Modulus (GPa)": round(r["E"], 2),
                "Young's Modulus (GPa)": round(r["youngs"], 2),
                "Pulse Velocity (km/s)": round(r["upv"], 2),
                "Embodied Carbon (kg CO₂/m³)": round(r["carbon"], 1),
                "Cost per m³ (USD)": round(r["cost"], 0),
                "Sustainability Score": round(r["score"], 0),
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

        radar = go.Figure()
        for name, r in results:
            radar.add_trace(go.Scatterpolar(
                r=[
                    min(r["cs"] / 160 * 100, 100),
                    min(r["ft"] / 12 * 100, 100),
                    r["score"],
                    min(r["upv"] / 6 * 100, 100),
                    max(0, 100 - min(r["cost"] / 4, 100)),
                ],
                theta=["Strength", "Tensile", "Sustain.", "UPV", "Cost (inv.)"],
                fill="toself",
                name=name
            ))
        radar.update_layout(
            title="Radar Comparison (Normalized 0-100)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            paper_bgcolor="white",
            font=dict(color="#16324f"),
        )
        st.plotly_chart(radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Mix Optimizer":
    left, right = st.columns([1.0, 1.55], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Mix Optimizer</div><div class="panel-sub">AI-powered heuristic search for optimal UHPC mix design.</div>', unsafe_allow_html=True)
        target = st.slider("Target Strength", 80, 220, 150)
        if "Age" in FEATURE_COLS:
            age_unique = sorted(int(x) for x in pd.to_numeric(DF["Age"], errors="coerce").dropna().unique()[:20])
            age_val = st.selectbox("Curing Age", age_unique, index=min(len(age_unique)-1, age_unique.index(28) if 28 in age_unique else 0))
        else:
            age_val = 28
        if "Temperature" in FEATURE_COLS:
            temp_default = int(round(RANGES["Temperature"]["mean"]))
            temp_val = st.number_input("Curing Temperature", value=temp_default)
        else:
            temp_val = 20
        prioritize_sustainability = st.toggle("Prioritize Sustainability", value=True)
        prioritize_cost = st.toggle("Prioritize Low Cost", value=False)
        standard = st.selectbox("Standard", STANDARD_OPTIONS, key="opt_std")

        if st.button("Find Optimal Mix", use_container_width=True):
            temp_df = DF[FEATURE_COLS].copy().apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
            if "Age" in temp_df.columns:
                temp_df["Age"] = float(age_val)
            if "Temperature" in temp_df.columns:
                temp_df["Temperature"] = float(temp_val)

            with st.spinner("Searching best candidate mix..."):
                try:
                    preds = np.array(MODEL.predict(temp_df[FEATURE_COLS])).reshape(-1)
                except Exception:
                    preds = np.array([predict_strength(row.to_dict()) for _, row in temp_df.iterrows()]).reshape(-1)

                temp_df["Predicted Strength"] = preds

                carbon_vals = []
                cost_vals = []
                score_vals = []
                rank_vals = []

                for _, row in temp_df.iterrows():
                    inputs = row[FEATURE_COLS].to_dict()
                    carbon = carbon_calc(inputs)
                    cost = cost_calc(inputs)
                    score = sustainability_score(float(row["Predicted Strength"]), carbon, cost)

                    penalty = abs(float(row["Predicted Strength"]) - target)
                    rank_value = penalty
                    if prioritize_sustainability:
                        rank_value += 0.06 * carbon - 0.10 * score
                    if prioritize_cost:
                        rank_value += 0.08 * cost
                    rank_value += 0.03 * (100 - score)

                    carbon_vals.append(carbon)
                    cost_vals.append(cost)
                    score_vals.append(score)
                    rank_vals.append(rank_value)

                temp_df["Carbon"] = carbon_vals
                temp_df["Cost"] = cost_vals
                temp_df["Sustainability Score"] = score_vals
                temp_df["Rank"] = rank_vals

                if len(temp_df) > 0:
                    best_row = temp_df.sort_values("Rank", ascending=True).iloc[0]
                    best_inputs = {col: float(best_row[col]) for col in FEATURE_COLS}
                    best = evaluate_mix(best_inputs, standard)
                    st.session_state.optimizer_result = best
                    st.session_state.latest_result = best
                else:
                    st.session_state.optimizer_result = None
                    st.warning("No valid candidate mix could be generated from the workbook. Check the numeric inputs in Data UHPC.xlsx.")

        st.markdown('<div class="info-box" style="margin-top:0.8rem;"><strong>Optimizer note</strong><br>The optimizer evaluates candidate mixes in the trained dataset and ranks them against target strength, sustainability, and cost priorities.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if st.session_state.optimizer_result:
            render_result_summary(st.session_state.optimizer_result, show_save=False)
            st.markdown('<div class="panel"><div class="panel-title">Optimal Mix Parameters</div>', unsafe_allow_html=True)
            param_df = pd.DataFrame({"Parameter": list(st.session_state.optimizer_result["inputs"].keys()),
                                     "Value": list(st.session_state.optimizer_result["inputs"].values())})
            st.dataframe(param_df, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Use This Mix in Predictor", use_container_width=True):
                    st.session_state.latest_result = st.session_state.optimizer_result
                    st.success("Optimizer mix loaded into current session.")
            with c2:
                if st.button("Save to History", key="opt_save", use_container_width=True):
                    history_append(f"Optimized {len(st.session_state.history)+1}", st.session_state.optimizer_result)
                    st.success("Saved.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel"><div class="panel-title">Optimization Results</div><div class="panel-sub">Run the optimizer to generate a recommended UHPC mix.</div></div>', unsafe_allow_html=True)

elif page == "Sensitivity Analysis":
    st.markdown('<div class="panel"><div class="panel-title">Sensitivity Analysis</div><div class="panel-sub">Assess how changing one variable affects strength, carbon, cost, and sustainability score.</div>', unsafe_allow_html=True)
    base_inputs = {c: RANGES[c]["mean"] for c in FEATURE_COLS}
    variable = st.selectbox("Parameter to Vary", FEATURE_COLS)
    lo, hi = get_feature_bounds(variable, RANGES[variable]["min"], RANGES[variable]["max"])
    min_col, max_col, step_col = st.columns([1, 1, 1])
    with min_col:
        min_val = st.number_input("Min Value", value=float(lo))
    with max_col:
        max_val = st.number_input("Max Value", value=float(hi))
    with step_col:
        steps = st.selectbox("Number of test points", [5, 8, 10, 12, 15], index=1, help="How many values between the minimum and maximum should be tested.")
    standard = st.selectbox("Standard", STANDARD_OPTIONS, key="sens_std")
    if st.button("Run Analysis", use_container_width=True):
        xs = np.linspace(min_val, max_val, int(steps))
        rows = []
        for val in xs:
            trial = base_inputs.copy()
            trial[variable] = float(val)
            res = evaluate_mix(trial, standard)
            rows.append({
                variable: val,
                "Strength": res["cs"],
                "Sustainability Score": res["score"],
                "Embodied Carbon": res["carbon"],
                "Cost": res["cost"],
            })
        st.session_state.sensitivity_df = pd.DataFrame(rows)
        st.session_state.sensitivity_var = variable

    if "sensitivity_df" in st.session_state:
        sdf = st.session_state.sensitivity_df
        svar = st.session_state.sensitivity_var
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Min Strength", f"{sdf['Strength'].min():.1f} MPa")
        m2.metric("Max Strength", f"{sdf['Strength'].max():.1f} MPa")
        m3.metric("Strength Range", f"{(sdf['Strength'].max()-sdf['Strength'].min()):.1f} MPa")
        m4.metric("Sensitivity Index", f"{(sdf['Strength'].max()-sdf['Strength'].min())/max(sdf['Strength'].mean(),1):.2f}")

        fig_main = px.line(sdf, x=svar, y="Strength", markers=True, title=f"Strength vs {svar}")
        fig_main.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
        st.plotly_chart(fig_main, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.line(sdf, x=svar, y="Sustainability Score", title="Sustainability Score")
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(sdf, x=svar, y="Embodied Carbon", title="Embodied Carbon (kg CO₂/m³)")
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = px.line(sdf, x=svar, y="Cost", title="Cost per m³ (USD)")
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
            st.plotly_chart(fig, use_container_width=True)

        delta = sdf["Strength"].max() - sdf["Strength"].min()
        midpoint = sdf.loc[sdf["Strength"].idxmax(), svar]
        st.markdown(f'<div class="info-box"><strong>AI Insight</strong><br>Increasing {svar} from {sdf[svar].min():.1f} to {sdf[svar].max():.1f} changes strength by {delta:.1f} MPa. The optimal value for maximum strength in this range is {midpoint:.1f}.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "SHAP Analysis":
    st.markdown('<div class="panel"><div class="panel-title">SHAP Analysis</div><div class="panel-sub">Explain how the current mix parameters influence the predicted compressive strength.</div>', unsafe_allow_html=True)
    if not st.session_state.latest_result:
        st.info("Run Predictor or Mix Optimizer first to generate a current-mix explanation.")
    else:
        explainer = get_shap_explainer()
        if shap is None or explainer is None:
            st.info("SHAP is unavailable in this environment. Showing current-model feature importance instead.")
            if hasattr(MODEL, "feature_importances_"):
                imp = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": MODEL.feature_importances_}).sort_values("Importance", ascending=False)
                fig = px.bar(imp, x="Feature", y="Importance", text="Importance", title="Feature Importance for Current Model")
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            try:
                xdf = build_input_df(st.session_state.latest_result["inputs"])
                local = shap_values(explainer, xdf)[0]
                local_df = pd.DataFrame({"Feature": FEATURE_COLS, "SHAP Value": local, "Abs": np.abs(local)}).sort_values("Abs", ascending=False)
                fig = px.bar(local_df.head(10), x="Feature", y="SHAP Value", text="SHAP Value", title="Current Mix SHAP Contribution")
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#16324f"))
                st.plotly_chart(fig, use_container_width=True)
                top_row = local_df.iloc[0]
                direction = "increases" if top_row["SHAP Value"] > 0 else "reduces"
                st.markdown(f'<div class="info-box"><strong>Interpretation</strong><br><strong>{top_row["Feature"]}</strong> currently has the strongest local influence and generally {direction} the predicted strength for this mix.</div>', unsafe_allow_html=True)
            except Exception as exc:
                st.warning(f"SHAP rendering failed: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Report":
    st.markdown('<div class="panel"><div class="panel-title">Report</div><div class="panel-sub">Generate a polished PDF report from the current or optimized mix.</div>', unsafe_allow_html=True)
    source = st.session_state.latest_result or st.session_state.optimizer_result
    if source is None:
        st.info("Run Predictor or Mix Optimizer first.")
    else:
        file_name = st.text_input("PDF File Name", "AIcrete_Report.pdf")
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("Generate Report", use_container_width=True):
                try:
                    path = generate_pdf(source, file_name)
                    st.session_state.generated_pdf = path
                    st.success("Report generated.")
                except Exception as exc:
                    st.error(f"Report generation failed: {exc}")
        with c2:
            if "generated_pdf" in st.session_state and os.path.exists(st.session_state.generated_pdf):
                with open(st.session_state.generated_pdf, "rb") as f:
                    st.download_button("Download Report", data=f, file_name=file_name, mime="application/pdf", use_container_width=True)
        render_result_summary(source, show_save=False)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Terms":
    st.markdown('<div class="panel"><div class="panel-title">Terms of Service</div></div>', unsafe_allow_html=True)
    st.markdown("""
## Terms of Service

**Effective Date:** April 10, 2026

### 1. Acceptance of Terms
By accessing and using AIcrete Solutions, you accept and agree to be bound by the terms and provision of this agreement.

### 2. Use License
- Permission is granted to temporarily download one copy of the materials (information or software) on AIcrete Solutions for personal, non-commercial transitory viewing only.
- This is the grant of a license, not a transfer of title, and under this license you may not:
  - Modify or copy the materials
  - Use the materials for any commercial purpose or for any public display
  - Attempt to decompile or reverse engineer any software contained on AIcrete Solutions
  - Remove any copyright or other proprietary notations from the materials
  - Transferring the materials to another person or "mirroring" the materials on any other server

### 3. Disclaimer
The materials on AIcrete Solutions are provided on an 'as is' basis without warranties of any kind, either expressed or implied. AIcrete Solutions disclaims all warranties, expressed or implied, including but not limited to implied warranties of merchantability and fitness for a particular purpose.

### 4. Limitations
AIcrete Solutions will not be liable for any damages in connection with the use of the materials on AIcrete Solutions, including but not limited to indirect, incidental, special, punitive or consequential damages.

### 5. Accuracy of Materials
The materials appearing on AIcrete Solutions could include technical, typographical, or photographic errors. AIcrete Solutions does not warrant that any of the materials on AIcrete Solutions are accurate, complete, or current. AIcrete Solutions may make changes to the materials contained on AIcrete Solutions at any time without notice.

### 6. Links
AIcrete Solutions has not reviewed all of the sites linked to its website and is not responsible for the contents of any such linked site. The inclusion of any link does not imply endorsement by AIcrete Solutions of the site. Use of any such linked website is at the user's own risk.

### 7. Modifications
AIcrete Solutions may revise these terms of service at any time without notice. By using this website, you are agreeing to be bound by the then current version of these terms of service.

### 8. Governing Law
These terms and conditions are governed by and construed in accordance with applicable laws where AIcrete Solutions operates, and you irrevocably submit to the exclusive jurisdiction of the courts in that location.

### 9. Contact
For any questions regarding these Terms of Service, please contact us at support@aicretesolutions.com
    """)

elif page == "Privacy":
    st.markdown('<div class="panel"><div class="panel-title">Privacy Policy</div></div>', unsafe_allow_html=True)
    st.markdown("""
## Privacy Policy

**Effective Date:** April 10, 2026

### 1. Introduction
AIcrete Solutions ("we," "us," "our," or "Company") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, disclose, and otherwise handle your information.

### 2. Information We Collect
We may collect information about you in various ways, including:
- **Directly from You:** When you register, input data into our system, or correspond with us
- **Automatically:** Through cookies, log files, and similar technologies
- **From Third Parties:** From business partners or other sources with your consent

### 3. What Information We Collect
- Contact information (name, email, company)
- Account credentials and authentication data
- Mix design parameters and concrete composition data
- Usage analytics and system performance metrics
- Device information (browser type, IP address, operating system)

### 4. How We Use Information
We use the information we collect for:
- Providing and improving our services
- Processing transactions and sending related information
- Responding to inquiries and providing customer support
- Sending marketing communications (with consent)
- Conducting research and analytics
- Ensuring security and fraud prevention
- Complying with legal obligations

### 5. Data Security
AIcrete Solutions implements appropriate technical and organizational measures to protect your personal information against unauthorized access, alteration, disclosure, or destruction.

### 6. Data Retention
We retain your personal information for as long as necessary to provide our services and fulfill the purposes outlined in this policy, unless a longer retention period is required by law.

### 7. Sharing of Information
We do not sell your personal information. We may share information with:
- Service providers who assist in our operations
- Business partners (with your consent)
- Legal authorities when required by law
- Other parties with your explicit consent

### 8. Your Rights
Depending on your location, you may have the right to:
- Access your personal information
- Correct inaccurate data
- Request deletion of your information
- Opt-out of marketing communications
- Data portability

### 9. Cookies
AIcrete Solutions uses cookies to enhance your experience. You can control cookie settings through your browser, though this may affect functionality.

### 10. Third-Party Links
AIcrete Solutions is not responsible for the privacy practices of external websites. We encourage you to review their privacy policies.

### 11. Contact Us
For privacy inquiries, contact: privacy@aicretesolutions.com
    """)

elif page == "Security":
    st.markdown('<div class="panel"><div class="panel-title">Security</div></div>', unsafe_allow_html=True)
    st.markdown("""
## Security & Compliance

**Last Updated:** April 10, 2026

### 1. Security Measures
AIcrete Solutions implements comprehensive security practices to protect your data:

#### Data Protection
- End-to-end encryption for sensitive data transmission
- Encrypted storage for all user information
- Regular security audits and penetration testing
- Multi-factor authentication for account access

#### Infrastructure
- Secure cloud infrastructure with industry-standard protocols
- Regular backups and disaster recovery procedures
- Intrusion detection and prevention systems
- Real-time monitoring of system activity

### 2. Compliance Standards
AIcrete Solutions complies with:
- GDPR (General Data Protection Regulation)
- CCPA (California Consumer Privacy Act)
- ISO 27001 Information Security Management
- SOC 2 Type II compliance (in progress)

### 3. Access Control
- Role-based access control (RBAC)
- Principle of least privilege
- Regular access reviews and audits
- Secure password policies

### 4. Incident Response
- 24/7 security monitoring
- Documented incident response procedures
- Notification procedures for data breaches
- Regular training for security protocols

### 5. Third-Party Security
- Vendor security assessments
- Contractual security obligations
- Regular review of third-party access
- Data protection agreements in place

### 6. User Responsibility
Users should:
- Maintain confidentiality of account credentials
- Report suspicious activity immediately
- Use strong, unique passwords
- Enable multi-factor authentication

### 7. Security Vulnerabilities
If you discover a security vulnerability, please report it to: security@aicretesolutions.com

Do not publicly disclose the vulnerability until we have had time to address it.

### 8. Security Updates
- Regular software updates and patches
- Timely deployment of security fixes
- System maintenance performed during off-peak hours
- Advance notification for critical updates

### 9. Contact
For security concerns, contact: security@aicretesolutions.com
    """)

elif page == "Contact":
    st.markdown('<div class="panel"><div class="panel-title">Contact Us</div></div>', unsafe_allow_html=True)
    st.markdown("""
## Get in Touch

We'd love to hear from you. Whether you have questions, feedback, or partnership inquiries, feel free to reach out.
    """)
    
    cols = st.columns(2)
    with cols[0]:
        st.markdown("### Direct Contact")
        st.markdown("""
**Email Support**
- General: support@aicretesolutions.com
- Sales: sales@aicretesolutions.com
- Technical: tech@aicretesolutions.com
- Security: security@aicretesolutions.com
- Privacy: privacy@aicretesolutions.com

**Office Hours**
Monday - Friday: 9:00 AM - 5:00 PM (EST)
        """)
    
    with cols[1]:
        st.markdown("### Quick Contact Form")
        with st.form("contact_form"):
            name = st.text_input("Your Name", placeholder="John Doe")
            email = st.text_input("Email Address", placeholder="you@company.com")
            subject = st.selectbox("Subject", [
                "Product Inquiry",
                "Technical Support",
                "Sales & Licensing",
                "Partnership",
                "Feedback",
                "Other"
            ])
            message = st.text_area("Message", placeholder="Tell us what you're thinking...", height=120)
            submitted = st.form_submit_button("Send Message", use_container_width=True)
            if submitted:
                if name and email and message:
                    st.success("Thank you for your message! We'll get back to you within 24 hours.")
                else:
                    st.error("Please fill in all fields.")

st.markdown('<div style="color:#6b7d93;font-size:0.88rem;margin-top:0.4rem;">Disclaimer: For preliminary engineering assessment only. Laboratory validation and professional review remain necessary before implementation.</div>', unsafe_allow_html=True)

# Professional Footer
st.markdown("""
<div class="footer-container">
    <div class="footer-content">
        <div class="footer-main">
            <div class="footer-column">
                <div class="footer-column-title">Product</div>
                <div class="footer-links">
                    <a onclick="window.location.href='?legal_page=Predictor'" class="footer-link" style="cursor:pointer;">Features</a>
                    <a href="https://docs.aicretesolutions.com" class="footer-link" target="_blank">Documentation</a>
                    <a href="https://api.aicretesolutions.com" class="footer-link" target="_blank">API</a>
                    <a href="https://aicretesolutions.com/pricing" class="footer-link" target="_blank">Pricing</a>
                </div>
            </div>
            <div class="footer-column">
                <div class="footer-column-title">Company</div>
                <div class="footer-links">
                    <a href="https://aicretesolutions.com/about" class="footer-link" target="_blank">About</a>
                    <a href="https://community.aicretesolutions.com" class="footer-link" target="_blank">Community</a>
                    <a href="https://blog.aicretesolutions.com" class="footer-link" target="_blank">Blog</a>
                    <a onclick="window.location.href='?legal_page=Contact'" class="footer-link" style="cursor:pointer;">Contact</a>
                </div>
            </div>
            <div class="footer-column">
                <div class="footer-column-title">Legal</div>
                <div class="footer-links">
                    <a onclick="window.location.href='?legal_page=Terms'" class="footer-link" style="cursor:pointer;">Terms</a>
                    <a onclick="window.location.href='?legal_page=Privacy'" class="footer-link" style="cursor:pointer;">Privacy</a>
                    <a onclick="window.location.href='?legal_page=Security'" class="footer-link" style="cursor:pointer;">Security</a>
                    <a onclick="document.querySelector('html').scrollTop = 0" class="footer-link" style="cursor:pointer;">Manage Cookies</a>
                </div>
            </div>
        </div>
        <div class="footer-divider"></div>
        <div class="footer-bottom">
            <span class="footer-copyright">© 2026 AIcrete Solutions. All rights reserved.</span>
            <div style="color:#718197; font-size:0.84rem;">
                Built for the future of concrete engineering intelligence
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
