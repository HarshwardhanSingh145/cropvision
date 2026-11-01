import io
import base64
import pandas as pd
import streamlit as st
from PIL import Image
from engine import analyze_crop_leaf, get_prediction_history

st.set_page_config(
    page_title="CropVision AI - Leaf Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    .diagnosis-header {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 2px solid #38bdf8;
        border-radius: 12px;
        padding: 22px 26px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
    }
    .diagnosis-title-tag {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .diagnosis-main-name {
        font-size: 32px;
        font-weight: 900;
        color: #38bdf8;
        margin-bottom: 8px;
    }
    .diagnosis-meta {
        font-size: 15px;
        color: #cbd5e1;
    }
    
    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        padding: 16px 20px !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] div,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: #f8fafc !important;
        font-size: 20px !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
        color: #38bdf8 !important;
    }
    
    .status-card {
        padding: 16px 20px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 15px;
        font-size: 15px;
        font-weight: 600;
        border: 1px solid;
    }
    .status-success {
        background-color: #064e3b;
        border-color: #059669;
        color: #ecfdf5;
    }
    .status-warning {
        background-color: #78350f;
        border-color: #d97706;
        color: #fffbeb;
    }
    .status-danger {
        background-color: #7f1d1d;
        border-color: #dc2626;
        color: #fef2f2;
    }
</style>
""", unsafe_allow_html=True)

def render_history_table(limit=15):
    records = get_prediction_history(limit=limit)
    if records:
        df = pd.DataFrame(
            records, 
            columns=["Timestamp", "Filename", "Disease", "Confidence (%)", "Severity (%)", "Stage", "Status"]
        )
        st.dataframe(df, width="stretch")
    else:
        st.info("No predictions recorded yet.")

st.title("CropVision AI - Crop Disease Diagnosis & Severity Assessment")
st.caption("Computer vision pipeline combining EfficientNet-B0 classification, Grad-CAM explainability, and U-Net segmentation.")

tab_diagnose, tab_benchmark, tab_history = st.tabs([
    "Leaf Diagnosis", 
    "CNN Benchmark", 
    "Prediction History"
])

crop_options = [
    "Auto-Detect (All 14 Crops)",
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn (Maize)",
    "Grape",
    "Orange",
    "Peach",
    "Pepper Bell",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato"
]

with tab_diagnose:
    # 2 Important Disclaimers
    st.markdown("""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px; margin-bottom: 22px;">
        <div style="background: #0f172a; padding: 14px 18px; border-radius: 8px; border: 1px solid #1e293b; border-left: 4px solid #38bdf8;">
            <div style="font-weight: 700; color: #38bdf8; font-size: 14px; margin-bottom: 6px;">Important Industry Scope</div>
            <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5;">
                This system is exclusively trained for <b>14 agricultural field & orchard crops</b> (Apple, Tomato, Potato, Corn, Grape, Bell Pepper, Peach, Strawberry, Cherry, Squash, Soybean, Blueberry, Raspberry, Orange). <b>Do not use on indoor houseplants, ornamental plants, or succulents.</b>
            </div>
        </div>
        <div style="background: #0f172a; padding: 14px 18px; border-radius: 8px; border: 1px solid #1e293b; border-left: 4px solid #10b981;">
            <div style="font-weight: 700; color: #10b981; font-size: 14px; margin-bottom: 6px;">Photo Upload Guidelines</div>
            <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5;">
                Upload a <b>front-facing photograph with clear, sharp edges</b> and balanced natural lighting. Ensure the single leaf occupies the primary area of the photo. Avoid heavy shadows, extreme angles, or blurry captures.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Upload Crop Leaf Image")
    st.write("Select the crop family or keep Auto-Detect, then upload your leaf photograph:")
    
    col_u1, col_u2 = st.columns([1, 2])
    with col_u1:
        selected_crop = st.selectbox(
            "Plant / Crop Domain",
            crop_options,
            index=0,
            help="Locking to the specific crop variety ensures targeted diagnostic accuracy."
        )
    with col_u2:
        uploaded_file = st.file_uploader(
            "Choose a leaf image file", 
            type=["jpg", "jpeg", "png", "webp"],
            key="leaf_uploader"
        )

    if uploaded_file is not None:
        target_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name

        with st.spinner("Processing image through neural pipeline..."):
            result = analyze_crop_leaf(target_bytes, filename=filename, crop_filter=selected_crop)

        if not result["valid"]:
            if result.get("unsupported_leaf"):
                st.markdown(f"""
                <div class="status-card status-danger" style="background-color: #450a0a; border-color: #ef4444; color: #fef2f2; padding: 18px 22px;">
                    <div style="font-size: 18px; font-weight: 800; color: #f87171; margin-bottom: 8px;">Unsupported Plant / Indoor Leaf Detected</div>
                    <div style="font-size: 14px; line-height: 1.6;">{result['reason']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-card status-danger">
                    Image Quality Check Failed: {result['reason']}
                </div>
                """, unsafe_allow_html=True)
        else:
            if "Uncertain" in result["status"]:
                st.markdown(f"""
                <div class="diagnosis-header" style="border-color: #d97706;">
                    <div class="diagnosis-title-tag" style="color: #fbbf24;">Low Confidence Flag ({result['confidence']}%)</div>
                    <div class="diagnosis-main-name" style="color: #fbbf24;">Suspected: {result['disease']}</div>
                    <div class="diagnosis-meta">
                        <b>Status:</b> {result['status']} &nbsp;|&nbsp; 
                        <b>Severity:</b> {result['severity_pct']}% ({result['severity_stage']} Stage)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif "Healthy" in result["status"]:
                st.markdown(f"""
                <div class="diagnosis-header" style="border-color: #10b981;">
                    <div class="diagnosis-title-tag" style="color: #34d399;">Diagnosis Result</div>
                    <div class="diagnosis-main-name" style="color: #34d399;">{result['disease']}</div>
                    <div class="diagnosis-meta">
                        <b>Confidence:</b> {result['confidence']}% &nbsp;|&nbsp; 
                        <b>Status:</b> Verified Healthy Foliage
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="diagnosis-header">
                    <div class="diagnosis-title-tag">Diagnosis Result</div>
                    <div class="diagnosis-main-name">{result['disease']}</div>
                    <div class="diagnosis-meta">
                        <b>Confidence:</b> {result['confidence']}% &nbsp;|&nbsp; 
                        <b>Severity:</b> {result['severity_pct']}% ({result['severity_stage']} Stage) &nbsp;|&nbsp; 
                        <b>Status:</b> {result['status']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Disease", result["disease"])
            m2.metric("Confidence", f"{result['confidence']}%", f"Uncertainty: {result['uncertainty']}%")
            m3.metric("Severity", f"{result['severity_pct']}%", f"Stage: {result['severity_stage']}")
            m4.metric("Lesion Spots", f"{result['spots_detected']} Detected", f"Sharpness: {result['blur_score']}")

            if "Uncertain" in result["status"]:
                st.markdown(f"""
                <div class="status-card status-warning">
                    {result['status']}: Confidence is below 45%. If this is a specific crop, please select the exact crop type from the dropdown above.
                </div>
                """, unsafe_allow_html=True)
            elif "Healthy" in result["status"]:
                st.markdown(f"""
                <div class="status-card status-success">
                    {result['status']}: Foliage tissue is confirmed healthy with no active necrotic lesions.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-card status-success">
                    {result['status']}: Pathological lesions isolated and logged to database.
                </div>
                """, unsafe_allow_html=True)

            if "top_predictions" in result and len(result["top_predictions"]) > 1:
                with st.expander("Top Predictions Breakdown", expanded=False):
                    for rank, candidate in enumerate(result["top_predictions"], 1):
                        st.write(f"**Rank {rank}:** {candidate['disease']} — `{candidate['confidence']}%`")

            st.divider()
            st.subheader("Visual Analysis Pipeline")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.image(
                Image.open(io.BytesIO(target_bytes)), 
                caption="1. Original Upload", 
                width="stretch"
            )
            c2.image(
                base64.b64decode(result["yolo_img"]), 
                caption=f"2. Lesion Localization ({result['spots_detected']} Spots)", 
                width="stretch"
            )
            c3.image(
                base64.b64decode(result["gradcam_img"]), 
                caption="3. Grad-CAM Attention Map", 
                width="stretch"
            )
            c4.image(
                base64.b64decode(result["seg_img"]), 
                caption=f"4. Lesion Segmentation ({result['severity_pct']}%)", 
                width="stretch"
            )

            st.divider()
            st.subheader("Recent Predictions (Database)")
            render_history_table(limit=5)
    else:
        st.info("Select a crop variety (optional) and upload a leaf image above to view diagnosis and segmentation results.")
        st.divider()
        st.subheader("Recent Predictions (Database)")
        render_history_table(limit=5)

with tab_benchmark:
    st.subheader("Model Benchmark Comparison")
    st.write("Performance evaluation of transfer learning backbones tested on the dataset:")
    try:
        df_bench = pd.read_csv("cnn_benchmark.csv")
        st.dataframe(df_bench, width="stretch")
    except Exception as e:
        st.warning(f"Could not load benchmark data: {e}")

with tab_history:
    st.subheader("Prediction History Database")
    st.write("Saved records from the SQLite database:")
    render_history_table(limit=30)
