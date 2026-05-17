import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import io
import time
from datetime import datetime
from google import genai
from google.genai import types

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="NeuroScan Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# UI STYLE OPTIMIZATIONS (UNIFIED BUTTON STYLING)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Main Layout Fixes */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #090D16 !important;
    color: #F3F4F6 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Sidebar Text & Label Brightness Fix */
[data-testid="stSidebar"] {
    background-color: #0E1524 !important;
    border-right: 1px solid #1F293D;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
    color: #E5E7EB !important;
}

/* High-Contrast Chat Typography Fixes */
[data-testid="stChatMessage"] {
    color: #F3F4F6 !important;
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    margin-bottom: 10px;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2 {
    color: #FFFFFF !important;
}

/* File Uploader Accentuation & Visibility */
[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.3) !important;
    border: 1px dashed #3B82F6 !important;
    border-radius: 12px;
    padding: 10px;
}
/* Fix the text label color inside the upload container */
[data-testid="stFileUploader"] section {
    color: #FFFFFF !important;
}
[data-testid="stFileUploader"] div {
    color: #E5E7EB !important;
}
/* Style the actual browse files button inside the uploader to match your theme */
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 10px rgba(59, 130, 246, 0.2) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    transform: translateY(-1px) !important;
    background: linear-gradient(135deg, #60A5FA 0%, #2563EB 100%) !important;
    box-shadow: 0 6px 14px rgba(59, 130, 246, 0.3) !important;
}

/* Custom Card Container */
.glass-card {
    background: rgba(19, 29, 49, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
}

/* Text overrides within Glass Cards */
.glass-card h3, .glass-card h4, .glass-card p {
    color: #FFFFFF !important;
}

/* Chat scrollable height constraints */
.chat-scroll-area {
    max-height: 480px;
    overflow-y: auto;
    padding-right: 8px;
    margin-bottom: 15px;
}

/* Metrics Displays */
.metric-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 16px;
    border-radius: 12px;
    margin-top: 12px;
}
.metric-label {
    font-size: 14px;
    color: #9CA3AF !important;
    font-weight: 500;
}
.metric-value {
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF !important;
}

/* Premium Gradient Button Layout (Unified for Action & Download types) */
.stButton > button, .stDownloadButton > button {
    width: 100% !important;
    height: 48px;
    background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border-radius: 10px !important;
    border: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3) !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
    background: linear-gradient(135deg, #60A5FA 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
}

/* Informative Row Elements */
.info-row {
    display: flex;
    align-items: start;
    gap: 12px;
    padding: 12px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.01);
    border: 1px solid rgba(255, 255, 255, 0.03);
    margin-bottom: 8px;
}
.info-row-icon {
    font-size: 18px;
}
.info-row-text strong {
    color: #60A5FA !important;
}
.info-row-text p {
    color: #9CA3AF !important;
}

/* Target Highlights */
.badge-glioma { background: rgba(249, 115, 22, 0.15); color: #FB923C; border: 1px solid rgba(249, 115, 22, 0.3); padding: 6px 14px; border-radius: 8px; font-weight: 700; }
.badge-meningioma { background: rgba(244, 63, 94, 0.15); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.3); padding: 6px 14px; border-radius: 8px; font-weight: 700; }
.badge-pituitary { background: rgba(139, 92, 246, 0.15); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.3); padding: 6px 14px; border-radius: 8px; font-weight: 700; }
.badge-healthy { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 6px 14px; border-radius: 8px; font-weight: 700; }

</style>
""", unsafe_allow_html=True)

# =========================
# INITIALIZE LIVE LLM CLIENT
# =========================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    ai_client = genai.Client(api_key=api_key)
except Exception as e:
    st.error("Missing or broken GEMINI_API_KEY in your Streamlit Secrets Configuration.")
    ai_client = None

# =========================
# MODEL CONFIGURATION
# =========================
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

@st.cache_resource
def load_model():
    model = CNN()
    try:
        model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
    except:
        pass
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

def predict(img):
    img = Image.open(img).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        p = torch.softmax(out, 1)[0]
        conf, pred = torch.max(p, 0)
    return CLASS_NAMES[pred], float(conf), {CLASS_NAMES[i]: float(p[i]) for i in range(4)}

# =========================
# STATE INITIALIZATIONS
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the Neural Diagnostic Console. I am ready to evaluate clinical data anomalies, classification spreads, or general pathology indices. How can I assist you today?"}
    ]

if "active_pred" not in st.session_state: st.session_state["active_pred"] = "Not Analyzed Yet"
if "active_conf" not in st.session_state: st.session_state["active_conf"] = 0.0

# =========================
# SIDEBAR / CONFIGURATION PANEL
# =========================
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0px; color:#FFFFFF;'>🧠 NeuroScan AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9CA3AF; font-size:13px; margin-bottom:24px;'>Clinical Decision Support System</p>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:12px; font-weight:600; text-transform:uppercase; color:#9CA3AF; margin-bottom:8px;'>Metadata Input</p>", unsafe_allow_html=True)
    patient = st.text_input("Patient Identifier", value="PT-09241", label_visibility="collapsed")
    
    st.markdown("<p style='font-size:12px; font-weight:600; text-transform:uppercase; color:#9CA3AF; margin-top:16px; margin-bottom:8px;'>Modality</p>", unsafe_allow_html=True)
    scan_type = st.selectbox("Scan Type", ["MRI (T1-Weighted)", "MRI (T2-Weighted)", "CT Scan"], label_visibility="collapsed")
    
    st.markdown("<hr style='border-color:#1F293D; margin:24px 0;' />", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:12px; font-weight:600; text-transform:uppercase; color:#9CA3AF; margin-bottom:8px;'>Data Ingestion</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload DICOM/Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded:
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("Execute Neural Diagnostics")
        if analyze_btn:
            st.session_state['run_analysis'] = True

# =========================
# MAIN DASHBOARD INTERFACE
# =========================
tab_dashboard, tab_encyclopedia, tab_chat = st.tabs([
    "📊 Diagnostic Workstation", 
    "📖 Tumor Pathology Directory",
    "🤖 AI Consultation Desk"
])

with tab_dashboard:
    if uploaded:
        col_img, col_metrics = st.columns([1.2, 1], gap="large")
        
        with col_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='margin-top:0; margin-bottom:16px; font-size:16px; color:#9CA3AF;'>Active Scan Geometry</h4>", unsafe_allow_html=True)
            st.image(uploaded, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_metrics:
            st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("<h4 style='margin-top:0; margin-bottom:16px; font-size:16px; color:#9CA3AF;'>Inference Diagnostics</h4>", unsafe_allow_html=True)
            
            if st.session_state.get('run_analysis'):
                with st.spinner("Processing tensors through classification heads..."):
                    progress_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.005)
                        progress_bar.progress(percent_complete + 1)
                    
                    pred, conf, probs = predict(uploaded)
                    st.session_state["active_pred"] = pred
                    st.session_state["active_conf"] = conf * 100
                    
                badge_mapping = {
                    "Glioma": f'<span class="badge-glioma">GLIOMA DETECTED</span>',
                    "Meningioma": f'<span class="badge-meningioma">MENINGIOMA DETECTED</span>',
                    "Pituitary": f'<span class="badge-pituitary">PITUITARY ADENOMA DETECTED</span>',
                    "No Tumor": f'<span class="badge-healthy">NEGATIVE FOR NEOPLASM</span>'
                }
                badge_html = badge_mapping.get(st.session_state["active_pred"], st.session_state["active_pred"])
                
                st.markdown(f"""
                <div style='margin-bottom: 24px;'>
                    <p style='font-size:12px; text-transform:uppercase; color:#9CA3AF; margin-bottom:6px;'>System Classification</p>
                    {badge_html}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-container">
                    <span class="metric-label">Target Confidence</span>
                    <span class="metric-value" style="color: #38BDF8;">{st.session_state["active_conf"]:.2f}%</span>
                </div>
                <div class="metric-container">
                    <span class="metric-label">Assigned Case File</span>
                    <span class="metric-value">{patient}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<p style='font-size:12px; text-transform:uppercase; color:#9CA3AF; margin-top:24px; margin-bottom:8px;'>Probability Spread</p>", unsafe_allow_html=True)
                
                df = pd.DataFrame({
                    "Classification Cluster": list(probs.keys()),
                    "Certainty (%)": [v * 100 for v in probs.values()]
                }).sort_values("Certainty (%)", ascending=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df["Certainty (%)"],
                    y=df["Classification Cluster"],
                    orientation="h",
                    marker=dict(
                        color=df["Certainty (%)"],
                        colorscale=[[0, '#1E293B'], [0.5, '#3B82F6'], [1, '#60A5FA']],
                        line=dict(color='rgba(0,0,0,0)', width=0)
                    ),
                    text=[f" {v:.1f}%" for v in df["Certainty (%)"]],
                    textposition='outside',
                    cliponaxis=False
                ))
                
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9CA3AF", family="Plus Jakarta Sans"),
                    height=200,
                    margin=dict(l=10, r=50, t=10, b=10),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                report = io.StringIO()
                report.write(f"NEUROSCAN CDSS TELEMETRY REPORT\n{'='*30}\n")
                report.write(f"Timestamp: {datetime.utcnow().isoformat()}Z\n")
                report.write(f"Subject Ref ID: {patient}\n")
                report.write(f"Modality Context: {scan_type}\n")
                report.write(f"Primary Diagnosis: {st.session_state['active_pred']}\n")
                report.write(f"Statistical Weight: {st.session_state['active_conf']:.4f}%\n")
                
                st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
                
                # Custom CSS handles this down button seamlessly now
                st.download_button(
                    label="📄 Export Clinical Documentation Asset",
                    data=report.getvalue(),
                    file_name=f"NeuroScan_{patient}.txt",
                    mime="text/plain"
                )
            else:
                st.info("Configuration loaded. Select 'Execute Neural Diagnostics' from the lateral sidebar to request model evaluation.")
                
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 80px 20px; border: 2px dashed rgba(255,255,255,0.05); border-radius:16px; background: rgba(255,255,255,0.01);">
            <span style="font-size: 48px;">🧠</span>
            <h3 style="margin-top: 16px; color: #FFFFFF;">No Structural Scan Selected</h3>
            <p style="color: #9CA3AF; max-width: 420px; margin: 8px auto 0 auto; font-size: 14px;">
                Please ingest an imaging subject from the side navigation platform to activate neural diagnostics, feature mappings, and reporting arrays.
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab_encyclopedia:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#FFFFFF;'>Histological Classification Index</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9CA3AF; margin-bottom:24px;'>Reference criteria for categorical structural variations tracked by the current network architecture layer.</p>", unsafe_allow_html=True)
    
    diagnostics_knowledge_base = [
        ("🧬", "Glioma Spectrum", "Malignant primary tumors arising directly from structural glial configurations inside the central nervous system framework."),
        ("🧠", "Meningioma Formation", "Typically localized, benign primary growths emerging along the protective meningeal membranes encapsulating spinal and cerebral masses."),
        ("📍", "Pituitary Adenoma", "Neoplastic growths localized on the pituitary glandular apparatus, driving localized cranial mass effects and neuroendocrine variances."),
        ("⚪", "Unremarkable Tissue Matrix (No Tumor)", "Represents structural configurations free of identifiable diagnostic neoplastic metrics or mass displacement anomalies.")
    ]
    
    for icon, title, desc in diagnostics_knowledge_base:
        st.markdown(f"""
        <div class="info-row">
            <div class="info-row-icon">{icon}</div>
            <div class="info-row-text">
                <strong>{title}</strong>
                <p style="margin: 4px 0 0 0; font-size: 13px; color: #9CA3AF;">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 🤖 TAB 3: LIVE AI CHATBOT
# =========================
with tab_chat:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#FFFFFF;'>Co-Pilot Clinical Chatbot</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9CA3AF; margin-bottom:16px;'>Inquire about current patient metrics, tumor grading classifications, or request treatment documentation overviews with a real-time clinical model.</p>", unsafe_allow_html=True)
    
    if ai_client is None:
        st.warning("Chatbot disabled. Please add a valid `GEMINI_API_KEY` to your secrets management settings.")
    else:
        st.markdown('<div class="chat-scroll-area">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)

        if prompt := st.chat_input("Ask about this case (e.g., 'What are next steps for this patient classification?')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                case_context = f"""
                Active Patient Metadata Context:
                - Patient ID: {patient}
                - Modality Scan Type: {scan_type}
                - Embedded CNN Prediction: {st.session_state['active_pred']}
                - Neural Network Confidence Level: {st.session_state['active_conf']:.2f}%
                """
                
                system_instruction = (
                    "You are a professional, expert AI clinical co-pilot specializing in neurology and neuro-oncology. "
                    "Provide evidence-based medical summaries, grading schemas, and operational workflows. "
                    "Maintain a clinical, highly scientific, yet concise tone. "
                    f"Here is information on the case currently loaded in the dashboard:\n{case_context}"
                )
                
                try:
                    response_stream = ai_client.models.generate_content_stream(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.3
                        )
                    )
                    
                    full_response = ""
                    for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as api_err:
                    st.error(f"API Error encountered: {str(api_err)}")
                    
    st.markdown('</div>', unsafe_allow_html=True)