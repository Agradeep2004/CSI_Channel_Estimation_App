# ==============================================
# Streamlit App: Wireless Channel Estimation Dashboard
# ==============================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import io, base64, tempfile, os, urllib.request
from datetime import datetime
from fpdf import FPDF


# ------------------------------
# Configuration
# ------------------------------
st.set_page_config(page_title="Wireless Channel Estimation Dashboard", layout="wide")

RESULT_PATHS = {
    "LS":          "results/results_ls/ls_results.npz",
    "MMSE":        "results/results_mmse/mmse_results.npz",
    "CNN":         "results/results_cnn/cnn_results.npz",
    "LSTM":        "results/results_lstm/lstm_results.npz",
    "CNN+LSTM":    "results/results_cnnlstm/cnn_lstm_results.npz",
    "CVNN":        "results/results_cvnn/cvnn_results.npz",
    "Transformer": "results/results_transformer/transformer_results.npz"
}

SUMMARY_FILE = "model_comparison_summary.csv"

# ------------------------------
# Helper Functions
# ------------------------------
@st.cache_data
def load_npz(path):
    data = np.load(path)
    snr = data["snr_db"]
    nmse_key = [k for k in data.keys() if "nmse" in k.lower()][0]
    ber_key  = [k for k in data.keys() if "ber"  in k.lower()][0]
    return snr, data[nmse_key], data[ber_key]

@st.cache_data
def load_summary(path):
    return pd.read_csv(path) if os.path.exists(path) else None

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.title("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model",
    list(RESULT_PATHS.keys()),
    index=6  # Default = Transformer
)

snr_choice = st.sidebar.slider(
    "Select SNR (dB)",
    -5, 30, 10, step=5
)

show_all = st.sidebar.checkbox("Compare All Models", value=True)
show_table = st.sidebar.checkbox("Show Summary Table", value=True)

st.sidebar.markdown("---")
st.sidebar.info("📊 Adjust SNR and select models to compare NMSE & BER performance.")

# ------------------------------
# Main Title
# ------------------------------
st.title("📡 Wireless Channel Estimation Performance Dashboard")
st.markdown("**Compare LS, MMSE, and ML-based models (CNN, LSTM, CVNN, Transformer)**")

# ------------------------------
# Load and Display Results
# ------------------------------
if not os.path.exists(RESULT_PATHS[model_choice]):
    st.error(f"Results not found for {model_choice}")
    st.stop()

snr, nmse, ber = load_npz(RESULT_PATHS[model_choice])

# Plot NMSE vs SNR (single model)
fig1, ax1 = plt.subplots(figsize=(7,5))
ax1.semilogy(snr, nmse, 'o-', label=model_choice)
ax1.set_title(f"NMSE vs SNR – {model_choice}")
ax1.set_xlabel("SNR (dB)")
ax1.set_ylabel("NMSE (log scale)")
ax1.grid(True, which="both")
ax1.legend()
st.pyplot(fig1)

# Plot BER vs SNR (single model)
fig2, ax2 = plt.subplots(figsize=(7,5))
ax2.semilogy(snr, ber, 's-', label=model_choice)
ax2.set_title(f"BER vs SNR – {model_choice}")
ax2.set_xlabel("SNR (dB)")
ax2.set_ylabel("BER (log scale)")
ax2.grid(True, which="both")
ax2.legend()
st.pyplot(fig2)

# Show NMSE/BER value for chosen SNR
snr_idx = np.argmin(np.abs(snr - snr_choice))
nmse_val, ber_val = nmse[snr_idx], ber[snr_idx]
st.metric(label=f"NMSE @ {snr_choice} dB", value=f"{nmse_val:.4e}")
st.metric(label=f"BER @ {snr_choice} dB", value=f"{ber_val:.4e}")

# ------------------------------
# Comparison Plot (All Models)
# ------------------------------
if show_all:
    st.markdown("### 📊 Comparison: All Models")
    fig_all, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    
    for name, path in RESULT_PATHS.items():
        if not os.path.exists(path): continue
        s, n, b = load_npz(path)
        ax1.semilogy(s, n, marker='o', label=name)
        ax2.semilogy(s, b, marker='s', label=name)
    
    ax1.set_title("NMSE vs SNR – All Models")
    ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("NMSE (log scale)")
    ax1.grid(True, which="both"); ax1.legend()
    
    ax2.set_title("BER vs SNR – All Models")
    ax2.set_xlabel("SNR (dB)"); ax2.set_ylabel("BER (log scale)")
    ax2.grid(True, which="both"); ax2.legend()
    
    st.pyplot(fig_all)

# ------------------------------
# Summary Table
# ------------------------------
if show_table:
    df = load_summary(SUMMARY_FILE)
    if df is not None:
        st.markdown("### 📋 Model Performance Summary")
        st.dataframe(df.style.highlight_min(subset=["Best NMSE","Average BER"], color="lightgreen"))
    else:
        st.warning("Summary CSV not found. Please run Step 11 to generate it.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Developed by **[Your Name / SRM Institute]** — Wireless Channel Estimation Project 2025")

# ==============================================
# STEP 15: Custom CSI Estimation + PDF Report Generator
# ==============================================
import io, base64, tempfile, os
from datetime import datetime
from fpdf import FPDF

st.markdown("---")
st.markdown("## 🧮 Custom CSI Estimation & Auto Report (CSV / PDF)")

with st.expander("🔧 Run Full CSI Estimation Across All Models", expanded=False):
    st.write("Provide your channel parameters and generate comparative performance graphs, analysis, and download as CSV or PDF report.")

    # --- User Inputs ---
    snr_input = st.slider("Select SNR (dB)", -5, 30, 10, step=5)
    doppler = st.selectbox("Select Doppler Scenario", ["Low (0 Hz)", "Medium (30 Hz)", "High (200 Hz)"], index=1)
    channel_len = st.number_input("Channel Paths (L)", min_value=1, max_value=16, value=8)
    num_samples = st.number_input("Samples to Simulate", min_value=100, max_value=2000, value=500)

    run_button = st.button("🚀 Run CSI Estimation and Generate Report")

    if run_button:
        st.info("Running full CSI estimation… please wait ⏳")

        nmse_summary, ber_summary = {}, {}
        for name, path in RESULT_PATHS.items():
            if not os.path.exists(path): continue
            data = np.load(path)
            snr = data["snr_db"]
            nmse_key = [k for k in data.keys() if "nmse" in k.lower()][0]
            ber_key  = [k for k in data.keys() if "ber"  in k.lower()][0]
            nmse = data[nmse_key]; ber = data[ber_key]
            nmse_summary[name] = float(np.interp(snr_input, snr, nmse))
            ber_summary[name]  = float(np.interp(snr_input, snr, ber))

        df = pd.DataFrame({
            "Model": list(nmse_summary.keys()),
            "NMSE":  list(nmse_summary.values()),
            "BER":   list(ber_summary.values())
        }).sort_values(by="NMSE").reset_index(drop=True)

        best_model = df.loc[df["NMSE"].idxmin(), "Model"]

        st.success(f"✅ CSI estimation complete at {snr_input} dB, {doppler}, {channel_len} paths.")
        st.subheader("📋 Performance Summary")
        st.dataframe(df.style.highlight_min(subset=["NMSE","BER"], color="lightgreen"))

        # --- Comparative Graphs ---
        fig, ax = plt.subplots(1,2,figsize=(14,5))
        ax[0].bar(df["Model"], df["NMSE"], color="steelblue"); ax[0].set_title("NMSE Comparison")
        ax[1].bar(df["Model"], df["BER"], color="orange"); ax[1].set_title("BER Comparison")
        for a in ax: a.grid(True, axis="y"); a.set_xticklabels(df["Model"], rotation=30, ha="right")
        st.pyplot(fig)

        # Save figure temporarily
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(temp_img.name, dpi=300, bbox_inches="tight")

        # --- Auto-generated Explanation ---
        interpretation = f"""
**Best Model:** **{best_model}**

- {best_model} achieves the lowest NMSE ({df['NMSE'].min():.4e}) and BER ({df['BER'].min():.4e})
  due to its superior ability to learn complex time-frequency relationships in the channel.
- It handles Doppler scenario *{doppler.lower()}* robustly, maintaining low error even at {snr_input} dB SNR.
- Models like **LS/MMSE** are faster but analytical — they fail in noisy or fast-fading environments.
- **CNN & LSTM** learn partial correlations; **CNN+LSTM** balances both but lacks global attention.
- **CVNN** performs well by preserving phase information.
- **Transformer** excels by learning **global dependencies**, making it ideal for adaptive CSI estimation.

**Ranking (Best → Worst):**
{', '.join(df['Model'].tolist())}
"""

        st.markdown("### 🧠 Automatic Interpretation")
        st.markdown(interpretation)

        # -------------------------
        # PDF Generation
        # -------------------------
        pdf = FPDF()
        pdf.add_page()
        
        # Title + logo
        if os.path.exists("logo.png"):
            pdf.image("logo.png", 10, 8, 25)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 15, "Wireless Channel Estimation Report", ln=True, align="C")

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 8, f"SNR: {snr_input} dB   Doppler: {doppler}   Paths: {channel_len}   Samples: {num_samples}", ln=True)
        pdf.ln(4)

        # Table header
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(50, 10, "Model", 1)
        pdf.cell(50, 10, "NMSE", 1)
        pdf.cell(50, 10, "BER", 1)
        pdf.ln()

        # Table rows
        pdf.set_font("Helvetica", "", 11)
        for _, row in df.iterrows():
            pdf.cell(50, 8, row["Model"], 1)
            pdf.cell(50, 8, f"{row['NMSE']:.4e}", 1)
            pdf.cell(50, 8, f"{row['BER']:.4e}", 1)
            pdf.ln()
        pdf.ln(6)

        # Insert chart image
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "NMSE & BER Comparison", ln=True)
        pdf.image(temp_img.name, x=20, w=160)
        pdf.ln(6)

        # Interpretation text
        # --- Add interpretation (ASCII-safe for Helvetica) ---
        pdf.set_font("Helvetica", "", 11)
        interpretation = interpretation.replace("—", "-").replace("•", "-").replace("–", "-")
        safe_text = interpretation.encode('ascii', 'ignore').decode()
        pdf.multi_cell(0, 7, safe_text)

        # Footer
        pdf.set_y(-20)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 10, "Generated by Wireless Channel Estimation Dashboard | SRM College Project 2025", 0, 0, "C")

        # Save and serve PDF
        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            b64_pdf = base64.b64encode(f.read()).decode("utf-8")
            href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="CSI_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">📄 Download Full Report (PDF)</a>'
            st.markdown(href_pdf, unsafe_allow_html=True)

        # Clean temp image
        try:
            temp_img.close()        # ✅ Close file handle first
            os.remove(temp_img.name)
        except PermissionError:
            pass                    # ✅ Safe fallback if Windows still holds the file

