import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from fpdf import FPDF
from io import BytesIO

st.set_page_config(page_title="ML Model Evaluator Pro", layout="wide")

# Helper functions
def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download {filename}</a>'

def generate_pdf_report(report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=report_text)
    return bytes(pdf.output(dest='S'))

# UI Setup
st.title("🕵️ AI Model Performance Inspector Pro")
st.markdown("### Comprehensive model evaluation toolkit with interactive insights")

with st.expander("📤 Step 1: Upload Data", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        true_file = st.file_uploader("True Labels (CSV)", type=["csv"], key="true")
    with col2:
        pred_files = st.file_uploader("Prediction Files (CSV)", type=["csv"], 
                                    accept_multiple_files=True, key="pred")

if true_file and pred_files:
    # Data Processing
    true_df = pd.read_csv(true_file)
    true_df.columns = [col.lower().strip() for col in true_df.columns]
    true_df.rename(columns={'id': 'Id', 'label': 'Label', 'labels': 'Label'}, inplace=True)

    if not {'Id', 'Label'}.issubset(true_df.columns):
        st.error("❌ True labels file must contain 'Id' and 'Label' columns")
        st.stop()

    if len(pred_files) > 1:
        num_files = st.slider(
            "Number of prediction files to analyze",
            min_value=1,
            max_value=len(pred_files),
            value=len(pred_files),
            help="Select how many prediction files to evaluate"
        )
    else:
        num_files = 1
        st.write("📁 Analyzing the uploaded prediction file")

    # Initialize comparison metrics
    all_metrics = []
    
    # Process all selected files
    for file_idx in range(num_files):
        pred_df = pd.read_csv(pred_files[file_idx])
        pred_df.columns = [col.lower().strip() for col in pred_df.columns] 
        pred_df.rename(columns={'id': 'Id', 'label': 'Label', 'labels': 'Label'}, inplace=True)
 
        # Create expandable section for each file
        with st.expander(f"📊 Analysis for: {pred_files[file_idx].name}", expanded=True):
            if not {'Id', 'Label'}.issubset(pred_df.columns):
                st.error(f"Invalid format in {pred_files[file_idx].name}")
                continue

            merged_df = pd.merge(true_df, pred_df, on='Id', suffixes=('_True', '_Pred'), how='left')
            discrepancies = merged_df[merged_df['Label_True'] != merged_df['Label_Pred']]
            
            # Calculate Metrics with 4 decimal precision
            accuracy = accuracy_score(merged_df['Label_True'], merged_df['Label_Pred'])
            cm = confusion_matrix(merged_df['Label_True'], merged_df['Label_Pred'])
            report_text = classification_report(merged_df['Label_True'], merged_df['Label_Pred'], digits=4)
            report_df = pd.DataFrame(classification_report(
                merged_df['Label_True'], merged_df['Label_Pred'], output_dict=True, digits=4)).transpose()

            # Store metrics for comparison
            all_metrics.append({
                'File': pred_files[file_idx].name,
                'Accuracy': accuracy,
                'Mismatches': len(discrepancies)
            })

            # Results Display
            col_metrics, col_cm, col_report = st.columns([1, 2, 3])
            
            with col_metrics:
                st.metric("Accuracy", f"{accuracy:.4%}")  
                st.metric("Mismatches", len(discrepancies))
                
                csv = discrepancies.to_csv(index=False).encode()
                st.markdown(create_download_link(csv, f"mismatches_{file_idx+1}.csv"), 
                          unsafe_allow_html=True)

            with col_cm:
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                          xticklabels=np.unique(merged_df['Label_True']),
                          yticklabels=np.unique(merged_df['Label_True']))
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Labels')
                plt.ylabel('True Labels')
                
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                st.image(buf, use_column_width=True)
                st.download_button(
                    f"Download Confusion Matrix {file_idx+1}",
                    data=buf,
                    file_name=f"confusion_matrix_{file_idx+1}.png",
                    mime="image/png",
                    key=f"cm_dl_{file_idx}"
                )
                plt.close()

            with col_report:
                st.dataframe(report_df.style.format("{:.4f}")  # 4 decimal places
                           .background_gradient(cmap='Blues', axis=0))
                
                col_pdf, col_txt = st.columns(2)
                with col_pdf:
                    pdf_report = generate_pdf_report(report_text)
                    st.download_button(
                        f"📥 PDF Report {file_idx+1}",
                        data=pdf_report,
                        file_name=f"report_{file_idx+1}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{file_idx}"
                    )
                with col_txt:
                    st.download_button(
                        f"📥 TXT Report {file_idx+1}",
                        data=report_text,
                        file_name=f"report_{file_idx+1}.txt",
                        mime="text/plain",
                        key=f"txt_{file_idx}"
                    )

            
            # with st.expander("🔍 Detailed Mismatch Analysis", expanded=False):
            st.markdown("🔍 Detailed Mismatch Analysis")
            with st.container():
                col_filter, col_preview = st.columns([2, 3])
                
                with col_filter:
                    selected_class = st.multiselect(
                        "Filter by true class:", 
                        options=merged_df['Label_True'].unique(),
                        key=f"filter_{file_idx}"
                    )
                    filtered_disc = discrepancies[discrepancies['Label_True'].isin(selected_class)] if selected_class else discrepancies
                    st.dataframe(filtered_disc.style.applymap(
                        lambda x: "background-color: #2E5A88",
                        subset=['Label_True', 'Label_Pred']
                    ))

                with col_preview:
                    if not filtered_disc.empty:
                        sample = filtered_disc.sample(min(3, len(filtered_disc)))
                        for _, row in sample.iterrows():
                            st.markdown(f"""
                            - **ID {row['Id']}**:  
                              True: `{row['Label_True']}` → Predicted: `{row['Label_Pred']}`
                            """)

    # Model Comparison Section
    if len(all_metrics) > 1:
        st.subheader("🏆 Model Comparison")
        comparison_df = pd.DataFrame(all_metrics)
        st.dataframe(
            comparison_df.sort_values('Accuracy', ascending=False)
            .style.format({'Accuracy': "{:.4%}", 'Mismatches': "{:,}"})  # 4 decimal places
            .background_gradient(cmap='YlGnBu', subset=['Accuracy'])
        )

else:
    st.info("👋 Please upload both true labels and prediction files to begin analysis")

st.markdown("---")
st.caption("🔒 All data is processed in memory and never stored. Built with Streamlit 🎈")