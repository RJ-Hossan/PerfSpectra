import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from fpdf import FPDF
from io import BytesIO

st.set_page_config(page_title="ML Model Evaluator Pro", layout="wide")

def initialize_label_mapping():
    if 'label_mapping' not in st.session_state:
        st.session_state.label_mapping = {}
    if 'num_labels' not in st.session_state:
        st.session_state.num_labels = 2

def update_num_labels():
    st.session_state.label_mapping = {
        i: st.session_state.label_mapping.get(i, f"Class {i}") 
        for i in range(st.session_state.num_labels)
    }

def get_label_name(label):
    return st.session_state.label_mapping.get(label, str(label))

def calculate_g_scores(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    g_scores = {}
    
    for i in range(n_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        g_score = np.sqrt(sensitivity * specificity)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = sensitivity
        
        g_score_pr = np.sqrt(precision * recall) if (precision > 0 and recall > 0) else 0
        
        g_scores[f"Class {i}"] = {
            'G-Score (Sens-Spec)': g_score,
            'G-Score (Prec-Rec)': g_score_pr,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'Recall': recall
        }
    
    overall_g_score = np.exp(np.mean([np.log(metrics['G-Score (Sens-Spec)']) 
                                    for metrics in g_scores.values() 
                                    if metrics['G-Score (Sens-Spec)'] > 0]))
    
    overall_g_score_pr = np.exp(np.mean([np.log(metrics['G-Score (Prec-Rec)']) 
                                       for metrics in g_scores.values() 
                                       if metrics['G-Score (Prec-Rec)'] > 0]))
    
    return g_scores, overall_g_score, overall_g_score_pr

def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download {filename}</a>'

def generate_pdf_report(report_text, g_scores, overall_g_score, overall_g_score_pr):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=report_text)
    
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, "G-Score Analysis", ln=True)
    pdf.set_font("Arial", size=12)
    
    pdf.cell(0, 10, f"Overall G-Score (Sensitivity-Specificity): {overall_g_score:.4f}", ln=True)
    pdf.cell(0, 10, f"Overall G-Score (Precision-Recall): {overall_g_score_pr:.4f}", ln=True)
    
    for class_name, metrics in g_scores.items():
        label_num = int(class_name.split()[-1])
        mapped_name = get_label_name(label_num)
        pdf.cell(0, 10, f"\n{mapped_name}:", ln=True)
        for metric_name, value in metrics.items():
            pdf.cell(0, 10, f"{metric_name}: {value:.4f}", ln=True)
    
    return bytes(pdf.output(dest='S'))

initialize_label_mapping()

st.title("🕵️ AI Model Performance Inspector Pro")
st.markdown("### Comprehensive model evaluation toolkit with interactive insights")

with st.expander("🏷️ Configure Label Mappings", expanded=True):
    st.markdown("### Define Label Mappings")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.number_input("Number of Labels", min_value=2, value=st.session_state.num_labels, 
                       key="num_labels", on_change=update_num_labels)
    
    with col2:
        st.markdown("#### Map Numeric Labels to Actual Names")
        cols = st.columns(4)
        for i in range(st.session_state.num_labels):
            with cols[i % 4]:
                st.text_input(f"Label {i}", 
                            value=st.session_state.label_mapping.get(i, f"Class {i}"),
                            key=f"label_{i}",
                            on_change=lambda i=i: st.session_state.label_mapping.update({i: st.session_state[f"label_{i}"]}))

with st.expander("📤 Step 1: Upload Data", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        true_file = st.file_uploader("True Labels (CSV)", type=["csv"], key="true")
    with col2:
        pred_files = st.file_uploader("Prediction Files (CSV)", type=["csv"], 
                                    accept_multiple_files=True, key="pred")

if true_file and pred_files:
    true_df = pd.read_csv(true_file)
    true_df.columns = [col.lower().strip() for col in true_df.columns]
    true_df.rename(columns={'id': 'Id', 'label': 'Label', 'labels': 'Label'}, inplace=True)

    if not {'Id', 'Label'}.issubset(true_df.columns):
        st.error("❌ True labels file must contain 'Id' and 'Label' columns")
        st.stop()

    if len(pred_files) > 1:
        num_files = st.slider("Number of prediction files to analyze",
                            min_value=1,
                            max_value=len(pred_files),
                            value=len(pred_files),
                            help="Select how many prediction files to evaluate")
    else:
        num_files = 1
        st.write("📁 Analyzing the uploaded prediction file")

    all_metrics = []
    
    for file_idx in range(num_files):
        pred_df = pd.read_csv(pred_files[file_idx])
        pred_df.columns = [col.lower().strip() for col in pred_df.columns] 
        pred_df.rename(columns={'id': 'Id', 'label': 'Label', 'labels': 'Label'}, inplace=True)
 
        with st.expander(f"📊 Analysis for: {pred_files[file_idx].name}", expanded=True):
            if not {'Id', 'Label'}.issubset(pred_df.columns):
                st.error(f"Invalid format in {pred_files[file_idx].name}")
                continue

            merged_df = pd.merge(true_df, pred_df, on='Id', suffixes=('_True', '_Pred'), how='left')
            discrepancies = merged_df[merged_df['Label_True'] != merged_df['Label_Pred']]
            
            accuracy = accuracy_score(merged_df['Label_True'], merged_df['Label_Pred'])
            cm = confusion_matrix(merged_df['Label_True'], merged_df['Label_Pred'])
            report_text = classification_report(merged_df['Label_True'], merged_df['Label_Pred'], digits=4)
            report_df = pd.DataFrame(classification_report(
                merged_df['Label_True'], merged_df['Label_Pred'], output_dict=True, digits=4)).transpose()
            
            g_scores, overall_g_score, overall_g_score_pr = calculate_g_scores(
                merged_df['Label_True'], merged_df['Label_Pred'])

            all_metrics.append({
                'File': pred_files[file_idx].name,
                'Accuracy': accuracy,
                'Overall G-Score (Sens-Spec)': overall_g_score,
                'Overall G-Score (Prec-Rec)': overall_g_score_pr,
                'Mismatches': len(discrepancies)
            })

            col_metrics, col_cm, col_report = st.columns([1, 2, 3])
            
            with col_metrics:
                st.metric("Accuracy", f"{accuracy:.4%}")
                st.metric("Overall G-Score (Sens-Spec)", f"{overall_g_score:.4f}")
                st.metric("Overall G-Score (Prec-Rec)", f"{overall_g_score_pr:.4f}")
                st.metric("Mismatches", len(discrepancies))
                
                csv = discrepancies.to_csv(index=False).encode()
                st.markdown(create_download_link(csv, f"mismatches_{file_idx+1}.csv"), 
                          unsafe_allow_html=True)

            with col_cm:
                fig, ax = plt.subplots(figsize=(8,6))

                # Extract unique labels and their mapped names
                unique_labels = np.unique(merged_df['Label_True'])
                mapped_labels = [get_label_name(label) for label in unique_labels]

                # Create heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=True,
                            xticklabels=mapped_labels,
                            yticklabels=mapped_labels,
                            annot_kws={"size": 20})
                plt.title("Confusion Matrix", fontsize=18)
                plt.xlabel("Predicted Labels", fontsize=18, labelpad=10)
                plt.ylabel("Actual Labels", fontsize=18, labelpad=10)
                plt.xticks(fontsize=16, rotation=0, ha='center')
                plt.yticks(fontsize=16, rotation=90, va='center')
                plt.tight_layout()

                
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
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
                g_scores_df = pd.DataFrame.from_dict(g_scores, orient='index')
                g_scores_df.index = [get_label_name(int(idx.split()[-1])) 
                                   for idx in g_scores_df.index]
                
                report_df.index = [get_label_name(label) if label in st.session_state.label_mapping 
                                 else label for label in report_df.index]
                st.dataframe(report_df.style.format("{:.4f}")
                           .background_gradient(cmap='Blues', axis=0))
                
                st.markdown("### G-Score Analysis")
                st.dataframe(g_scores_df.style.format("{:.4f}")
                           .background_gradient(cmap='Blues', axis=0))
                
                col_pdf, col_txt = st.columns(2)
                with col_pdf:
                    pdf_report = generate_pdf_report(report_text, g_scores, 
                                                   overall_g_score, overall_g_score_pr)
                    st.download_button(
                        f"📥 PDF Report {file_idx+1}",
                        data=pdf_report,
                        file_name=f"report_{file_idx+1}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{file_idx}"
                    )
                with col_txt:
                    full_report = (f"{report_text}\n\nG-Score Analysis:\n"
                                 f"Overall G-Score (Sens-Spec): {overall_g_score:.4f}\n"
                                 f"Overall G-Score (Prec-Rec): {overall_g_score_pr:.4f}\n\n")
                    for class_name, metrics in g_scores.items():
                        mapped_name = get_label_name(int(class_name.split()[-1]))
                        full_report += f"\n{mapped_name}:\n"
                        for metric_name, value in metrics.items():
                            full_report += f"{metric_name}: {value:.4f}\n"
                    
                    st.download_button(
                        f"📥 TXT Report {file_idx+1}",
                        data=full_report,
                        file_name=f"report_{file_idx+1}.txt",
                        mime="text/plain",
                        key=f"txt_{file_idx}"
                    )

            st.markdown("🔍 Detailed Mismatch Analysis")
            with st.container():
                col_filter, col_preview = st.columns([2, 3])
                
                with col_filter:
                    selected_class = st.multiselect(
                        "Filter by true class:", 
                        options=[get_label_name(label) for label in merged_df['Label_True'].unique()],
                        key=f"filter_{file_idx}"
                    )
                    
                    selected_numeric = [label for label, name in st.session_state.label_mapping.items() 
                                     if name in selected_class]
                    filtered_disc = (discrepancies[discrepancies['Label_True'].isin(selected_numeric)] 
                                   if selected_class else discrepancies)
                    
                    display_disc = filtered_disc.copy()
                    display_disc['Label_True'] = display_disc['Label_True'].map(get_label_name)
                    display_disc['Label_Pred'] = display_disc['Label_Pred'].map(get_label_name)
                    
                    st.dataframe(display_disc.style.applymap(
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
            .style.format({
                'Accuracy': "{:.4%}",
                'Overall G-Score (Sens-Spec)': "{:.4f}",
                'Overall G-Score (Prec-Rec)': "{:.4f}",
                'Mismatches': "{:,}"
            })
            .background_gradient(cmap='YlGnBu', subset=['Accuracy'])
        )

else:
    st.info("👋 Please upload both true labels and prediction files to begin analysis")

st.markdown("---")
st.caption("🔒 All data is processed in memory and never stored. Built with Streamlit 🎈")