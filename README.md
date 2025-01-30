# PerfSpectra
**PerfSpectra** is an interactive **Streamlit-based** tool for evaluating classification models. It helps researchers, data scientists, and ML practitioners analyze model performance with **confusion matrices, classification reports, and mismatch analysis**.  

This tool is especially useful for **writing shared task papers**, as it automates error analysis and provides downloadable reports in multiple formats (CSV, PDF, TXT).  

---

## 🚀 Features  

✅ **Confusion Matrix** 📊 – Visualizes model performance  
✅ **Classification Report** 📝 – Precision, Recall, F1-score  
✅ **Mismatch Analysis** 🔍 – Highlights misclassified samples  
✅ **Downloadable Reports** 📥 – Get insights in CSV, PDF, and TXT  
✅ **Multi-file Support** 📂 – Compare multiple prediction files  
✅ **Interactive UI** 🎨 – Built with **Streamlit** for ease of use  

---

## 🛠️ Installation  

1️⃣ Clone the repository:  
```bash
git clone https://github.com/RJ-Hossan/PerfSpectra.git
cd PerfSpectra
```

2️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```

3️⃣ Run the app:  
```bash
streamlit run app.py
```

---

## 📤 How to Use  

1️⃣ Upload **True Labels (CSV)** with columns:  
   - `Id` (Unique identifier)  
   - `Label` (True class labels)  

2️⃣ Upload **Prediction Files (CSV)** with columns:  
   - `Id` (Matching unique identifier)  
   - `Label` (Predicted class labels)  

3️⃣ Get **accuracy, confusion matrix, classification report, and mismatches**  

4️⃣ **Download reports** in CSV, PDF, or TXT format  

---

## ⚠️ Limitations  

⚠️ **File Format**: CSV only  
⚠️ **Column Names**: Must contain `Id` and `Label` (case-insensitive)  
⚠️ **Task Support**: Currently for **classification models only**  

---

## 🤝 Contribute  

This project is **open-source**, and contributions are welcome!  

### Steps to contribute:  
1️⃣ Fork the repo 🍴  
2️⃣ Create a new branch 🔀  
3️⃣ Make your changes ✨  
4️⃣ Submit a pull request 📩  

---

## 📜 License  

MIT License – Feel free to use and modify!  

---

## 🔗 Connect  

💬 Have suggestions? Want to contribute? Drop an issue or connect with me on **[LinkedIn](https://www.linkedin.com/in/mdrefajhossan/)**  

**⭐ If you find this useful, don't forget to star the repo!** 🌟  

---
