# 🎯 QuickDraw Prediction Web App

An intelligent, interactive **Streamlit web app** that predicts likely number combinations for the **New Jersey Lottery Quick Draw** game. It analyzes historical draw data, highlights co-occurrence and frequency patterns, visualizes heatmaps, detects square/box formations, and supports exporting analytics — all from uploaded **PDFs or screenshots**.

---

## 🚀 Features

- 🖼️ **Upload Screenshot**: Auto-detects yellow-highlighted drawn numbers via OCR
- 📄 **Upload PDF**: Extracts historical draw data and trains prediction logic
- 🔁 **Training Toggle**: Enable/disable learning from new uploads
- 🧮 **Co-occurrence Analysis**: Top number pairs and triplets that appear together
- 📦 **Square/Box Detection**: Detects frequent rectangular patterns from remaining grid numbers
- 🔥 **Heatmaps**: Visualizes number frequency across an 8×10 grid
- 📈 **Accuracy Metrics**: Calculates how many drawn numbers match predicted ones
- ⬇️ **Export**: Download CSVs of pair and triplet co-occurrence stats

---

## 🧠 How It Works

1. **User uploads a PDF or Screenshot**
2. **OCR (Tesseract)** extracts drawn numbers
3. **Numbers are analyzed** for frequency, co-occurrence, and geometric box patterns
4. **Heatmaps and predictions** are generated for:
   - Top numbers
   - Top pairs & triplets
   - Square/box formations from leftover grid values
5. **Accuracy stats** are shown by comparing predicted vs. actual draws

---

## 🧰 Tech Stack

- **Frontend/UI**: Streamlit
- **OCR**: Tesseract via `pytesseract`
- **PDF Handling**: `pdf2image`, `PIL`
- **Data Processing**: Pandas, NumPy, Regular Expressions
- **Visualization**: Matplotlib, Seaborn
- **State Management**: Streamlit `session_state`

---

## 📂 File Upload Types

- **📸 Screenshot (JPG/PNG)**: Extracts highlighted numbers (yellow color range detection)
- **📄 PDF**: Processes each page to find valid draws (sets of 20 numbers per line)

---

## 📊 Prediction Logic

- Tracks frequency of each number across draws
- Builds co-occurrence matrix for pairs & triplets
- Detects and overlays box/square shapes in 8x10 grid using leftover numbers
- Stores prediction accuracy metrics based on recent draws

---

## 🛠️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
