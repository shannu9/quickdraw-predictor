# QuickDraw Prediction Web App (Streamlit-based)
# Features: Upload PDF or Screenshot | Auto OCR | Grid Filtering | Heatmaps | Co-occurrence & Triplet Analysis | Square/Box Detection | Accuracy Stats | Export Options | Training Toggle

import streamlit as st
import pandas as pd
import numpy as np
import pytesseract
import cv2
import os
import re
from PIL import Image
from collections import defaultdict, Counter
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

st.set_page_config(layout="wide")

# -------------------- STATE --------------------
if 'training_enabled' not in st.session_state:
    st.session_state['training_enabled'] = True
if 'co_occurrence' not in st.session_state:
    st.session_state['co_occurrence'] = defaultdict(lambda: defaultdict(int))
if 'triplet_occurrence' not in st.session_state:
    st.session_state['triplet_occurrence'] = defaultdict(int)
if 'occurrence' not in st.session_state:
    st.session_state['occurrence'] = defaultdict(int)
if 'predicted_blocks' not in st.session_state:
    st.session_state['predicted_blocks'] = set()
if 'last_draw' not in st.session_state:
    st.session_state['last_draw'] = []
if 'pdf_error' not in st.session_state:
    st.session_state['pdf_error'] = ""

# -------------------- OCR FROM IMAGE --------------------
def extract_drawn_numbers_from_image(uploaded_image):
    image = Image.open(uploaded_image)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow = cv2.bitwise_and(img_cv, img_cv, mask=mask)
    gray = cv2.cvtColor(yellow, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789')
    numbers = list(map(int, re.findall(r'\b[1-9]\b|\b[1-7][0-9]\b|\b80\b', text)))
    return sorted(set(numbers))

# -------------------- TRAINING FROM PDF --------------------
def process_pdf_and_train(uploaded_file):
    try:
        st.session_state['pdf_error'] = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name
        images = convert_from_path(temp_pdf_path)
        draw_results = []
        pattern = re.compile(r"(?:\b\d{1,2}\b[\s]*){10,25}")
        total_pages = len(images)
        for i, image in enumerate(images):
            st.info(f"Processing page {i+1}/{total_pages} ({int((i+1)/total_pages*100)}%)")
            text = pytesseract.image_to_string(image)
            matches = pattern.findall(text)
            for match in matches:
                numbers = list(map(int, re.findall(r'\b\d{1,2}\b', match)))
                if len(numbers) == 20:
                    draw_results.append(numbers)
        df = pd.DataFrame(draw_results)
        if not df.empty:
            for row in df.values:
                for num in row:
                    st.session_state['occurrence'][num] += 1
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        a, b = sorted((row[i], row[j]))
                        st.session_state['co_occurrence'][a][b] += 1
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        for k in range(j + 1, len(row)):
                            t = tuple(sorted((row[i], row[j], row[k])))
                            st.session_state['triplet_occurrence'][t] += 1
            st.success(f"✅ Trained from {len(df)} draws.")
        else:
            raise ValueError("No valid draw lines found.")
    except Exception as e:
        st.session_state['pdf_error'] = str(e)

# -------------------- VISUALIZATIONS --------------------
def draw_heatmap():
    heatmap = np.zeros((8, 10))
    for num, freq in st.session_state['occurrence'].items():
        r, c = divmod(num - 1, 10)
        heatmap[r][c] = freq
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_title("🎨 Number Frequency Heatmap")
    st.pyplot(fig)

def highlight_boxes(drawn):
    grid = np.arange(1, 81).reshape(8, 10)
    mask = np.isin(grid, drawn, invert=True)
    boxes = set()
    for r in range(8):
        for c in range(10):
            for h in range(1, 9 - r):
                for w in range(1, 11 - c):
                    block = mask[r:r+h, c:c+w]
                    if np.all(block):
                        box_nums = {grid[r+i][c+j] for i in range(h) for j in range(w)}
                        boxes.add(frozenset(box_nums))
    st.session_state['predicted_blocks'] = set.union(*boxes) if boxes else set()
    if boxes:
        st.write("### 📦 Possible Square/Rectangle Formations")
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(8):
            for j in range(10):
                color = 'gray' if grid[i, j] in drawn else 'white'
                ax.add_patch(plt.Rectangle((j, 7 - i), 1, 1, color=color, ec='black'))
                ax.text(j + 0.5, 7.5 - i, str(grid[i, j]), ha='center', va='center', fontsize=8)
        for b in list(boxes)[:5]:
            for num in b:
                i, j = divmod(num - 1, 10)
                ax.add_patch(plt.Rectangle((j, 7 - i), 1, 1, fill=False, edgecolor='blue', linewidth=1.5))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        st.pyplot(fig)

# -------------------- PREDICTIONS & EXPORT --------------------
def predict_top_numbers(n=20):
    return sorted(st.session_state['occurrence'].items(), key=lambda x: x[1], reverse=True)[:n]

def predict_top_pairs(n=10):
    pairs = []
    for a in st.session_state['co_occurrence']:
        for b in st.session_state['co_occurrence'][a]:
            pairs.append(((a, b), st.session_state['co_occurrence'][a][b]))
    return sorted(pairs, key=lambda x: x[1], reverse=True)[:n]

def predict_top_triplets(n=10):
    return sorted(st.session_state['triplet_occurrence'].items(), key=lambda x: x[1], reverse=True)[:n]

def calculate_accuracy(drawn, predicted_nums, predicted_blocks):
    match_nums = set(drawn) & set(predicted_nums)
    match_box = set(drawn) & set(predicted_blocks)
    return len(match_nums)/len(predicted_nums), len(match_box)/max(len(predicted_blocks), 1)

def export_analysis():
    pair_data = [{'A': a, 'B': b, 'Count': st.session_state['co_occurrence'][a][b]} for a in st.session_state['co_occurrence'] for b in st.session_state['co_occurrence'][a]]
    df_pairs = pd.DataFrame(pair_data)
    st.download_button("⬇️ Export Pair Co-occurrence", df_pairs.to_csv(index=False), file_name="pairs.csv")
    triplet_data = [{'Triplet': str(k), 'Count': v} for k, v in st.session_state['triplet_occurrence'].items()]
    df_triplets = pd.DataFrame(triplet_data)
    st.download_button("⬇️ Export Triplet Co-occurrence", df_triplets.to_csv(index=False), file_name="triplets.csv")
