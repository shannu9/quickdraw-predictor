# QuickDraw Prediction Web App (Streamlit-based)
# Features: Upload Training Data | OCR from Screenshot | Toggle Training | Grid + Prediction + Heatmaps + Box Detection + Accuracy

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
if 'occurrence' not in st.session_state:
    st.session_state['occurrence'] = defaultdict(int)
if 'predicted_blocks' not in st.session_state:
    st.session_state['predicted_blocks'] = set()
if 'last_draw' not in st.session_state:
    st.session_state['last_draw'] = []
if 'pdf_error' not in st.session_state:
    st.session_state['pdf_error'] = ""

# -------------------- FUNCTIONS --------------------
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

def predict_top_numbers(n=20):
    return sorted(st.session_state['occurrence'].items(), key=lambda x: x[1], reverse=True)[:n]

def predict_top_pairs(n=10):
    pairs = []
    for a in st.session_state['co_occurrence']:
        for b in st.session_state['co_occurrence'][a]:
            pairs.append(((a, b), st.session_state['co_occurrence'][a][b]))
    return sorted(pairs, key=lambda x: x[1], reverse=True)[:n]

def calculate_accuracy(drawn, predicted_nums, predicted_blocks):
    match_nums = set(drawn) & set(predicted_nums)
    match_block = set(drawn) & set(predicted_blocks)
    return len(match_nums) / len(predicted_nums), len(match_block) / max(len(predicted_blocks), 1)

def highlight_boxes(drawn):
    grid = np.arange(1, 81).reshape(8, 10)
    mask = np.isin(grid, list(drawn), invert=True)
    blocks = set()
    for r in range(8):
        for c in range(10):
            for h in range(1, 9 - r):
                for w in range(1, 11 - c):
                    region = mask[r:r+h, c:c+w]
                    if np.all(region):
                        box_nums = set(grid[r + i][c + j] for i in range(h) for j in range(w))
                        blocks.add(frozenset(box_nums))
                        if h >= 2 and w >= 2:
                            break
    st.session_state['predicted_blocks'] = set.union(*blocks) if blocks else set()
    if blocks:
        st.write("### 📦 Possible Square/Rectangle Formations")
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(8):
            for j in range(10):
                color = 'gray' if grid[i, j] in drawn else 'white'
                ax.add_patch(plt.Rectangle((j, 7 - i), 1, 1, color=color, ec='black'))
                ax.text(j + 0.5, 7.5 - i, str(grid[i, j]), ha='center', va='center', fontsize=8)
        for b in list(blocks)[:5]:
            for num in b:
                i, j = divmod(num - 1, 10)
                ax.add_patch(plt.Rectangle((j, 7 - i), 1, 1, fill=False, edgecolor='blue', linewidth=1.5))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("No valid rectangles found in leftover grid.")

def draw_heatmap():
    heat_values = np.zeros((8, 10))
    for num, freq in st.session_state['occurrence'].items():
        i, j = divmod(num - 1, 10)
        heat_values[i, j] = freq
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heat_values, annot=True, fmt=".0f", cmap="YlOrRd", cbar=True, ax=ax)
    ax.set_title("Number Frequency Heatmap")
    st.pyplot(fig)

# -------------------- MAIN INTERFACE --------------------
st.sidebar.title("Quick Draw Dashboard")
train_toggle = st.sidebar.checkbox("Enable Training from Uploaded Data", value=True)
st.session_state['training_enabled'] = train_toggle

st.title("🎯 NJ Lottery Quick Draw Prediction Tool")

uploaded_img = st.file_uploader("Upload Screenshot of Latest Draw (highlighted numbers)", type=['png', 'jpg', 'jpeg'])
if uploaded_img:
    draw = extract_drawn_numbers_from_image(uploaded_img)
    st.session_state['last_draw'] = draw
    st.success(f"Detected Drawn Numbers: {draw}")
    highlight_boxes(draw)
else:
    st.warning("Upload a screenshot to begin analysis.")

uploaded_pdf = st.file_uploader("Upload Historical Draw Data (PDF)", type=['pdf'])
if uploaded_pdf and st.session_state['training_enabled']:
    with st.spinner("Training from uploaded PDF..."):
        process_pdf_and_train(uploaded_pdf)
if st.session_state['pdf_error']:
    st.error(f"❌ PDF Error: {st.session_state['pdf_error']}")

if st.button("🔮 Show Predictions"):
    top_nums = predict_top_numbers()
    top_pairs = predict_top_pairs()
    st.subheader("Top Predicted Numbers:")
    st.write([num for num, _ in top_nums])
    st.subheader("Top Pairs (Likely to be drawn together):")
    st.write([pair for pair, _ in top_pairs])

    if st.session_state['last_draw']:
        pred_list = [num for num, _ in top_nums]
        box_list = list(st.session_state['predicted_blocks'])
        acc_pred, acc_box = calculate_accuracy(st.session_state['last_draw'], pred_list, box_list)
        st.markdown(f"**🎯 Prediction Accuracy:** {acc_pred * 100:.2f}%")
        st.markdown(f"**📦 Box/Rectangle Accuracy:** {acc_box * 100:.2f}%")

if st.button("📊 Show Heatmap"):
    draw_heatmap()
