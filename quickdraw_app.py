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
            progress = int((i + 1) / total_pages * 100)
            st.info(f"Processing page {i+1}/{total_pages} ({progress}%)")
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
            st.success(f"Successfully trained on {len(df)} draws from PDF.")
        else:
            raise ValueError("No valid draw lines found in the PDF.")

    except Exception as e:
        st.session_state['pdf_error'] = str(e)

# -------------------- MAIN INTERFACE --------------------
...
uploaded_pdf = st.file_uploader("Upload Historical Draw Data (PDF)", type=['pdf'])
if uploaded_pdf and st.session_state['training_enabled']:
    with st.spinner("Training from uploaded PDF..."):
        process_pdf_and_train(uploaded_pdf)
if st.session_state['pdf_error']:
    st.error(f"❌ PDF Error: {st.session_state['pdf_error']}")
