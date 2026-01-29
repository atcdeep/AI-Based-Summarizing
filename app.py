import io
import os
import re
import json
import warnings
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# File handling
import PyPDF2
from docx import Document
from PIL import Image
import pytesseract

# NLP & similarity
import nltk
import yake
from nltk.tokenize import sent_tokenize

# Hugging Face
from transformers import T5ForConditionalGeneration, T5Tokenizer

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)

# ------------------------------
# Config
# ------------------------------
MODEL_FILE = "cached_model.pkl"
ARXIV_FILE = "arxiv_data_cleaned_fast.csv"
SCOPUS_FILE = "scorpus metadata.csv"

app = Flask(__name__, template_folder='.')
CORS(app)

# ------------------------------
# Utility Functions
# ------------------------------
def highlight_keywords(text, keywords):
    text_html = text
    for kw in keywords:
        if kw.strip():
            pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            text_html = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text_html)
    return text_html

def keyword_counts_in_text(text, keywords):
    counts = {}
    for kw in keywords:
        kw_escaped = re.escape(kw)
        counts[kw] = len(re.findall(r'\b' + kw_escaped + r'\b', text, flags=re.IGNORECASE))
    return counts

def extract_text_from_file(file):
    text = ""
    try:
        if file.filename.endswith('.txt'):
            text = file.read().decode("utf-8", errors="ignore")
        elif file.filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif file.filename.endswith('.docx'):
            doc = Document(io.BytesIO(file.read()))
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(io.BytesIO(file.read()))
            try:
                text = pytesseract.image_to_string(img)
            except Exception:
                text = ""
    except Exception as e:
        print(f"Error extracting text: {e}")
        text = ""
    return text

# ------------------------------
# Load datasets
# ------------------------------
def load_dataset(file_path, sample_frac=0.1):
    df = pd.read_csv(file_path)
    for col in ["titles_cleaned", "summaries_cleaned", "terms_cleaned"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    def parse_terms_cell(x):
        try:
            cleaned = x.strip().strip("[]")
            parts = [t.strip().strip("'\"") for t in cleaned.split(",") if t.strip()]
            return parts
        except:
            return []

    df["terms_list"] = df["terms_cleaned"].apply(parse_terms_cell)
    sample_frac = max(min(sample_frac, 1.0), 0.01)
    df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    return df

# ------------------------------
# Train classifier
# ------------------------------
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(df["titles_cleaned"].fillna("") + " " + df["summaries_cleaned"].fillna(""))
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["terms_list"])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga'))
    classifier.fit(X_train, Y_train)
    joblib.dump((vectorizer, mlb, classifier), MODEL_FILE)
    return vectorizer, mlb, classifier

# Load or train model
if os.path.exists(MODEL_FILE):
    try:
        vectorizer, mlb, classifier = joblib.load(MODEL_FILE)
        print("Pretrained model loaded")
    except:
        arxiv_df = load_dataset(ARXIV_FILE, 0.1)
        vectorizer, mlb, classifier = train_model(arxiv_df)
        print("Model retrained")
else:
    arxiv_df = load_dataset(ARXIV_FILE, 0.1)
    vectorizer, mlb, classifier = train_model(arxiv_df)
    print("Model trained fresh")

# ------------------------------
# Load Flan-T5
# ------------------------------
print("Loading T5 model...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
try:
    import torch
    if torch.cuda.is_available():
        t5_model = t5_model.to("cuda")
        print("T5 model moved to GPU")
except:
    print("T5 model on CPU")

# ------------------------------
# Summarization
# ------------------------------
def summarize_with_t5(text, max_chunk_len=500, final_max_len=250):
    if not text.strip():
        return ""
    sentences = sent_tokenize(text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current.split()) + len(sent.split()) <= max_chunk_len:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())

    partial_summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk.strip(), return_tensors="pt", truncation=True, max_length=512)
        try:
            import torch
            if t5_model.device.type == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
        except:
            pass
        summary_ids = t5_model.generate(**inputs, max_length=150, min_length=40, num_beams=4, early_stopping=True)
        partial_summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    combined = " ".join(partial_summaries)
    inputs = tokenizer(combined, return_tensors="pt", truncation=True, max_length=512)
    try:
        import torch
        if t5_model.device.type == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
    except:
        pass
    final_ids = t5_model.generate(**inputs, max_length=final_max_len, min_length=80, num_beams=4, early_stopping=True)
    return tokenizer.decode(final_ids[0], skip_special_tokens=True)

# ------------------------------
# Extractive summary
# ------------------------------
def extractive_summary(text, top_n=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= top_n:
        return text
    tfidf = TfidfVectorizer(stop_words="english")
    vecs = tfidf.fit_transform(sentences)
    sim_matrix = cosine_similarity(vecs)
    scores = sim_matrix.sum(axis=1)
    top_idx = scores.argsort()[-top_n:][::-1]
    return " ".join([sentences[i] for i in top_idx])

# ------------------------------
# Keyword extraction
# ------------------------------
def suggest_topics(text, top_n=5):
    if not text.strip():
        return []
    kw_extractor = yake.KeywordExtractor(top=top_n, stopwords=None)
    return [kw for kw, _ in kw_extractor.extract_keywords(text)]

# ------------------------------
# Compute similarity
# ------------------------------
def compute_similarity(page_text, summary_text):
    page_sents = sent_tokenize(page_text)
    summary_sents = sent_tokenize(summary_text)
    if not page_sents or not summary_sents:
        return 0.0
    tfidf = TfidfVectorizer(stop_words="english")
    sims = []
    for s in summary_sents:
        vecs = tfidf.fit_transform(page_sents + [s])
        sim = cosine_similarity(vecs[-1:], vecs[:-1]).max()
        sims.append(float(sim))
    return sum(sims)/len(sims) if sims else 0.0

# ------------------------------
# Flask Routes
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dataset_stats', methods=['GET'])
def get_dataset_stats():
    try:
        arxiv_df = load_dataset(ARXIV_FILE, 0.1)
        scopus_df = load_dataset(SCOPUS_FILE, 0.1)
        return jsonify({
            'arxiv_rows': len(arxiv_df),
            'scopus_rows': len(scopus_df)
        })
    except Exception as e:
        return jsonify({'arxiv_rows': 0, 'scopus_rows': 0})

@app.route('/api/extract_text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    text = extract_text_from_file(file)
    
    if not text.strip():
        return jsonify({'error': 'Could not extract text from file'}), 400
    
    # Extract keywords for highlighting
    keywords = suggest_topics(text, top_n=10)
    highlighted_text = highlight_keywords(text, keywords)
    
    return jsonify({
        'extracted_text': text,
        'highlighted_text': highlighted_text,
        'keywords': keywords
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    try:
        data = request.json
        text = data.get('text', '')
        sample_fraction = float(data.get('sample_fraction', 0.1))
        confidence_threshold = float(data.get('confidence_threshold', 0.05))
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate summaries
        summary_abstractive = summarize_with_t5(text)
        summary_extractive = extractive_summary(text, top_n=5)
        
        # Extract keywords and topics
        keywords = suggest_topics(text, top_n=10)
        topics = suggest_topics(text, top_n=5)
        
        # Compute similarity
        sim_score = compute_similarity(text, summary_abstractive)
        
        # TF-IDF similarity data
        tfidf_data = {
            'labels': ['Similar', 'Dissimilar'],
            'data': [sim_score * 100, max(0, 1 - sim_score) * 100],
            'colors': ['#3498db', '#e74c3c']
        }
        
        # Summary lengths
        length_data = {
            'labels': ['Original', 'Abstractive', 'Extractive'],
            'data': [
                len(text.split()),
                len(summary_abstractive.split()),
                len(summary_extractive.split())
            ],
            'colors': ['#2c3e50', '#27ae60', '#3498db']
        }
        
        # Keyword counts
        kw_count_dict = keyword_counts_in_text(text, keywords)
        keyword_data = {
            'labels': list(kw_count_dict.keys()),
            'data': list(kw_count_dict.values()),
            'colors': '#f39c12'
        }
        
        # Classifier predictions
        X_new = vectorizer.transform([text[:2000]])
        try:
            probs = classifier.predict_proba(X_new)[0]
        except:
            probs = classifier.decision_function(X_new)[0]
            probs = 1/(1+np.exp(-probs))
        
        top_indices = probs.argsort()[::-1]
        categories = [(mlb.classes_[i], float(probs[i])) 
                     for i in top_indices[:10] 
                     if float(probs[i]) >= confidence_threshold]
        
        return jsonify({
            'success': True,
            'extractive_summary': summary_extractive,
            'abstractive_summary': summary_abstractive,
            'keywords': keywords,
            'topics': topics,
            'similarity_score': sim_score,
            'tfidf_data': tfidf_data,
            'length_data': length_data,
            'keyword_data': keyword_data,
            'categories': categories
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_assistant():
    try:
        data = request.json
        question = data.get('question', '')
        document_text = data.get('document_text', '')
        
        if not question or not document_text:
            return jsonify({'error': 'Missing question or document text'}), 400
        
        # Create context from document
        sentences = sent_tokenize(document_text)
        chunk_size = 500
        chunks, tmp = [], ""
        for sent in sentences:
            if len(tmp.split()) + len(sent.split()) <= chunk_size:
                tmp += " " + sent
            else:
                chunks.append(tmp.strip())
                tmp = sent
        if tmp:
            chunks.append(tmp.strip())
        
        partial_summaries = []
        for chunk in chunks:
            inputs = tokenizer(chunk.strip(), return_tensors="pt", truncation=True, max_length=512)
            try:
                import torch
                if t5_model.device.type == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
            except:
                pass
            ids = t5_model.generate(
                **inputs,
                max_length=150,
                min_length=40,
                num_beams=4,
                early_stopping=True
            )
            partial_summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))
        
        combined_ctx = " ".join(partial_summaries)
        context_summary = summarize_with_t5(combined_ctx, final_max_len=250)
        
        # Generate answer
        prompt = f"Question: {question}\n\nContext: {context_summary}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        try:
            import torch
            if t5_model.device.type == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
        except:
            pass
        
        outputs = t5_model.generate(
            **inputs,
            max_length=450,
            min_length=100,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_excel', methods=['POST'])
def download_excel():
    try:
        data = request.json
        analysis_results = data.get('analysis_results', {})
        
        # Create Excel file
        output = io.BytesIO()
        
        # Create DataFrames
        df_tfidf = pd.DataFrame({
            'Metric': ['Similarity', 'Dissimilarity'],
            'Value': [
                analysis_results.get('similarity_score', 0) * 100,
                max(0, 1 - analysis_results.get('similarity_score', 0)) * 100
            ]
        })
        
        df_length = pd.DataFrame({
            'Summary Type': ['Original', 'Abstractive', 'Extractive'],
            'Length (words)': [
                analysis_results.get('original_length', 0),
                len(analysis_results.get('abstractive_summary', '').split()),
                len(analysis_results.get('extractive_summary', '').split())
            ]
        })
        
        df_keywords = pd.DataFrame({
            'Keyword': analysis_results.get('keywords', []),
            'Frequency': list(analysis_results.get('keyword_counts', {}).values())
        })
        
        df_categories = pd.DataFrame(
            analysis_results.get('categories', []),
            columns=['Category', 'Confidence']
        )
        
        # Write to Excel
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_tfidf.to_excel(writer, sheet_name='TF-IDF', index=False)
            df_length.to_excel(writer, sheet_name='Summary Lengths', index=False)
            if not df_keywords.empty:
                df_keywords.to_excel(writer, sheet_name='Keywords', index=False)
            if not df_categories.empty:
                df_categories.to_excel(writer, sheet_name='Categories', index=False)
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='document_analysis.xlsx'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)