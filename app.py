import os, io, json, logging, re
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import joblib
import pdfplumber
import fitz  # PyMuPDF fallback
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Alternative sentence tokenization without NLTK
def simple_sentence_tokenize(text):
    """
    Simple sentence tokenizer that splits on common sentence endings
    followed by whitespace and capital letters or numbers.
    """
    # Handle common abbreviations that shouldn't be split
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|Inc|Ltd|Co|Corp|etc|vs|e\.g|i\.e)\.\s+', r'\1<DOT> ', text)
    
    # Split on sentence endings followed by space and capital letter/number
    sentences = re.split(r'[.!?]+\s+(?=[A-Z0-9])', text)
    
    # Restore the abbreviation dots
    sentences = [s.replace('<DOT>', '.') for s in sentences if s.strip()]
    
    return sentences

# Alternative: Use spaCy for better sentence tokenization (if available)
def get_sentence_tokenizer():
    """
    Try to use the best available sentence tokenizer
    """
    # First try spaCy (more robust)
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            def spacy_tokenize(text):
                doc = nlp(text)
                return [sent.text for sent in doc.sents]
            return spacy_tokenize
        except OSError:
            # Model not installed, try blank model
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            def spacy_blank_tokenize(text):
                doc = nlp(text)
                return [sent.text for sent in doc.sents]
            return spacy_blank_tokenize
    except ImportError:
        pass
    
    # Then try NLTK
    try:
        import nltk
        # Ensure punkt is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e:
                logging.warning(f"Failed to download NLTK punkt: {e}")
                return simple_sentence_tokenize
        
        from nltk.tokenize import sent_tokenize
        return sent_tokenize
    except (ImportError, LookupError) as e:
        logging.warning(f"NLTK not available: {e}, using simple tokenizer")
        return simple_sentence_tokenize

# Initialize the sentence tokenizer
sent_tokenize = get_sentence_tokenizer()

# ---------------- Configuration ----------------
RAW_TEXT_DIR = "./raw_texts"
MODEL_DIR = "./models"
MODEL_FILE = os.path.join(MODEL_DIR, "forgery_model.joblib")
CHUNK_SIZE = 5
SIMILARITY_THRESHOLD = 0.80
MAX_FILE_SIZE_MB = 20

os.makedirs(RAW_TEXT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---------------- Serve index.html ----------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# ---------------- PDF extraction ----------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    texts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ''
                if page_text.strip():  # Only add non-empty pages
                    texts.append(f"--- PAGE {i+1} ---\n{page_text}")
        text = "\n".join(texts)
        if not text.strip():
            raise Exception("Empty text from pdfplumber")
        return text
    except Exception as e:
        logging.warning(f"pdfplumber failed ({e}), using PyMuPDF fallback")
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            texts = []
            for i, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():  # Only add non-empty pages
                    texts.append(f"--- PAGE {i+1} ---\n{page_text}")
            doc.close()
            return "\n".join(texts)
        except Exception as ex:
            logging.error(f"PyMuPDF also failed: {ex}")
            raise Exception("Failed to extract text from PDF")

# ---------------- Storage ----------------
def save_raw_text(doc_id: str, text: str):
    path = os.path.join(RAW_TEXT_DIR, f"{doc_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"doc_id": doc_id, "text": text}, f, ensure_ascii=False)
    return path

def load_raw_text(doc_id: str) -> str:
    path = os.path.join(RAW_TEXT_DIR, f"{doc_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No raw text stored for doc_id '{doc_id}'")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["text"]

def list_documents():
    return [f.split(".json")[0] for f in os.listdir(RAW_TEXT_DIR) if f.endswith(".json")]

# ---------------- Chunking ----------------
def make_chunks(text: str):
    """Create chunks from text using sentence tokenization"""
    try:
        sentences = sent_tokenize(text)
        # Filter out very short sentences (likely parsing errors)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            # Fallback: split by periods if no sentences found
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        
        chunks = []
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk = " ".join(sentences[i:i+CHUNK_SIZE])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        return chunks if chunks else [text]  # Return original text if chunking fails
    except Exception as e:
        logging.warning(f"Chunking failed: {e}, using fallback method")
        # Simple fallback: split by periods
        parts = text.split('.')
        chunks = []
        for i in range(0, len(parts), CHUNK_SIZE):
            chunk = ". ".join(parts[i:i+CHUNK_SIZE])
            if chunk.strip():
                chunks.append(chunk)
        return chunks if chunks else [text]

# ---------------- Model ----------------
def create_pipeline():
    clf = LogisticRegression(max_iter=1000, random_state=42)
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2), 
            max_features=50000, 
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )),
        ("clf", clf)
    ])

def save_model(model):
    joblib.dump(model, MODEL_FILE)
    logging.info(f"Model saved to {MODEL_FILE}")

def load_model():
    if os.path.exists(MODEL_FILE):
        logging.info(f"Loading model from {MODEL_FILE}")
        return joblib.load(MODEL_FILE)
    return None

# ---------------- Routes ----------------
@app.route("/train", methods=["POST"])
def train():
    if "file" not in request.files:
        return jsonify({"error":"no file uploaded"}), 400
    
    try:
        import pandas as pd
        df = pd.read_csv(request.files["file"])
        
        if "text" not in df.columns or "label" not in df.columns:
            return jsonify({"error":"CSV must contain 'text' and 'label' columns"}), 400
        
        # Clean the data
        df = df.dropna(subset=['text', 'label'])
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
        if len(df) == 0:
            return jsonify({"error": "No valid data found after cleaning"}), 400
        
        X = df["text"].astype(str).tolist()
        y = df["label"].tolist()
        
        logging.info(f"Training model with {len(X)} samples")
        
        model = create_pipeline()
        model.fit(X, y)
        save_model(model)
        
        # Generate predictions and report
        preds = model.predict(X)
        report = classification_report(y, preds, output_dict=True, zero_division=0)
        
        return jsonify({
            "message": "model trained successfully",
            "samples_trained": len(X),
            "report": report
        })
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/ingest", methods=["POST"])
def ingest():
    if "file" not in request.files:
        return jsonify({"error":"no file uploaded"}), 400
    
    file = request.files["file"]
    
    # Check file size
    if hasattr(file, 'content_length') and file.content_length and file.content_length > MAX_FILE_SIZE_MB*1024*1024:
        return jsonify({"error": f"file too large (max {MAX_FILE_SIZE_MB}MB)"}), 400
    
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error":"only PDF files allowed"}), 400
    
    try:
        pdf_bytes = file.read()
        
        # Check actual file size
        if len(pdf_bytes) > MAX_FILE_SIZE_MB*1024*1024:
            return jsonify({"error": f"file too large (max {MAX_FILE_SIZE_MB}MB)"}), 400
        
        doc_id = request.form.get("doc_id") or f"doc_{np.random.randint(100000, 999999)}"
        
        # Validate doc_id (alphanumeric and underscore only)
        if not re.match(r'^[a-zA-Z0-9_]+$', doc_id):
            return jsonify({"error": "doc_id must contain only letters, numbers, and underscores"}), 400
        
        text = extract_text_from_pdf_bytes(pdf_bytes)
        save_raw_text(doc_id, text)
        
        logging.info(f"Document {doc_id} ingested successfully, {len(text)} characters")
        
        return jsonify({
            "doc_id": doc_id, 
            "chars": len(text),
            "message": f"Document '{doc_id}' ingested successfully"
        })
        
    except Exception as e:
        logging.error(f"Ingest failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/raw/<doc_id>", methods=["GET"])
def raw_text(doc_id):
    try:
        text = load_raw_text(doc_id)
        return jsonify({"doc_id": doc_id, "text": text, "chars": len(text)})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

@app.route("/documents", methods=["GET"])
def documents():
    try:
        docs = list_documents()
        return jsonify({"documents": docs, "count": len(docs)})
    except Exception as e:
        logging.error(f"Failed to list documents: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/compare", methods=["POST"])
def compare():
    if "file" not in request.files or "doc_id" not in request.form:
        return jsonify({"error":"file and doc_id required"}), 400
    
    doc_id = request.form["doc_id"]
    
    try:
        old_text = load_raw_text(doc_id)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    
    try:
        new_pdf_bytes = request.files["file"].read()
        new_text = extract_text_from_pdf_bytes(new_pdf_bytes)
        
        old_chunks = make_chunks(old_text)
        new_chunks = make_chunks(new_text)
        
        model = load_model()
        results = []
        
        max_chunks = max(len(old_chunks), len(new_chunks))
        
        for idx in range(max_chunks):
            old_chunk = old_chunks[idx] if idx < len(old_chunks) else ""
            new_chunk = new_chunks[idx] if idx < len(new_chunks) else ""
            
            # Calculate similarity
            similarity = SequenceMatcher(None, old_chunk, new_chunk).ratio()
            
            # Model prediction
            pred_label, confidence = None, None
            if model and new_chunk.strip():
                try:
                    pred_label = int(model.predict([new_chunk])[0])
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba([new_chunk])[0]
                        confidence = float(max(proba))
                except Exception as e:
                    logging.warning(f"Model prediction failed for chunk {idx}: {e}")
            
            # Flag suspicious chunks
            is_suspicious = (
                similarity < SIMILARITY_THRESHOLD or 
                pred_label == 1 or
                (old_chunk and not new_chunk) or  # Deleted content
                (not old_chunk and new_chunk)     # Added content
            )
            
            if is_suspicious:
                results.append({
                    "index": idx,
                    "similarity": round(similarity, 4),
                    "old_preview": old_chunk[:500] + ("..." if len(old_chunk) > 500 else ""),
                    "new_preview": new_chunk[:500] + ("..." if len(new_chunk) > 500 else ""),
                    "pred_label": pred_label,
                    "confidence": round(confidence, 4) if confidence else None,
                    "change_type": (
                        "deleted" if old_chunk and not new_chunk else
                        "added" if not old_chunk and new_chunk else
                        "modified"
                    )
                })
        
        # Overall similarity
        overall_similarity = SequenceMatcher(None, old_text, new_text).ratio()
        percent_changed = (1.0 - overall_similarity) * 100
        
        # Change summary
        if percent_changed < 5:
            summary = "minimal changes"
        elif percent_changed < 15:
            summary = "minor changes"
        elif percent_changed < 35:
            summary = "moderate changes"
        else:
            summary = "major changes"
        
        logging.info(f"Comparison completed: {percent_changed:.2f}% changed, {len(results)} suspicious chunks")
        
        return jsonify({
            "doc_id": doc_id,
            "overall_similarity": round(overall_similarity, 4),
            "percent_changed": round(percent_changed, 2),
            "change_summary": summary,
            "num_changed_chunks": len(results),
            "total_chunks": max_chunks,
            "changed_chunks": results
        })
        
    except Exception as e:
        logging.error(f"Compare failed: {e}")
        return jsonify({"error": str(e)}), 500

# Add a health check endpoint
@app.route("/health", methods=["GET"])
def health():
    model_loaded = load_model() is not None
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "documents_count": len(list_documents())
    })

if __name__ == "__main__":
    logging.info("Starting PDF Forgery Detection Server...")
    logging.info(f"Sentence tokenizer: {sent_tokenize.__name__ if hasattr(sent_tokenize, '__name__') else 'custom'}")
    app.run(host="0.0.0.0", port=7860, debug=True)