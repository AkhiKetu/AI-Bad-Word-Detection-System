# app.py — Flask backend for AI Bad Word Detection System
# Works with hybrid_detector_model_confirm.py and json_data.jsonl
# Serves frontend and analyzes paragraphs (Bangla).

from flask import Flask, request, jsonify, send_from_directory
import os

# Import only the functions (TRAIN_JSONL removed)
from hybrid_detector_model_confirm import (
    normalize_text,
    load_examples_from_jsonl,
    detect_in_paragraph,
    query_ollama_label,
    sanitize_model_output,
)

# ✅ Use your correct dataset
TRAIN_JSONL = "train_data.jsonl"

app = Flask(__name__, static_folder=".", static_url_path="")

# -------------------------------------------------------
# FAST IN-PROCESS DETECTOR FUNCTION
# -------------------------------------------------------
def analyze_paragraph_with_matches(paragraph: str, model_name: str = "mybanglamodel"):
    """
    Analyze paragraph in-process (no subprocess).
    Returns (labels_list, matches_list)
    where matches_list = [{ "text": ..., "label": ... }]
    """
    paragraph_norm = normalize_text(paragraph)
    examples = load_examples_from_jsonl(TRAIN_JSONL)
    matches = detect_in_paragraph(paragraph_norm, examples)

    if not matches:
        return ["None"], []

    found_labels = []
    found_matches = []

    for sent, ds_label in matches:
        # skip model confirmation for "None" labels to save time
        if ds_label.lower() == "none":
            final_label = "None"
        else:
            resp = query_ollama_label(model_name, sent, timeout=60)
            resp_sanitized = (
                sanitize_model_output(resp)
                if resp and not resp.startswith("[")
                else ds_label
            )
            final_label = resp_sanitized if resp_sanitized else ds_label

        found_matches.append({"text": sent, "label": final_label})
        found_labels.append(final_label.strip().capitalize())

    # Deduplicate labels (preserve order)
    uniq = []
    seen = set()
    for l in found_labels:
        key = l.lower()
        if key not in seen:
            uniq.append(l)
            seen.add(key)

    return uniq or ["None"], found_matches


# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    paragraph = (data.get("text") or "").strip()
    if not paragraph:
        return jsonify({"error": "No text provided"}), 400

    try:
        labels, matches = analyze_paragraph_with_matches(paragraph, model_name="mybanglamodel")
        return jsonify({"labels": labels, "matches": matches})
    except Exception as e:
        return jsonify({"error": f"Server error: {e}"}), 500


# Serve static files (frontend)
@app.route("/")
def root():
    return send_from_directory(".", "index.html")


@app.route("/index.html")
def index_html():
    return send_from_directory(".", "index.html")


@app.route("/analyze.html")
def analyze_html():
    return send_from_directory(".", "analyze.html")


@app.route("/main.css")
def main_css():
    return send_from_directory(".", "main.css")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


if __name__ == "__main__":
    # Run Flask app
    print("🚀 Starting ToxicScan Flask backend...")
    print("📁 Using dataset:", TRAIN_JSONL)
    app.run(host="127.0.0.1", port=5000, debug=True)
