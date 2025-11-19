# hybrid_detector_model_confirm.py
# Robust hybrid detector for Bangla paragraphs using dataset + optional Ollama confirmation.
# Works seamlessly with app.py (Flask backend) and analyze.html frontend.

import argparse, json, os, subprocess, re, sys, io, unicodedata
from typing import List, Tuple

# ---------- Clean ANSI/spinner noise ----------
ANSI_RE = re.compile(r"(?:\x1B[@-Z\\-_]|\x1B\[[0-?]*[ -/]*[@-~]|\x9B[0-?]*[ -/]*[@-~])")


def strip_ansi(s: str) -> str:
    if not s:
        return ""
    s = ANSI_RE.sub("", s)
    s = re.sub(r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]", "", s)
    return s


# ---------- UTF-8 safety ----------
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
except Exception:
    pass

# ---------- Constants ----------
DEFAULT_TIMEOUT = 120
FALLBACK = [
    ("জয়বাংলার ভোটচোর বলে কথা", "Political Hate"),
    ("হিন্দু বাচ্চাদের কি দেওয়া হলো", "Religious Hate"),
    ("মা পৃথিবীর আপন জন্য", "None"),
]


# ---------- Helpers ----------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\s+", " ", s.strip())
    return s


def load_examples_from_jsonl(path: str) -> List[Tuple[str, str]]:
    """Load dataset (input/output pairs) and deduplicate."""
    if not os.path.exists(path):
        print(f"[warn] Dataset not found: {path}, using fallback")
        return FALLBACK[:]

    best = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = normalize_text(obj.get("input") or obj.get("text") or "")
            label = str(obj.get("output") or obj.get("label") or "None").strip()
            if not text:
                continue
            prev = best.get(text)
            if prev is None or (prev.lower() == "none" and label.lower() != "none"):
                best[text] = label
    return list(best.items()) or FALLBACK[:]


def detect_in_paragraph(
    paragraph: str, examples: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """
    Detect matching dataset sentences in a Bangla paragraph.
    Uses exact substring + token Jaccard + bigram Dice similarity.
    """

    def tokenize(s: str) -> list[str]:
        return normalize_text(s).split(" ")

    def jaccard(a: list[str], b: list[str]) -> float:
        A, B = set(a), set(b)
        return len(A & B) / len(A | B) if (A | B) else 0.0

    def dice(a: str, b: str) -> float:
        def bigrams(x):
            return [x[i : i + 2] for i in range(len(x) - 1)]

        A, B = bigrams(normalize_text(a)), bigrams(normalize_text(b))
        if not A or not B:
            return 0.0
        inter = len([x for x in A if x in B])
        return (2 * inter) / (len(A) + len(B))

    paragraph_norm = normalize_text(paragraph)
    sentences = re.split(r"(?<=[.!?।])\s+|\n+", paragraph_norm)
    matches = []

    for sent_text in sentences:
        sent_norm = normalize_text(sent_text)
        if not sent_norm:
            continue
        best_label, best_score = "Other", 0.0
        sent_tokens = tokenize(sent_norm)

        for ex_text, ex_label in examples:
            ex_norm = normalize_text(ex_text)
            if not ex_norm:
                continue

            # Exact substring match first
            score = 1.0 if ex_norm in sent_norm else 0.0
            if score < 1.0:
                j = jaccard(sent_tokens, tokenize(ex_norm))
                d = dice(sent_norm, ex_norm)
                score = max(j, 0.6 * j + 0.4 * d)

            if score > best_score:
                best_label, best_score = ex_label, score

        # threshold = 0.5
        if best_score >= 0.5:
            matches.append((sent_text, best_label))
        else:
            matches.append((sent_text, "Other"))

    return matches


def sanitize_model_output(raw: str) -> str:
    raw = strip_ansi(raw or "")
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return ""
    first = lines[0]
    for prefix in ("Label:", "label:", "Output:", "output:"):
        if first.startswith(prefix):
            first = first[len(prefix) :].strip()
            break
    if (first.startswith('"') and first.endswith('"')) or (
        first.startswith("'") and first.endswith("'")
    ):
        first = first[1:-1].strip()
    return first.strip()


def _call_ollama_stdin(model, prompt, timeout):
    cmd = ["ollama", "run", model]
    return subprocess.run(
        cmd,
        input=prompt + "\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        timeout=timeout,
    )


def query_ollama_label(
    model_name: str, sentence: str, timeout: int = DEFAULT_TIMEOUT
) -> str:
    prompt = f'You are a strict Bangla classifier. OUTPUT ONLY the label text (single line).\n\nSentence: "{sentence}"\n\nLabel:'
    try:
        proc = _call_ollama_stdin(model_name, prompt, timeout)
        out = strip_ansi(proc.stdout or "")
        if proc.returncode == 0 and out:
            cleaned = sanitize_model_output(out)
            if cleaned:
                return cleaned
        return sanitize_model_output(out)
    except subprocess.TimeoutExpired:
        return "[error] timeout"
    except FileNotFoundError:
        return "[error] ollama CLI not found"
    except Exception as e:
        return f"[error] {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Bangla paragraph detector using dataset + optional model confirmation."
    )
    parser.add_argument("paragraph", help="Bangla paragraph")
    parser.add_argument("--model", "-m", default="mybanglamodel")
    parser.add_argument("--data", "-d", default="train_data.jsonl")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    examples = load_examples_from_jsonl(args.data)
    paragraph = normalize_text(args.paragraph)
    matches = detect_in_paragraph(paragraph, examples)

    if not matches:
        print("No matches found.")
        return

    confirmed = []
    for sent, label in matches:
        print(f"\nSentence: {sent}")
        print(f" → Dataset label: {label}")
        if label.lower() != "none":
            resp = query_ollama_label(args.model, sent, timeout=args.timeout)
            clean = (
                sanitize_model_output(resp)
                if resp and not resp.startswith("[")
                else label
            )
            final = clean or label
        else:
            final = label
        print(f" → Final label: {final}")
        confirmed.append(final.capitalize())

    uniq = []
    seen = set()
    for l in confirmed:
        key = l.lower()
        if key not in seen:
            uniq.append(l)
            seen.add(key)

    print("\nUnique confirmed labels:", uniq)


if __name__ == "__main__":
    main()
