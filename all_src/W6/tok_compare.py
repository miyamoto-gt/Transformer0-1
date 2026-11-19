import os,sys
from transformers import AutoTokenizer
import tiktoken
current_dir = os.path.dirname(os.path.abspath(__file__))
W2_DIR = os.path.abspath(os.path.join(current_dir, "..", "W2"))
sys.path.insert(0, W2_DIR)
from data_utils import load_data

samples = {
    "ja": "吾輩は猫である。名前はまだ無い。",
    "en": "To be, or not to be, that is the question.",
    "code": "def forward(self, x):\n    return self.proj(x) + x",
}

def load_tokenizers():
    train, val, stoi, itos  = load_data(os.path.join(W2_DIR, "data", "input.txt"))
    enc = tiktoken.get_encoding("o200k_base")
    tok = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-1.8b")
    return stoi,enc,tok
def count(tok_name, s):
    if tok_name=="char":
        try:
            return len([stoi[c] for c in s])
        except KeyError:
            return "KeyError"
    elif tok_name=="tiktoken":
        return len(enc.encode(s))
    elif tok_name=="llmjp":
        return len(tok.encode(s, add_special_tokens=False))
    else:
        raise ValueError(f"nameはchar tiktoken llmjp のどれか: {tok_name}")
def pieces(tok_name,s):
    if tok_name=="tiktoken":
        ids = enc.encode(s)
        for i in ids:
            return [enc.decode([i]) for i in ids]
    elif tok_name=="llmjp":
        ids = tok.encode(s, add_special_tokens=False)
        return tok.convert_ids_to_tokens(ids)
    else:
            raise ValueError(f"nameは tiktoken llmjp のどれか: {tok_name}")
# 3つの文を、文字数・バイト数・各トークナイザのトークン数で比べる(char(W2) は語彙にない文字も表示)
print(f"{'text':>4} | {'chars':>5} | {'bytes':>5} | {'char(W2)':>8} | {'tiktoken':>8} | {'llm-jp':>6} | {'missing':>7} | {'missing_text'} ")
stoi, enc, tok = load_tokenizers()

for name,s in samples.items():
    c=count("char", s)
    t=count("tiktoken", s)
    l=count("llmjp", s)
    cs=len(s)
    b=len(s.encode("utf-8"))
    missing = sorted(set(ch for ch in s if ch not in stoi))
    miss_str = "".join(missing)
    print(f"{name:>4} | {cs:>5} | {b:>5} | {c:>8} | {t:>8} | {l:>6} | {len(missing):>7} | {miss_str}")
# 各トークナイザの区切り方(断片)を表示する
for text_name, s in samples.items():
    tt = pieces("tiktoken", s)
    lt = pieces("llmjp", s)
    print(f"[{text_name}] tiktoken({len(tt)}): " + " | ".join(repr(p) for p in tt))
    print(f"[{text_name}] llmjp({len(lt)}): " + " | ".join(repr(p) for p in lt))

# llm-jp トークナイザの方式(BPE / Unigram)を表示する
print(type(tok.backend_tokenizer.model).__name__)