"""W6 環境確認スクリプト(all_src/W6/check_env.py に置いて `uv run python check_env.py`)

確認すること:
  1. torch / torchvision が入っていて、MNIST がダウンロードできるか
  2. 画像1枚の shape と値域
  3. トークナイザ3種(char / tiktoken / llm-jp)が動くか
  4. W2 の model.py が W6 から import できるか
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
W2_DIR = HERE.parent / "W2"  # W4 と同じ参照方法に合わせて必要なら書き換える


def section(title):
    print(f"\n=== {title} ===")


# 1. torch / torchvision
section("1. torch / torchvision")
import torch
print("torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
try:
    import torchvision
    from torchvision import datasets, transforms
    print("torchvision:", torchvision.__version__)
except ImportError:
    sys.exit("torchvision がない → `uv add torchvision`")

# 2. MNIST
section("2. MNIST")
ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
x, y = ds[0]
print("len(train):", len(ds))
print("x.shape:", tuple(x.shape), "dtype:", x.dtype)
print("x.min/max:", round(x.min().item(), 3), round(x.max().item(), 3))
print("label:", y)

# 3. トークナイザ
section("3. トークナイザ")
samples = {
    "ja": "吾輩は猫である。名前はまだ無い。",
    "en": "To be, or not to be, that is the question.",
    "code": "def forward(self, x):\n    return self.proj(x) + x",
}
for name, s in samples.items():
    print(f"[{name}] char tokens: {len(s)}")

try:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    for name, s in samples.items():
        print(f"[{name}] tiktoken(o200k) tokens: {len(enc.encode(s))}")
except ImportError:
    print("tiktoken なし → `uv add tiktoken`")

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-1.8b")
    for name, s in samples.items():
        print(f"[{name}] llm-jp tokens: {len(tok.encode(s, add_special_tokens=False))}")
except Exception as e:  # 未インストール・ネットワーク・モデル名違いをまとめて拾う
    print("llm-jp tokenizer 読み込み失敗:", type(e).__name__, e)

# 4. W2 の胴体を import できるか
import os, sys, csv

current_dir = os.path.dirname(os.path.abspath(__file__))
W2_DIR = os.path.abspath(os.path.join(current_dir, "..", "W2"))
sys.path.insert(0, W2_DIR)

import torch
import torch.nn.functional as F
import model
try: # noqa: F401
    print("W2 model.py import OK:", [n for n in dir(model) if not n.startswith("_")])
except Exception as e:
    print("W2 import 失敗:", type(e).__name__, e)

print("\n全部終わり。出力を貼ってください。")