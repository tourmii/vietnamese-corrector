import os
import sys
import time
import subprocess
from pathlib import Path

import torch

# =========================
# 1. Setup repo
# =========================
REPO_DIR = Path("/kaggle/working/VietnameseOcrCorrection")

if not REPO_DIR.exists():
    subprocess.check_call([
        "git", "clone",
        "https://github.com/buiquangmanhhp1999/VietnameseOcrCorrection.git",
        str(REPO_DIR)
    ])

sys.path.insert(0, str(REPO_DIR))
os.chdir(str(REPO_DIR))

# Nếu thiếu dependency thì chạy cell này trước:
# !pip install -q unidecode lmdb nltk tqdm

from config import alphabet
from model.vocab import Vocab
from model.seq2seq import Seq2Seq

# =========================
# 2. Config
# =========================
WEIGHT_PATH = Path("/kaggle/input/datasets/khanhchipham/seq2seq-rnn/seq2seq_tourmii_best_rougeL.pth")

device = torch.device("cpu")   # dùng CPU cho an toàn infer 1 câu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert WEIGHT_PATH.exists(), f"Không tìm thấy weight: {WEIGHT_PATH}"

# =========================
# 3. Load vocab + model đúng size
# =========================
vocab = Vocab(alphabet)
VOCAB_SIZE = len(vocab)   # quan trọng: dùng len(vocab), không dùng len(alphabet)

print("len(alphabet):", len(alphabet))
print("len(vocab):", VOCAB_SIZE)

model = Seq2Seq(
    VOCAB_SIZE,
    encoder_hidden=256,
    decoder_hidden=256
)

ckpt = torch.load(str(WEIGHT_PATH), map_location="cpu")

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    ckpt = ckpt["model_state_dict"]
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]

ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

model.load_state_dict(ckpt, strict=True)
model.to(device)
model.eval()

print("Loaded weight:", WEIGHT_PATH)

# =========================
# 4. Greedy inference 1 câu
# =========================
def safe_decode_ids(ids, vocab):
    chars = []

    pad_id = getattr(vocab, "pad", 0)
    go_id = getattr(vocab, "go", 1)
    eos_id = getattr(vocab, "eos", 2)
    mask_id = getattr(vocab, "mask", None)

    special_ids = [pad_id, go_id, eos_id]
    if mask_id is not None:
        special_ids.append(mask_id)

    for idx in ids:
        idx = int(idx)

        if idx == eos_id:
            break

        if idx in special_ids:
            continue

        if idx in vocab.i2c:
            chars.append(vocab.i2c[idx])

    return "".join(chars).strip()


def unpack_decoder_output(decoder_result):
    # decoder repo thường trả về: output, hidden, attention
    if isinstance(decoder_result, tuple):
        return decoder_result[0], decoder_result[1]
    raise TypeError(f"Unexpected decoder output type: {type(decoder_result)}")


@torch.no_grad()
def correct_one(text, max_extra_len=20, max_len=256):
    text = str(text).strip()

    if not text:
        return ""

    src_ids = vocab.encode(text)

    # chặn id lỗi
    src_ids = [int(x) for x in src_ids if 0 <= int(x) < VOCAB_SIZE]

    src_tensor = torch.LongTensor([src_ids]).to(device)  # shape: [batch, seq_len]

    encoder_outputs, hidden = model.encoder(src_tensor)

    input_token = torch.LongTensor([vocab.go]).to(device)
    output_ids = []

    decode_len = min(len(src_ids) + max_extra_len, max_len)

    for _ in range(decode_len):
        decoder_result = model.decoder(input_token, hidden, encoder_outputs)
        output, hidden = unpack_decoder_output(decoder_result)

        next_id = int(output.argmax(1).item())

        if next_id == vocab.eos:
            break

        if next_id < 0 or next_id >= VOCAB_SIZE:
            break

        output_ids.append(next_id)
        input_token = torch.LongTensor([next_id]).to(device)

    return safe_decode_ids(output_ids, vocab)

# =========================
# 5. Test 1 câu
# =========================
sample_text = "Trên cơsở kếT quả kiểm tra hiện trạng, Tòa an nhân dân tối cao xẻm xét"

print("Input:")
print(sample_text)
print("-" * 50)

start = time.time()
output = correct_one(sample_text)
end = time.time()

print("Output:")
print(output)
print("-" * 50)
print(f"Inference time: {end - start:.4f} seconds")
if __name__ == "__main__":
    main()