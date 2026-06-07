# Vietnamese Corrector (BiLSTM Model)

This is the inference source code for the Vietnamese spelling correction model using the BiLSTM architecture (based on the AIVIVN solution).

## 1. Checkpoint Preparation (Required)
Before running the model, you **must** download the model weights (checkpoint) and vocabulary files from Google Drive; otherwise, the code will raise a missing file error.

- **Google Drive Link:** `https://drive.google.com/file/d/1dSvcjz08pVKW8oFsrxZbkFLlM8UB-hwH/view?usp=sharing`

After downloading, please extract and place all the files (such as `vocab.src`, `vocab.tgt`, `bilstm_model`) into the `checkpoint/` directory so that the structure looks like this:
```text
vietnamese-corrector/models/bilstm/
├── checkpoint/
│   ├── bilstm_model
│   ├── vocab.src
│   └── vocab.tgt
├── lm/
│   └── corpus-wplm-4g-v2.binary
├── data/
│   └── legal_vc.txt
├── infer.py
...
```


## 2. Running Inference (Single Sentence)
To correct the spelling of any sentence, open your Terminal, navigate to this directory, and run the following command:

```bash
python infer.py "hom nay thoi tiet dep qua, toi muon di choi."
```

**The output will look like this:**
```text
Loading vocab...
Loading model...
Input: hom nay thoi tiet dep qua, toi muon di choi.
Output: hôm nay thời tiết đẹp quá, tôi muốn đi chơi.
```

## 3. Important Files
- `infer.py`: The main script used to run inference.
- `model.py`: Contains the definition of the Seq2Seq network architecture (Encoder, Decoder, Attention).
- `dataset.py` & `alphabet.py`: Contain preprocessing logic and valid character sets.
