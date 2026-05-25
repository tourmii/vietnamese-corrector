import os
from datasets import DatasetDict, load_dataset

REPO_ID    = "tourmii/vietnamese-corrector-errors" 
TOKEN      = os.getenv("xxx")  


dataset = DatasetDict({
    "train": load_dataset("csv", data_files="train_noisy.csv", split="train"),
    "test":  load_dataset("csv", data_files="test_noisy.csv",  split="train"),
})

print(dataset)

dataset.push_to_hub(
    REPO_ID,
    token="xxx",
)

print(f"\n Dataset pushed to https://huggingface.co/datasets/{REPO_ID}")