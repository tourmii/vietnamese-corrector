import os
import kagglehub

os.environ["KAGGLE_CACHE_FOLDER"] = "../"

path = kagglehub.dataset_download("nguyenluonguy/vietnamese-news-10m")

print(path)
