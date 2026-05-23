import os
import kagglehub

os.environ["KAGGLE_CACHE_FOLDER"] = "../"

path = kagglehub.dataset_download("nguyenluonguy/old-news-vn-dataset-cleaned")

print(path)
