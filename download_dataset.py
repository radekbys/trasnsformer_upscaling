import kagglehub
import shutil
import os

path = kagglehub.dataset_download("joe1995/div2k-dataset")
file_names = os.listdir(path)

for file_name in file_names:
    shutil.move(os.path.join(path, file_name), "./dataset")
