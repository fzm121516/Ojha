import zipfile
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def unzip_file(zip_file, target_dir, file_name):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extract(file_name, target_dir)

def main(zip_file, target_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        file_list = zip_ref.namelist()

    os.makedirs(target_dir, exist_ok=True)

    with ThreadPoolExecutor() as executor:
        with tqdm(total=len(file_list), desc="Unzipping", unit="file") as pbar:
            futures = {executor.submit(unzip_file, zip_file, target_dir, file_name): file_name for file_name in file_list}
            for future in futures:
                future.add_done_callback(lambda p: pbar.update(1))

if __name__ == "__main__":
    zip_file = "path/to/your/file.zip"  # 替换为你的zip文件路径
    target_dir = "path/to/extract"       # 替换为目标目录路径
    main(zip_file, target_dir)
