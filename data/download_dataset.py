import os
import tarfile
import urllib.request
import shutil
from config import url, tar_filename, extracted_folder

# Download the tar file
def download_data():
    urllib.request.urlretrieve(url, tar_filename)
    print(f"Downloaded {tar_filename}. Extraction will begin now.")

    # Check if the folder already exists
    if os.path.exists(extracted_folder):
        print(f"The folder '{extracted_folder}' already exists. Removing the existing folder.")
        
        # Remove the existing folder to avoid overwriting or duplication
        shutil.rmtree(extracted_folder)
        print(f"Removed the existing folder: {extracted_folder}")

    # Extract the contents of the tar file
    with tarfile.open(tar_filename, "r") as tar_ref:
        tar_ref.extractall()  # This will extract to the current directory
        print(f"Extracted {tar_filename} successfully.")