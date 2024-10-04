import os
import requests
import zipfile
import logging
import shutil


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

LOGGER = logging.getLogger(__name__)
BASE_DIR = "data/ISIC2017"
URLS = {
    "train_set": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip",
    "train_masks": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip",
    "val_set": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip",
    "val_masks": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip",
    "test_set": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip",
    "test_masks": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip"
}


def ensure_dir(dir: str) -> bool:
    """ Ensures specified dir exists, tries to create if not
    Returns: success state for the creation of directory
    """
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except:
        return False
    return True
    

def download_file(url: str, dest_folder: str) -> str:
    """ downloads the data with 1kb chunks to the destination folder
    Returns: The filename of the downloaded file
    """
    fn = os.path.join(dest_folder, url.split("/")[-1])
    LOGGER.info(f"Starting download for {fn} from {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # raise http error if something is wrong with the connection
        with open(fn, "wb") as f:
            for chunk in r.iter_content(chunk_size=2 ** 13):  # 2^10 * 8 bits = 1 kb per chunk
                f.write(chunk)
    LOGGER.info(f"Download completed: {fn}")
    return fn


def unzip(zip_path: str, dest_folder: str, name: str):
    """ unzips the file given and deletes the zip file"""
    with zipfile.ZipFile(zip_path, "r") as zip:
        zip.extractall(dest_folder)
    os.remove(zip_path)  # remove downloaded zip as it is extracted and unnecessary space

    renamed_path = os.path.join(dest_folder, name)
    shutil.move(zip_path.split(".zip")[0], renamed_path)

    LOGGER.info(F"Unzipped and deleted {zip_path}")


def process_dataset(name: str, url: str, base_folder: str) -> bool:
    LOGGER.info(f"Processing {name}")

    dataset_path = os.path.join(base_folder, name)
    if os.path.exists(dataset_path):
        LOGGER.info(f"{name} already exists. Skipping download and extraction.")
        return False

    zip_file = download_file(url, base_folder)
    unzip(zip_file, base_folder, name)
    LOGGER.info(f"{name} processed")
    return True


def download_set(set_type: str):
    """ set_type: str -> Either train, val, or test """
    ensure_dir(BASE_DIR)

    process_dataset(f"{set_type}_set", URLS[f"{set_type}_set"], BASE_DIR)
    process_dataset(f"{set_type}_masks", URLS[f"{set_type}_masks"], BASE_DIR)


