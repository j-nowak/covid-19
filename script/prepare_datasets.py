import numpy as np
import os
import random
import shutil
import cv2
import urllib.request
import zipfile
import kaggle
import pandas as pd
import pydicom
import os.path
from os import path

RAW_DATA_PATH = './data/raw/'
PROCESSED_DATA_PATH = './data/processed/'

COVID_CLASS = 'covid19'
NORMAL_CLASS = 'normal'
PNEUMONIA_CLASS = 'pneumonia'

IEEE8023_DATASET_URL = 'https://github.com/ieee8023/covid-chestxray-dataset/archive/master.zip'
IEEE8023_DATASET_NAME = 'covid-chestxray-dataset'
IEEE8023_DATASET_META = 'metadata.csv'

FIGURE1_DATASET_URL = 'https://github.com/agchung/Figure1-COVID-chestxray-dataset/archive/master.zip'
FIGURE1_DATASET_NAME = 'figure1-dataset'
FIGURE1_DATASET_META = 'metadata.csv'

KAGGLE_DATASET_NAME = 'rsna-pneumonia-detection-challenge'
KAGGLE_DATASET_NORMAL_META = 'stage_2_detailed_class_info.csv'
KAGGLE_DATASET_PNEUMONIA_META = 'stage_2_train_labels.csv'


def download_github_dataset_zip(dataset_url, dataset_name):
    dataset_path = get_raw_dataset_path(dataset_name)
    if path.exists(dataset_path):
        return

    dataset_zip = dataset_name + '.zip'
    urllib.request.urlretrieve(dataset_url, RAW_DATA_PATH + dataset_zip)
    with zipfile.ZipFile(RAW_DATA_PATH + dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_PATH)
    os.remove(RAW_DATA_PATH + dataset_zip)
    os.rename(dataset_path + '-master', dataset_path)


def download_kaggle_competition(dataset_name):
    dataset_path = get_raw_dataset_path(dataset_name)
    if path.exists(dataset_path):
        return

    os.mkdir(dataset_path)
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(dataset_name, path=dataset_path)

    dataset_zip = dataset_name + '.zip'
    with zipfile.ZipFile(dataset_path + dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_PATH)
    os.remove(dataset_path + dataset_zip)


def get_raw_dataset_path(dataset_name):
    return RAW_DATA_PATH + dataset_name + '/'


def get_metadata_path(dataset_name, metadata_name):
    return get_raw_dataset_path(dataset_name) + metadata_name


def copy_images(class_type, images):
    target = PROCESSED_DATA_PATH + class_type + '/'
    for original_path, original_name in images:
        shutil.copyfile(original_path, target + original_name)


def copy_dcm_images(class_type, images):
    target = PROCESSED_DATA_PATH + class_type + '/'
    for original_path, original_name in images:
        ds = pydicom.filereader.dcmread(original_path)
        pixel_array_numpy = ds.pixel_array

        target_image_name = original_name + '.png'
        cv2.imwrite(target + target_image_name, pixel_array_numpy)


def process_ieee8023_meta():
    VIEW_TYPES = ['PA', 'AP']
    MODALITY_TYPES = ['X-ray']
    IGNORE_MAPPINGS = ['COVID-19, ARDS']

    metadata_path = get_metadata_path(IEEE8023_DATASET_NAME, IEEE8023_DATASET_META)

    metadata = pd.read_csv(metadata_path, nrows=None, usecols=['finding', 'view', 'modality', 'folder', 'filename'])
    metadata_filter = metadata['view'].isin(VIEW_TYPES) & metadata['modality'].isin(MODALITY_TYPES) & ~metadata['finding'].isin(IGNORE_MAPPINGS)
    return metadata[metadata_filter]


def process_figure1_meta():
    metadata_path = get_metadata_path(FIGURE1_DATASET_NAME, FIGURE1_DATASET_META)

    metadata = pd.read_csv(metadata_path, encoding='ISO-8859-1', nrows=None)
    metadata_filter = metadata['finding'].eq('COVID-19')
    return metadata[metadata_filter]

def process_rsna_normal_meta(rows_limit=1000):
    metadata_path = get_metadata_path(KAGGLE_DATASET_NAME, KAGGLE_DATASET_NORMAL_META)

    metadata = pd.read_csv(metadata_path, nrows=None)
    metadata_filter = metadata['class'].eq('Normal')
    return metadata[metadata_filter].head(rows_limit)

def process_rsna_pneumonia_meta(rows_limit=1000):
    metadata_path = get_metadata_path(KAGGLE_DATASET_NAME, KAGGLE_DATASET_PNEUMONIA_META)

    metadata = pd.read_csv(metadata_path, nrows=None)
    metadata_filter = metadata['Target'].eq(1)
    return metadata[metadata_filter].head(rows_limit)


def split_ieee8023_images(metadata):
    dataset_path = get_raw_dataset_path(IEEE8023_DATASET_NAME)

    covid_images = []
    normal_images = []
    pneumonia_images = []
    for _, row in metadata.iterrows():
        image_path = dataset_path + row['folder'] + '/' + row['filename']
        if row['finding'] == 'COVID-19':
            covid_images.append((image_path, row['filename']))
        elif row['finding'] == 'No Finding':
            normal_images.append((image_path, row['filename']))
        else:
            pneumonia_images.append((image_path, row['filename']))

    return covid_images, normal_images, pneumonia_images


def split_figure1_images(metadata):
    dataset_path = get_raw_dataset_path(FIGURE1_DATASET_NAME)

    covid_images = []
    for _, row in metadata.iterrows():
        extension = '.jpg' if row['patientid'] not in ['COVID-00015a', 'COVID-00015b'] else '.png'
        image_path = dataset_path + 'images/' + row['patientid'] + extension
        covid_images.append((image_path, row['patientid'] + extension))

    return covid_images


def split_rsna_challenge_images(metadata):
    dataset_path = get_raw_dataset_path(KAGGLE_DATASET_NAME)

    filtered_images = []
    for _, row in metadata.iterrows():
        image_path = dataset_path + 'stage_2_train_images/' + row['patientId'] + '.dcm'
        filtered_images.append((image_path, row['patientId']))

    return filtered_images


def process_ieee8023_images():
    metadata = process_ieee8023_meta()

    covid_images, normal_images, pneumonia_images = split_ieee8023_images(metadata)

    copy_images(COVID_CLASS, covid_images)
    copy_images(NORMAL_CLASS, normal_images)
    copy_images(PNEUMONIA_CLASS, pneumonia_images)


def process_figure1_images():
    metadata = process_figure1_meta()

    covid_images = split_figure1_images(metadata)

    copy_images(COVID_CLASS, covid_images)


def process_rsna_challenge_images():
    normal_metadata = process_rsna_normal_meta()
    normal_images = split_rsna_challenge_images(normal_metadata)
    copy_dcm_images(NORMAL_CLASS, normal_images)

    pneumonia_metadata = process_rsna_pneumonia_meta()
    pneumonia_images = split_rsna_challenge_images(pneumonia_metadata)
    copy_dcm_images(PNEUMONIA_CLASS, pneumonia_images)



def prepare_ieee8023_dataset():
    print('Prepare IEEE8023 dataset')
    download_github_dataset_zip(IEEE8023_DATASET_URL, IEEE8023_DATASET_NAME)
    process_ieee8023_images()


def prepare_figure1_dataset():
    print('Prepare FIGURE1 dataset')
    download_github_dataset_zip(FIGURE1_DATASET_URL, FIGURE1_DATASET_NAME)
    process_figure1_images()


def prepare_rsna_challenge_dataset():
    print('Prepare RSNA_CHALLENGE dataset')
    process_rsna_challenge_images()
