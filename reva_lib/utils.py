"""
Utility functions for ReVa AI Vaccine Design Library
"""

import os
import time
import datetime
import random
import string
import urllib
import requests
import warnings
import numpy as np
import pandas as pd
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.SeqUtils import IsoelectricPoint as IP
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem, rdMolTransforms
from scipy.constants import e, epsilon_0, Boltzmann
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')

# Directory paths
def get_base_path():
    """Get the base path of the library"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_asset_dir():
    """Get the asset directory path"""
    return os.path.join(get_base_path(), 'asset')

def get_model_dir():
    """Get the model directory path"""
    return os.path.join(get_asset_dir(), 'model')

def get_qmodel_dir():
    """Get the quantum model directory path"""
    return os.path.join(get_model_dir(), 'Quantum Model')

def get_label_dir():
    """Get the label directory path"""
    return os.path.join(get_asset_dir(), 'label')

def get_data_dir():
    """Get the data directory path"""
    return os.path.join(get_asset_dir(), 'data')

def get_json_dir():
    """Get the JSON directory path"""
    return os.path.join(get_asset_dir(), 'json')

# File operations
def create_folder(folder_path):
    """Create a folder if it doesn't exist"""
    try:
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")

def download_file(url, save_as):
    """Download a file from URL"""
    response = requests.get(url, stream=True, verify=False, timeout=6000)
    with open(save_as, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024000):
            if chunk:
                f.write(chunk)
                f.flush()
    return save_as

def export_string_to_text_file(string, filename):
    """Export a string to a text file"""
    with open(filename, 'w') as text_file:
        text_file.write(string)

# String and code generation
def generate_random_code(length):
    """Generate a random code of specified length"""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))

def generate_filename_with_timestamp_and_random(condition="classic"):
    """Generate filename with timestamp and random code"""
    now = datetime.datetime.now()
    date_time_str = now.strftime("%d%m%Y_%H%M%S")
    random_code = generate_random_code(15)
    
    if condition == "classic":
        filename = f"{date_time_str}_{random_code}"
    else:
        filename = f"{date_time_str}_{random_code}_quantum"
    return filename

def process_text_for_url(text_input):
    """Process text for URL encoding"""
    processed_text = text_input.replace(" ", "%20")
    processed_text = urllib.parse.quote_plus(processed_text)
    return processed_text

def request_url(url_input, text_input):
    """Make a URL request with text parameter"""
    processed_text = process_text_for_url(text_input)
    full_url = url_input + "?string_input=" + processed_text
    print(full_url)
    
    response = requests.get(full_url)
    if response.status_code == 200:
        return response.text
    else:
        return "Gagal"

# Validation functions
def minmaxint(val, max_val):
    """Validate and constrain integer values"""
    if isinstance(val, int):
        if val < 0:
            return 1
        elif val > max_val:
            return max_val
        else:
            return val
    else:
        return 1

# Sequence preprocessing
def preprocessing_begin(seq):
    """Preprocess protein sequence"""
    seq = str(seq).upper()
    delete_char = "BJOUXZ\n\t 1234567890*&^%$#@!~()[];:',.<><?/"
    for i in range(len(delete_char)):
        seq = seq.replace(delete_char[i], '')
    return seq

# Dictionary operations
def invert_dict(dictionary):
    """Invert a dictionary (swap keys and values)"""
    inverted_dict = {value: key for key, value in dictionary.items()}
    return inverted_dict

# Model loading
def load_pickle_model(model_path):
    """Load a pickle model"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_tensorflow_model(model_path):
    """Load a TensorFlow model"""
    return load_model(model_path)

# BLAST operations
def perform_blastp(query_sequence, blast_activate=False):
    """Perform BLASTp search"""
    if not blast_activate:
        return "Not Activated"
    
    start = time.time()
    try:
        result_handle = NCBIWWW.qblast("blastp", "nr", query_sequence)
    except Exception as e:
        print("BLASTp failed to connect")
        return "Skip because any error"

    print("BLASTp Starting...")
    blast_records = NCBIXML.parse(result_handle)
    for blast_record in blast_records:
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                similarity = (hsp.positives / hsp.align_length) * 100
                if similarity > 80:
                    return similarity
    
    print("BLASTp Finishing...")
    end = time.time()
    time_blast = end - start
    print(f"Time for BLASTp : {time_blast} s")
    return "Non-similarity"

# === DEVICE UTILS ===
def get_device():
    """Return device info for TensorFlow and HuggingFace (0=GPU, -1=CPU)"""
    # TensorFlow: will use GPU automatically if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return {'tf': 'GPU', 'hf': 0}
    else:
        return {'tf': 'CPU', 'hf': -1}

# LLM (HuggingFace) integration
_llm_pipeline = None

def get_llm_pipeline(model_name="microsoft/biogpt"):
    """Load and cache the LLM pipeline (default: microsoft/biogpt), using GPU if available."""
    global _llm_pipeline
    device = get_device()['hf']
    if _llm_pipeline is None or _llm_pipeline.model.name_or_path != model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        _llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return _llm_pipeline

def run_llm_review(input_text_or_path, model_name="microsoft/biogpt", max_new_tokens=256):
    """
    Run LLM review on a string or file using a local HuggingFace model.
    Args:
        input_text_or_path (str): Text to review or path to a file (CSV, TXT, etc.)
        model_name (str): HuggingFace model name (default: microsoft/biogpt)
        max_new_tokens (int): Max tokens to generate
    Returns:
        str: LLM output
    """
    if os.path.isfile(input_text_or_path):
        with open(input_text_or_path, 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        input_text = input_text_or_path
    llm = get_llm_pipeline(model_name)
    prompt = f"Review the following vaccine candidate results and provide a summary and recommendation.\n\n{input_text}\n\nReview:"
    result = llm(prompt, max_new_tokens=max_new_tokens, do_sample=True)
    return result[0]['generated_text'] 