import sys
import os
import time
import requests
# from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QTableWidget, QTableWidgetItem, QLineEdit, QPlainTextEdit
# from PyQt5.QtGui import QPixmap
# from PyQt5 import QtCore
# from PyQt5 import QtGui, QtWidgets
# from PyQt5.QtWidgets import *
# import openpyxl
import warnings
import sys
import numpy as np
import json
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio.SeqUtils import IsoelectricPoint as IP
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
# from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from scipy.constants import e, epsilon_0
from scipy.constants import Boltzmann
import datetime
import random
import string
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.constants import e
import pandas as pd
from rdkit.Chem import rdMolTransforms
# from collections import Counter
# from qiskit.algorithms.optimizers import COBYLA
# from qiskit.circuit.library import TwoLocal, ZZFeatureMap
# from qiskit.utils import algorithm_globals
# from qiskit import BasicAer
# from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import VQC, VQR
# import qiskit
import urllib
# from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session

# algorithm_globals.random_seed = 42
warnings.filterwarnings('ignore')

asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asset')
image_dir = os.path.join(asset_dir, 'img')
model_dir = os.path.join(asset_dir, 'model')
qmodel_dir = os.path.join(model_dir, 'Quantum Model')
label_dir = os.path.join(asset_dir, 'label')
data_dir = os.path.join(asset_dir, 'data')
json_dir = os.path.join(asset_dir, 'json')

icon_img = os.path.join(image_dir, 'kaede_kayano.jpg')

def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))

def ml_dock(data):
    # Load model
    with open(os.path.join(model_dir, 'Linear_Regression_Model.pkl'), "rb") as f:
        model = pickle.load(f)

    # Make predictions
    predictions = model.predict([data])

    return predictions[0]

def qml_dock(data, sampler=None):
    try:
        if sampler == None:
            #load model
            vqr = VQR.load(os.path.join(qmodel_dir, "VQR_quantum regression-based scoring function"))

            prediction = vqr.predict([data])
            return prediction[0][0]
        else:
            #load model
            vqr = VQR.load(os.path.join(qmodel_dir, "VQR_quantum regression-based scoring function"))
            # vqr.neural_network.sampler = sampler

            prediction = vqr.predict([data])
            return prediction[0][0]
    except:
        return None

def generate_random_code(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))

def generate_filename_with_timestamp_and_random(condition="classic"):
    # Dapatkan tanggal dan jam sekarang
    now = datetime.datetime.now()

    # Format tanggal dan jam
    date_time_str = now.strftime("%d%m%Y_%H%M%S")

    # Generate kode acak
    random_code = generate_random_code(15)  # Ubah panjang kode acak sesuai kebutuhan

    if condition == "classic":
        # Gabungkan hasilnya
        filename = f"{date_time_str}_{random_code}"
        return filename
    else:
        # Gabungkan hasilnya
        filename = f"{date_time_str}_{random_code}_quantum"
        return filename

def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")

def minmaxint(val, max_val):
    if isinstance(val, int):
        if val < 0:
            return 1
        elif val > max_val:
            return max_val
        else:
            return val
    else:
        return 1

def download_file(url, save_as):
	response = requests.get(url, stream=True, verify=False, timeout=6000)
	with open(save_as, 'wb') as f:
		for chunk in response.iter_content(chunk_size=1024000):
			if chunk:
				f.write(chunk)
				f.flush()
	return save_as

def process_text_for_url(text_input):
  """
  Fungsi untuk mengubah teks input agar sesuai dengan format URL.

  Args:
      text_input (str): Teks yang ingin diubah.

  Returns:
      str: Teks yang telah diubah menjadi format URL yang sesuai.
  """
  # Ganti spasi dengan tanda plus (+)
  processed_text = text_input.replace(" ", "%20")

  # URL encode teks
  processed_text = urllib.parse.quote_plus(processed_text)

  return processed_text

def request_url(url_input, text_input):
  """
  Fungsi untuk melakukan request URL dengan parameter teks.

  Args:
      url_input (str): URL yang ingin diakses.
      text_input (str): Teks yang akan digunakan sebagai request.

  Returns:
      Response: Objek respons dari request URL.
  """

  # Ubah teks input sesuai dengan format URL
  processed_text = process_text_for_url(text_input)

  # Buat URL lengkap dengan parameter teks
  full_url = url_input + "?string_input="+ processed_text
  print(full_url)

  # Lakukan request URL
  response = requests.get(full_url)

  if response.status_code == 200:
     result = response.text
     return result
  else:
     return "Gagal"

def export_string_to_text_file(string, filename):
  """Exports a string to a text file.

  Args:
    string: The string to export.
    filename: The filename to save the string to.
  """
  with open(filename, 'w') as text_file:
    text_file.write(string)

print("Starting App...")

class ReVa:
    def preprocessing_begin(seq):
        seq = str(seq).upper()
        delete_char = "BJOUXZ\n\t 1234567890*&^%$#@!~()[];:',.<><?/"
        for i in range(len(delete_char)):
            seq = seq.replace(delete_char[i],'')
        return seq


    def __init__(self, sequence, base_path, target_path, n_receptor, n_adjuvant, blast_activate=False, llm_url="", alphafold_url=""):
        self.sequence = ReVa.preprocessing_begin(sequence)
        self.base_path = base_path
        self.blast_activate = blast_activate
        self.target_path = target_path
        self.n_receptor = n_receptor
        self.n_adjuvant = n_adjuvant
        self.llm_url = llm_url
        self.alphafold_url = alphafold_url
        create_folder(self.target_path)

        self.alphabet = 'ACDEFGHIKLMNPQRSTVWY'  # Hanya huruf-huruf asam amino yang relevan
        self.num_features = len(self.alphabet)

        try:
            model_path = 'BPepTree.pkl'
            self.loaded_Bmodel = ReVa.load_pickle_model(self.base_path, model_path)
        except:
            print("Error Load Model Epitope")
            # QMessageBox.warning(self, "Error", f"Error on Load Model Epitope B")
        
        try:
            model_path = 'TPepTree.pkl'
            self.loaded_Tmodel = ReVa.load_pickle_model(self.base_path, model_path)
        except:
            print("Error")
            # QMessageBox.warning(self, "Error", f"Error on load model Epitope T")

        try:
            with open(os.path.join(label_dir, 'allergenicity_label_mapping.json'), 'r') as f:
                label_dict = json.load(f)
        except:
            print("Error")
            # QMessageBox.warning(self, "Error", f"Error on load label allergenisitas")

        self.reverse_label_mapping_allergen = {v: k for k, v in label_dict.items()}
        self.seq_length_allergen = 4857

        try:
            with open(os.path.join(label_dir,'toxin_label_mapping.json'), 'r') as f:
                label_dict = json.load(f)
        except:
            print("Error")
            # QMessageBox.warning(self, "Error", f"Error on load label toxin")

        self.reverse_label_mapping_toxin = {v: k for k, v in label_dict.items()}
        self.seq_length_toxin = 35

        try:
            with open(os.path.join(label_dir, 'antigenicity_label_mapping.json'), 'r') as f:
                label_dict = json.load(f)
        except:
            print("Error")
            # QMessageBox.warning(self, "Error", f"Error on load label antigenisitas")

        self.reverse_label_mapping_antigen = {v: k for k, v in label_dict.items()}
        self.seq_length_antigen = 83

        try:
            with open(os.path.join(label_dir,'BPepTree_label.json'), 'r') as f:
                label_dict = json.load(f)
            self.Blabel = ReVa.invert_dict(label_dict)
        except:
            print("Error")
            # QMessageBox.warning(self, "Error", f"Error on load label Epitope B")

        try:
            with open(os.path.join(label_dir,'TPepTree_label.json'), 'r') as f:
                label_dict = json.load(f)
            self.Tlabel = ReVa.invert_dict(label_dict)
        except:
            print("Error")
            # QMessageBox.warning(self, "Error", f"Error on load label Epitope T")

    def load_pickle_model(base_path, model_path):
        with open(os.path.join(model_dir, model_path), 'rb') as f:
            model = pickle.load(f)
        return model

    def combine_lists(list1, list2):
        result = []
        current_group = ""

        for i in range(len(list1)):
            if list2[i] == 'E':
                current_group += list1[i]
            else:
                if current_group:
                    result.append(current_group)
                    current_group = ""
                result.append(list1[i])

        if current_group:
            result.append(current_group)

        return result

    def engelman_ges_scale(aa):
        scale = {
            'A': 1.60, 'C': 2.00, 'D': -9.20, 'E': -8.20, 'F': 3.70,
            'G': 1.00, 'H': -3.00, 'I': 3.10, 'K': -8.80, 'L': 2.80,
            'M': 3.40, 'N': -4.80, 'P': -0.20, 'Q': -4.10, 'R': -12.3,
            'S': 0.60, 'T': 1.20, 'V': 2.60, 'W': 1.90, 'Y': -0.70
        }
        return scale.get(aa, 0.0)

    def get_position(aa):
        pos = [i+1 for i in range(0, len(aa))]
        return pos

    def one_hot_encoding(self, sequence):
        encoding = []
        for char in sequence:
            vector = [0] * self.num_features
            if char in self.alphabet:
                index = self.alphabet.index(char)
                vector[index] = 1
            encoding.append(vector)
        return encoding

    def extraction_feature(aa):
        pos = ReVa.get_position(aa)
        scale = [ReVa.engelman_ges_scale(aa[i]) for i in range(len(aa))]

        res = [[pos[i], scale[i], len(aa)] for i in range(len(pos))]

        return res

    def predict_label_and_probability_allergenicity(self, sequence):
        try:
            model_path = 'allerginicity.h5'
            model = load_model(os.path.join(model_dir, model_path))
        except:
            print("Error")
            # QMessageBox.warning(self, "Error", f"Error on load model allergenicity")

        try:
            sequence = sequence[:self.seq_length_allergen]
            sequence = [ReVa.one_hot_encoding(self, seq) for seq in [sequence]]
            sequence = [seq + [[0] * self.num_features] * (self.seq_length_allergen - len(seq)) for seq in sequence]
            sequence = np.array(sequence)
        except:
            print("Error")
            # # QMessageBox.warning(self, "Error", f"Gagal Preprocess sebelum prediksi Allergenisitas")
        
        try:
            prediction = model.predict(sequence)[0]
            predicted_label_index = 1 if prediction > 0.5 else 0
            predicted_label = self.reverse_label_mapping_allergen[predicted_label_index]

            return predicted_label, prediction
        except:
            print("Error")
            # # QMessageBox.warning(self, "Error", f"Gagal Prediksi Allergenisitas")

    def predict_label_and_probability_toxin(self, sequence):
        try:
            model_path = 'toxin.h5'
            model = load_model(os.path.join(model_dir, model_path))
        except:
            print("Error")
            # QMessageBox.warning(self, "Error", f"Error on load model toxin")

        sequence = sequence[:self.seq_length_toxin]
        sequence = [ReVa.one_hot_encoding(self, seq) for seq in [sequence]]
        sequence = [seq + [[0] * self.num_features] * (self.seq_length_toxin - len(seq)) for seq in sequence]
        sequence = np.array(sequence)
        
        try:
            prediction = model.predict(sequence)[0]
            predicted_label_index = 1 if prediction > 0.5 else 0
            predicted_label = self.reverse_label_mapping_toxin[predicted_label_index]
            
            return predicted_label, prediction
        except:
            print("Error")
            # QMessageBox.warning(self, "Error", f"Error on predict toxin")
    
    def predict_label_and_probability_antigenicity(self, sequence):
        try:
            model_path = 'antigenicity.h5'
            model = load_model(os.path.join(model_dir, model_path))
        except:
            print("Error")
            # QMessageBox.warning(self, "Error", f"Error on load model antigenisitas")

        sequence = sequence[:self.seq_length_antigen]
        sequence = [ReVa.one_hot_encoding(self, seq) for seq in [sequence]]
        sequence = [seq + [[0] * self.num_features] * (self.seq_length_antigen - len(seq)) for seq in sequence]
        sequence = np.array(sequence)

        try:
            prediction = model.predict(sequence)[0]
            predicted_label_index = 1 if prediction > 0.5 else 0
            predicted_label = self.reverse_label_mapping_antigen[predicted_label_index]
            
            return predicted_label, prediction
        except:
            print("Error")
            # # QMessageBox.warning(self, "Error", f"Error on prediksi antigenisitas")

    def invert_dict(dictionary):
        inverted_dict = {value: key for key, value in dictionary.items()}
        return inverted_dict

    def process_epitope(input_list):
        output_list = []
        current_group = []

        for item in input_list:
            if item == 'E':
                current_group.append(item)
            else:
                if current_group:
                    output_list.append(''.join(current_group))
                    current_group = []
                output_list.append(item)

        if current_group:
            output_list.append(''.join(current_group))

        return output_list

    def filter_epitope(data):
        filtered_seq = []
        filtered_label = []

        for i in range(len(data['seq'])):
            if data['label'][i] != '.':
                filtered_seq.append(data['seq'][i])
                filtered_label.append(data['label'][i])

        filtered_data = {'seq': filtered_seq, 'label': filtered_label}
        return filtered_data
    
    def string_to_list(input_string):
        return list(input_string)
    
    def calculate_hydrophobicity(sequence):
        hydrophobic_residues = ['A', 'I', 'L', 'M', 'F', 'V', 'W', 'Y']
        hydrophilic_residues = ['R', 'N', 'C', 'Q', 'E', 'G', 'H', 'K', 'S', 'T', 'D']
        hydrophobicity_scores = {
            'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
            'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
            'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
            'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
        }
        
        hydrophobicity = 0
        for residue in sequence:
            if residue in hydrophobic_residues:
                hydrophobicity += hydrophobicity_scores[residue]
            elif residue in hydrophilic_residues:
                hydrophobicity -= hydrophobicity_scores[residue]
            else:
                hydrophobicity -= 0.5  # penalty for unknown residues
        
        return hydrophobicity / len(sequence)

    def antigenicity(sequence, window_size=7):
        antigenicity_scores = []
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            antigenicity_score = sum([1 if window[j] == 'A' or window[j] == 'G' else 0 for j in range(window_size)])
            antigenicity_scores.append(antigenicity_score)
        return antigenicity_scores
    
    def emini_surface_accessibility(sequence, window_size=9):
        surface_accessibility_scores = []
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            surface_accessibility_score = sum([1 if window[j] in ['S', 'T', 'N', 'Q'] else 0 for j in range(window_size)])
            surface_accessibility_scores.append(surface_accessibility_score)
        return surface_accessibility_scores
    
    def perform_blastp(query_sequence, self):
        # return "N/A"
        # Lakukan BLASTp
        if self.blast_activate == True:
            start = time.time()
            try:
                result_handle = NCBIWWW.qblast("blastp", "nr", query_sequence)
            except Exception as e:
                # return str(e)
                print("BLASTp failed to connect")
                return "Skip because any error"

            # Analisis hasil BLASTp
            print("BLASTp Starting...")
            blast_records = NCBIXML.parse(result_handle)
            for blast_record in blast_records:
                for alignment in blast_record.alignments:
                    # Anda dapat menyesuaikan kriteria kesamaan sesuai kebutuhan
                    # Misalnya, jika Anda ingin mengembalikan hasil jika similarity > 90%
                    for hsp in alignment.hsps:
                        similarity = (hsp.positives / hsp.align_length) * 100
                        if similarity > 80:
                            return similarity
            print("BLASTp Finisihing...")
            end = time.time()
            time_blast = end-start
            print(f"Time for BLASTp : {time_blast} s")
            # Jika tidak ada protein yang cocok ditemukan
            return "Non-similarity"
        else:
            return "Not Activated"
        

    def predict_epitope(self):
        seq = self.sequence
        seq_extra = ReVa.extraction_feature(seq)
        try:
            pred_res_B = [self.Blabel[self.loaded_Bmodel.predict([seq_extra[i]])[0]] for i in range(len(seq_extra))]
            pred_res_T = [self.Tlabel[self.loaded_Tmodel.predict([seq_extra[i]])[0]] for i in range(len(seq_extra))]

            pred_proba_B = [np.max(self.loaded_Bmodel.predict_proba([seq_extra[i]])[0]) for i in range(len(seq_extra))]
            pred_proba_T = [np.max(self.loaded_Tmodel.predict_proba([seq_extra[i]])[0]) for i in range(len(seq_extra))]
        except:
            print("Error on epitope predict")
            # # QMessageBox.warning(self, "Error", f"Error on prediksi epitope")

        seq_B = ReVa.combine_lists(seq, pred_res_B)
        pred_B = ReVa.process_epitope(pred_res_B)
        seq_T = ReVa.combine_lists(seq, pred_res_T)
        pred_T = ReVa.process_epitope(pred_res_T)

        pred_res1 = {
            'B': {'amino acid': ReVa.string_to_list(seq), 'predictions': pred_res_B, 'probabilities': pred_proba_B},
            'T': {'amino acid': ReVa.string_to_list(seq), 'predictions': pred_res_T, 'probabilities': pred_proba_T}
        }

        pred_res2 = {
            'B': {'seq': seq_B, 'label': pred_B},
            'T': {'seq': seq_T, 'label': pred_T}
        }

        return pred_res1, pred_res2

    def predict_eval(self, Bpred, Tpred):
        BCell = ReVa.filter_epitope(Bpred)['seq']
        TCell = ReVa.filter_epitope(Tpred)['seq']
        
        Ballergen = []
        BallergenProb = []
        for i in range(len(BCell)):
            baller, ballerprob = ReVa.predict_label_and_probability_allergenicity(self, BCell[i])
            Ballergen.append(baller)
            BallergenProb.append(ballerprob[0])

        Tallergen = []
        TallergenProb = []
        for i in range(len(TCell)):
            baller, ballerprob = ReVa.predict_label_and_probability_allergenicity(self, TCell[i])
            Tallergen.append(baller)
            TallergenProb.append(ballerprob[0])

        Btoxin = []
        BtoxinProb = []
        Ttoxin = []
        TtoxinProb = []

        for i in range(len(BCell)):
            baller, ballerprob = ReVa.predict_label_and_probability_toxin(self, BCell[i])
            Btoxin.append(baller)
            BtoxinProb.append(ballerprob[0])

        for i in range(len(TCell)):
            baller, ballerprob = ReVa.predict_label_and_probability_toxin(self, TCell[i])
            Ttoxin.append(baller)
            TtoxinProb.append(ballerprob[0])

        BAntigen = []
        BAntigenProb = []
        TAntigen = []
        TAntigenProb = []

        for i in range(len(BCell)):
            baller, ballerprob = ReVa.predict_label_and_probability_antigenicity(self, BCell[i])
            BAntigen.append(baller)
            BAntigenProb.append(ballerprob[0])

        for i in range(len(TCell)):
            baller, ballerprob = ReVa.predict_label_and_probability_antigenicity(self, TCell[i])
            TAntigen.append(baller)
            TAntigenProb.append(ballerprob[0])

        Bhydrophobicity = []
        Bkolaskar = []
        Btangonkar = []
        Bemini = []
        Bsimilar = []
        BPhysicochemical = []

        for i in range(len(BCell)):
            Bhydrophobicity.append(ReVa.calculate_hydrophobicity(BCell[i]))
            Bkolaskar.append(ReVa.antigenicity(BCell[i]))
            Btangonkar.append(ReVa.antigenicity(BCell[i], window_size=5))
            Bemini.append(ReVa.emini_surface_accessibility(BCell[i]))
            Bsimilar.append(ReVa.perform_blastp(BCell[i], self))
            BPhysicochemical.append(ProtParamClone(BCell[i]).calculate())

        Thydrophobicity = []
        Tkolaskar = []
        Ttangonkar = []
        Temini = []
        Tsimilar = []
        TPhysicochemical = []

        for i in range(len(TCell)):
            Thydrophobicity.append(ReVa.calculate_hydrophobicity(TCell[i]))
            Tkolaskar.append(ReVa.antigenicity(TCell[i]))
            Ttangonkar.append(ReVa.antigenicity(TCell[i], window_size=5))
            Temini.append(ReVa.emini_surface_accessibility(TCell[i]))
            Tsimilar.append(ReVa.perform_blastp(TCell[i], self))
            TPhysicochemical.append(ProtParamClone(TCell[i]).calculate())
        
        classical_dock1B, classical_dock1T = ClassicalDocking(BCell, TCell, self.base_path, self.target_path, self.n_receptor).ForceField1()
        classical_dock1BAdjuvant, classical_dock1TAdjuvant = ClassicalDockingWithAdjuvant(BCell, TCell, self.base_path, self.target_path, self.n_receptor, self.n_adjuvant).ForceField1()

        dock1B, dock1T = MLDocking(BCell, TCell, self.base_path, self.target_path, self.n_receptor).MLDock1()
        dock1BAdjuvant, dock1TAdjuvant = MLDockingWithAdjuvant(BCell, TCell, self.base_path, self.target_path, self.n_receptor, self.n_adjuvant).MLDock1()

        pred = {
            'seq': {
                'B':BCell,
                'T':TCell
            },

            'allergenicity' : {
                'B' : Ballergen,
                'T' : Tallergen,
                'B_proba' : BallergenProb,
                'T_proba' : TallergenProb
            },

            'toxin' : {
                'B' : Btoxin,
                'T' : Ttoxin,
                'B_proba' : BtoxinProb,
                'T_proba' : TtoxinProb
            },

            'antigenicity' : {
                'B' : BAntigen,
                'T' : TAntigen,
                'B_proba' : BAntigenProb,
                'T_proba' : TAntigenProb
            },

            'hydrophobicity' : {
                'B' : Bhydrophobicity,
                'T' : Thydrophobicity
            },
            
            'kolaskar' : {
                'B' : Bkolaskar,
                'T' : Tkolaskar
            },

            'tangonkar' : {
                'B' : Btangonkar,
                'T' : Ttangonkar
            },

            'emini' : {
                'B' : Bemini,
                'T' : Temini
            },

            'similarity' : {
                'B' : Bsimilar,
                'T' : Tsimilar
            },

            'physicochemical' : {
                'B' : BPhysicochemical,
                'T' : TPhysicochemical
            },
            
            'classical dock(Force Field)' : {
                'B' : classical_dock1B,
                'T' : classical_dock1T
            },
        
            'classical dock(Force Field) With Adjuvant' : {
                'B' : classical_dock1BAdjuvant,
                'T' : classical_dock1TAdjuvant
            },

            'Machine Learning based dock' : {
                'B' : dock1B,
                'T' : dock1T
            },
        
            'Machine Learning based dock With Adjuvant' : {
                'B' : dock1BAdjuvant,
                'T' : dock1TAdjuvant
            },
            
        }

        # file_path = f'{self.target_path}/{generate_filename_with_timestamp_and_random()}_eval_res.json'

        sliced_pred = {key: pred[key] for key in list(pred.keys())[:9]}
        # Extract data for B and T cells
        B_data = {key: sliced_pred[key]['B'] for key in sliced_pred.keys() if key != 'seq'}
        T_data = {key: sliced_pred[key]['T'] for key in sliced_pred.keys() if key != 'seq'}

        # Create DataFrames
        B_result = pd.DataFrame(B_data)
        T_result = pd.DataFrame(T_data)

        print("DataFrames exported to B_result.xlsx and T_result.xlsx")

        file_path = f'{self.target_path}/{generate_filename_with_timestamp_and_random()}_eval_res_quantum.json'

        # Export to Excel files
        B_result.to_csv(f"{self.target_path}/B_result.csv", index=False)
        T_result.to_csv(f"{self.target_path}/T_result.csv", index=False)

        try:
            url = self.llm_url
            B_res_review = requests.post(url, files = {"uploaded_file": open(f'{self.target_path}/B_result.csv', 'rb')}, verify=False, timeout=6000)#.content.decode('utf8').replace("'", '"')
            T_res_review = requests.post(url, files = {"uploaded_file": open(f'{self.target_path}/T_result.csv', 'rb')}, verify=False, timeout=6000)#.content.decode('utf8').replace("'", '"')
        # print(B_res_review)
        # print(T_res_review)

        # B_res_review = json.loads(B_res_review)
        # T_res_review = json.loads(T_res_review)
        
            export_string_to_text_file(B_res_review.text, f'{self.target_path}/B_res_review.txt')
            export_string_to_text_file(T_res_review.text, f'{self.target_path}/T_res_review.txt')
        except:
            pass

        try:
            alphafold_res_dir = f'{self.target_path}/Alphafold Modelling Result'

            create_folder(alphafold_res_dir)
            create_folder(f'{alphafold_res_dir}/B')
            create_folder(f'{alphafold_res_dir}/T')

            url = self.alphafold_url
            if url[-1] == '/':
                pass
            else:
                url += '/'

            for epitope_type in list(['B', 'T']):
                for i, seq in enumerate(pred['seq'][epitope_type]):
                    # response = requests.post(url, data = {"protein_sequence": seq, "jobname":f"{epitope_type}_{i}_3D_{seq}"}, verify=False, timeout=6000)
                    response = requests.get(url+ "?protein_sequence="+seq+"&jobname="+f"{epitope_type}_{i}_3D_{seq}", verify=False, timeout=6000)
                    if response.status_code == 200:
                        response_res = json.loads(response.text)
                        print(response_res)
                        try:
                            download_file(url + response_res['result'],f"{alphafold_res_dir}/{epitope_type}/{epitope_type}_{i}_3D_{seq}.zip")
                        except:
                            print("Error/gagal download")
                    else:
                        print("Failed Protein Modelling")
                        continue

        except:
            pass

        # Export dictionary ke file JSON
        with open(file_path, 'w') as json_file:
            json.dump(str(pred), json_file)


        return pred
    
    def predict(self):
        print("Starting Predict Epitope...")
        pred1, pred2 = self.predict_epitope()
        print("Starting Predict Evalution For Epitope...")
        pred_eval = self.predict_eval(pred2['B'], pred2['T'])
        print("Finished Predict")
        return pred1, pred2, pred_eval

class QReVa:
    def preprocessing_begin(seq):
        seq = str(seq).upper()
        delete_char = "BJOUXZ\n\t 1234567890*&^%$#@!~()[];:',.<><?/"
        for i in range(len(delete_char)):
            seq = seq.replace(delete_char[i],'')
        return seq


    def __init__(self, sequence, base_path, target_path, n_receptor, n_adjuvant, blast_activate=False, qibm_api="", backend_type="ibmq_qasm_simulator", llm_url="", alphafold_url=""):
        self.sequence = QReVa.preprocessing_begin(sequence)
        self.base_path = base_path
        self.blast_activate = blast_activate
        self.target_path = target_path
        self.n_receptor = n_receptor
        self.n_adjuvant = n_adjuvant
        self.qibm_api = qibm_api
        self.backend_type = backend_type
        self.llm_url = llm_url
        self.alphafold_url = alphafold_url
        self.sampler = None
        create_folder(self.target_path)
        # self.estimator = None

        # try:
        #     self.service = QiskitRuntimeService(token=self.qibm_api, channel="ibm_quantum")
        #     self.backend = self.service.backend(self.backend_type)
        #     self.estimator, self.sampler = Estimator(session=Session(service=self.service, backend=self.backend)), Sampler(session=Session(service=self.service, backend=self.backend))
        # except:
        #     self.backend = None
        #     self.sampler = None


        self.alphabet = 'ACDEFGHIKLMNPQRSTVWY'  # Hanya huruf-huruf asam amino yang relevan
        self.num_features = len(self.alphabet)

        try:
            model_path = 'B_vqc_model'
            self.loaded_Bmodel = QReVa.quantum_load_model(self.base_path, model_path, self)
        except Exception as e:
            print(f"Error on Load Model Epitope B : {e}")
            # QMessageBox.warning(self, "Error", f"Error on Load Model Epitope B")
        
        try:
            model_path = 'T_vqc_model'
            self.loaded_Tmodel = QReVa.quantum_load_model(self.base_path, model_path, self)
        except:
            print("Error on load model Epitope T")
            # QMessageBox.warning(self, "Error", f"Error on load model Epitope T")

        try:
            with open(os.path.join(label_dir, 'allergenicity_label_mapping_quantum.json'), 'r') as f:
                label_dict = json.load(f)
        except:
            print("Error on load allergenicity label")
            # QMessageBox.warning(self, "Error", f"Error on load allergenicity label")

        self.reverse_label_mapping_allergen = {v: k for k, v in label_dict.items()}
        self.seq_length_allergen = 4857

        try:
            with open(os.path.join(label_dir,'toxin_label_mapping_quantum.json'), 'r') as f:
                label_dict = json.load(f)
        except:
            print("Error on load toxin label")
            # QMessageBox.warning(self, "Error", f"Error on load toxin label")

        self.reverse_label_mapping_toxin = {v: k for k, v in label_dict.items()}
        self.seq_length_toxin = 35

        try:
            with open(os.path.join(label_dir, 'antigenicity_label_mapping_quantum.json'), 'r') as f:
                label_dict = json.load(f)
        except:
            print("Error on load antigenicity label")
            # QMessageBox.warning(self, "Error", f"Error on load antigenicity label")

        self.reverse_label_mapping_antigen = {v: k for k, v in label_dict.items()}
        self.seq_length_antigen = 83

        try:
            with open(os.path.join(label_dir,'BPepTree_label_quantum.json'), 'r') as f:
                label_dict = json.load(f)
            self.Blabel = QReVa.invert_dict(label_dict)
        except:
            print("Error on load Epitope B label")
            # QMessageBox.warning(self, "Error", f"Error on load Epitope B label")

        try:
            with open(os.path.join(label_dir,'TPepTree_label_quantum.json'), 'r') as f:
                label_dict = json.load(f)
            self.Tlabel = QReVa.invert_dict(label_dict)
        except:
            print("Error on load Epitope T label")
            # QMessageBox.warning(self, "Error", f"Error on load Epitope T label")

    def quantum_load_model(base_path, model_path, self):
        # print(os.path.join(qmodel_dir, model_path))
        # with open(os.path.join(qmodel_dir, model_path), 'rb') as f:
        # if (self.sampler != None) and (self.backend != None) and (self.backend != 'ibmq_qasm_simulator'):
        #     model = VQC.load(os.path.join(qmodel_dir, model_path))
        #     model.neural_network.sampler = self.sampler
            
        #     return model
        # else:
        model = VQC.load(os.path.join(qmodel_dir, model_path))
        
        return model

    def combine_lists(list1, list2):
        result = []
        current_group = ""

        for i in range(len(list1)):
            if list2[i] == 'E':
                current_group += list1[i]
            else:
                if current_group:
                    result.append(current_group)
                    current_group = ""
                result.append(list1[i])

        if current_group:
            result.append(current_group)

        return result
    
    def q_extraction_feature(seq):
        com = molecular_function.calculate_amino_acid_center_of_mass(str(seq))
        weight = molecular_function.calculate_molecular_weight(str(molecular_function.seq_to_smiles(seq)))

        return com, weight

    def janin_hydrophobicity_scale(aa):
        scale = {
            'A': 0.42, 'C': 0.82, 'D': -1.23, 'E': -2.02, 'F': 1.37,
            'G': 0.58, 'H': -0.73, 'I': 1.38, 'K': -1.05, 'L': 1.06,
            'M': 0.64, 'N': -0.6, 'P': 0.12, 'Q': -0.22, 'R': -0.84,
            'S': -0.04, 'T': 0.26, 'V': 1.08, 'W': 1.78, 'Y': 0.79
        }
        return scale.get(aa, 0.0)


    def get_position(aa):
        pos = [i+1 for i in range(0, len(aa))]
        return pos

    def one_hot_encoding(self, sequence):
        encoding = []
        for char in sequence:
            vector = [0] * self.num_features
            if char in self.alphabet:
                index = self.alphabet.index(char)
                vector[index] = 1
            encoding.append(vector)
        return encoding

    def extraction_feature(aa):
        pos = QReVa.get_position(aa)
        scale = [QReVa.janin_hydrophobicity_scale(aa[i]) for i in range(len(aa))]

        res = [[pos[i], scale[i], len(aa)] for i in range(len(pos))]

        return res

    def predict_label_and_probability_allergenicity(self, sequence):
        try:
            model_path = 'allergen_vqc_model'
            model = QReVa.quantum_load_model(model_dir, model_path, self)
        except:
            print("Error on load model allergenicity")
            # QMessageBox.warning(self, "Error", f"Error on load model allergenicity")

        # try:
        #     sequence = sequence[:self.seq_length_allergen]
        #     sequence = [QReVa.one_hot_encoding(self, seq) for seq in [sequence]]
        #     sequence = [seq + [[0] * self.num_features] * (self.seq_length_allergen - len(seq)) for seq in sequence]
        #     sequence = np.array(sequence)
        # except:
        #     print("Error")
        #     # # QMessageBox.warning(self, "Error", f"Gagal Preprocess sebelum prediksi Allergenisitas")
        
        try:
            feature = [QReVa.q_extraction_feature(sequence)]
            prediction = int(model.predict(feature))
            print(f"Prediction of allergen : {prediction}")
            prediction = self.reverse_label_mapping_allergen[prediction]

            return prediction
        except:
            print("Error predict allergenicity")
            # # QMessageBox.warning(self, "Error", f"Gagal Prediksi Allergenisitas")

    def predict_label_and_probability_toxin(self, sequence):
        try:
            model_path = 'allergen_vqc_model'
            model = QReVa.quantum_load_model(model_dir, model_path, self)
        except:
            print("Error on load model toxin")
            # QMessageBox.warning(self, "Error", f"Error on load model toxin")

        try:
            feature = [QReVa.q_extraction_feature(sequence)]
            prediction = int(model.predict(feature))
            prediction = self.reverse_label_mapping_toxin[prediction]

            return prediction
        except:
            print("Error toxin predict")
            # # QMessageBox.warning(self, "Error", f"Gagal Prediksi Allergenisitas")
    
    def predict_label_and_probability_antigenicity(self, sequence):
        try:
            model_path = 'antigen_vqc_model'
            model = QReVa.quantum_load_model(model_dir, model_path, self)
        except:
            print("Error on load model antigenicity")
            # QMessageBox.warning(self, "Error", f"Error on load model antigenicity")

        try:
            feature = [QReVa.q_extraction_feature(sequence)]
            prediction = int(model.predict(feature))
            prediction = self.reverse_label_mapping_antigen[prediction]

            return prediction
        except:
            print("Error antigenicity predict")
            # # QMessageBox.warning(self, "Error", f"Gagal Prediksi Allergenisitas")

    def invert_dict(dictionary):
        inverted_dict = {value: key for key, value in dictionary.items()}
        return inverted_dict

    def process_epitope(input_list):
        output_list = []
        current_group = []

        for item in input_list:
            if item == 'E':
                current_group.append(item)
            else:
                if current_group:
                    output_list.append(''.join(current_group))
                    current_group = []
                output_list.append(item)

        if current_group:
            output_list.append(''.join(current_group))

        return output_list

    def filter_epitope(data):
        filtered_seq = []
        filtered_label = []

        for i in range(len(data['seq'])):
            if data['label'][i] != '.':
                filtered_seq.append(data['seq'][i])
                filtered_label.append(data['label'][i])

        filtered_data = {'seq': filtered_seq, 'label': filtered_label}
        return filtered_data
    
    def string_to_list(input_string):
        return list(input_string)
    
    def calculate_hydrophobicity(sequence):
        hydrophobic_residues = ['A', 'I', 'L', 'M', 'F', 'V', 'W', 'Y']
        hydrophilic_residues = ['R', 'N', 'C', 'Q', 'E', 'G', 'H', 'K', 'S', 'T', 'D']
        hydrophobicity_scores = {
            'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
            'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
            'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
            'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
        }
        
        hydrophobicity = 0
        for residue in sequence:
            if residue in hydrophobic_residues:
                hydrophobicity += hydrophobicity_scores[residue]
            elif residue in hydrophilic_residues:
                hydrophobicity -= hydrophobicity_scores[residue]
            else:
                hydrophobicity -= 0.5  # penalty for unknown residues
        
        return hydrophobicity / len(sequence)

    def antigenicity(sequence, window_size=7):
        antigenicity_scores = []
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            antigenicity_score = sum([1 if window[j] == 'A' or window[j] == 'G' else 0 for j in range(window_size)])
            antigenicity_scores.append(antigenicity_score)
        return antigenicity_scores
    
    def emini_surface_accessibility(sequence, window_size=9):
        surface_accessibility_scores = []
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            surface_accessibility_score = sum([1 if window[j] in ['S', 'T', 'N', 'Q'] else 0 for j in range(window_size)])
            surface_accessibility_scores.append(surface_accessibility_score)
        return surface_accessibility_scores
    
    def perform_blastp(query_sequence, self):
        # return "N/A"
        # Lakukan BLASTp
        if self.blast_activate == True:
            start = time.time()
            try:
                result_handle = NCBIWWW.qblast("blastp", "nr", query_sequence)
            except Exception as e:
                # return str(e)
                print("BLASTp failed to connect")
                return "Skip because any error"

            # Analisis hasil BLASTp
            print("BLASTp Starting..")
            blast_records = NCBIXML.parse(result_handle)
            for blast_record in blast_records:
                for alignment in blast_record.alignments:
                    # Anda dapat menyesuaikan kriteria kesamaan sesuai kebutuhan
                    # Misalnya, jika Anda ingin mengembalikan hasil jika similarity > 90%
                    for hsp in alignment.hsps:
                        similarity = (hsp.positives / hsp.align_length) * 100
                        if similarity > 80:
                            return similarity
            print("BLASTp Finisihing..")
            end = time.time()
            time_blast = end-start
            print(f"Time for BLASTp : {time_blast} s")
            # Jika tidak ada protein yang cocok ditemukan
            return "Non-similarity"
        else:
            return "Not Activated"
        

    def predict_epitope(self):
        seq = self.sequence
        seq_extra = QReVa.extraction_feature(seq)
        # print(seq_extra)
        # try:
        #     print([int(self.loaded_Bmodel.predict([seq_extra[i]])) for i in range(len(seq_extra))])
        # except  Exception as e:
        #     print(e)
        # print(self.loaded_Bmodel.predict([seq_extra[0]]))
        print("pass test")
        # try:
        pred_res_B = [self.Blabel[int(self.loaded_Bmodel.predict([seq_extra[i]]))] for i in range(len(seq_extra))]
        print("Prediction B epitope pass")
        pred_res_T = [self.Tlabel[int(self.loaded_Tmodel.predict([seq_extra[i]]))] for i in range(len(seq_extra))]
        print("Prediction T epitope pass")

        seq_B = QReVa.combine_lists(seq, pred_res_B)
        pred_B = QReVa.process_epitope(pred_res_B)
        seq_T = QReVa.combine_lists(seq, pred_res_T)
        pred_T = QReVa.process_epitope(pred_res_T)
        

        pred_res1 = {
            'B': {'amino acid': QReVa.string_to_list(seq), 'predictions': pred_res_B},
            'T': {'amino acid': QReVa.string_to_list(seq), 'predictions': pred_res_T}
        }

        pred_res2 = {
            'B': {'seq': seq_B, 'label': pred_B},
            'T': {'seq': seq_T, 'label': pred_T}
        }

        return pred_res1, pred_res2

            # pred_proba_B = [np.max(self.loaded_Bmodel.predict_proba([seq_extra[i]])[0]) for i in range(len(seq_extra))]
            # pred_proba_T = [np.max(self.loaded_Tmodel.predict_proba([seq_extra[i]])[0]) for i in range(len(seq_extra))]
        # except:
        #     print("Error on epitope predict")
            # # QMessageBox.warning(self, "Error", f"Error on prediksi epitope")

    def predict_eval(self, Bpred, Tpred):
        BCell = QReVa.filter_epitope(Bpred)['seq']
        TCell = QReVa.filter_epitope(Tpred)['seq']
        
        Ballergen = []
        for i in range(len(BCell)):
            baller = QReVa.predict_label_and_probability_allergenicity(self, BCell[i])
            Ballergen.append(baller)

        Tallergen = []
        for i in range(len(TCell)):
            baller = QReVa.predict_label_and_probability_allergenicity(self, TCell[i])
            Tallergen.append(baller)

        Btoxin = []
        Ttoxin = []

        for i in range(len(BCell)):
            baller = QReVa.predict_label_and_probability_toxin(self, BCell[i])
            Btoxin.append(baller)

        for i in range(len(TCell)):
            baller = QReVa.predict_label_and_probability_toxin(self, TCell[i])
            Ttoxin.append(baller)

        BAntigen = []
        TAntigen = []

        for i in range(len(BCell)):
            baller = QReVa.predict_label_and_probability_antigenicity(self, BCell[i])
            BAntigen.append(baller)

        for i in range(len(TCell)):
            baller = QReVa.predict_label_and_probability_antigenicity(self, TCell[i])
            TAntigen.append(baller)

        Bhydrophobicity = []
        Bkolaskar = []
        Btangonkar = []
        Bemini = []
        Bsimilar = []
        BPhysicochemical = []

        for i in range(len(BCell)):
            Bhydrophobicity.append(QReVa.calculate_hydrophobicity(BCell[i]))
            Bkolaskar.append(QReVa.antigenicity(BCell[i]))
            Btangonkar.append(QReVa.antigenicity(BCell[i], window_size=5))
            Bemini.append(QReVa.emini_surface_accessibility(BCell[i]))
            Bsimilar.append(QReVa.perform_blastp(BCell[i], self))
            BPhysicochemical.append(ProtParamClone(BCell[i]).calculate())

        Thydrophobicity = []
        Tkolaskar = []
        Ttangonkar = []
        Temini = []
        Tsimilar = []
        TPhysicochemical = []

        for i in range(len(TCell)):
            Thydrophobicity.append(QReVa.calculate_hydrophobicity(TCell[i]))
            Tkolaskar.append(QReVa.antigenicity(TCell[i]))
            Ttangonkar.append(QReVa.antigenicity(TCell[i], window_size=5))
            Temini.append(QReVa.emini_surface_accessibility(TCell[i]))
            Tsimilar.append(QReVa.perform_blastp(TCell[i], self))
            TPhysicochemical.append(ProtParamClone(TCell[i]).calculate())
        
        # print(BCell)
        classical_dock1B, classical_dock1T = ClassicalDocking(BCell, TCell, self.base_path, self.target_path, self.n_receptor).ForceField1()
        # print(classical_dock1B, classical_dock1T)
        classical_dock1BAdjuvant, classical_dock1TAdjuvant = ClassicalDockingWithAdjuvant(BCell, TCell, self.base_path, self.target_path, self.n_receptor, self.n_adjuvant).ForceField1()

        dock1B, dock1T = QMLDocking(BCell, TCell, self.base_path, self.target_path, self.n_receptor, self.sampler).MLDock1()
        dock1BAdjuvant, dock1TAdjuvant = QMLDockingWithAdjuvant(BCell, TCell, self.base_path, self.target_path, self.n_receptor, self.n_adjuvant, self.sampler).MLDock1()
        # print(dock1B, dock1T)
        pred = {
            'seq': {
                'B':BCell,
                'T':TCell
            },

            'allergenicity' : {
                'B' : Ballergen,
                'T' : Tallergen
            },

            'toxin' : {
                'B' : Btoxin,
                'T' : Ttoxin
            },

            'antigenicity' : {
                'B' : BAntigen,
                'T' : TAntigen
            },

            'hydrophobicity' : {
                'B' : Bhydrophobicity,
                'T' : Thydrophobicity
            },
            
            'kolaskar' : {
                'B' : Bkolaskar,
                'T' : Tkolaskar
            },

            'tangonkar' : {
                'B' : Btangonkar,
                'T' : Ttangonkar
            },

            'emini' : {
                'B' : Bemini,
                'T' : Temini
            },

            'similarity' : {
                'B' : Bsimilar,
                'T' : Tsimilar
            },

            'physicochemical' : {
                'B' : BPhysicochemical,
                'T' : TPhysicochemical
            },
            
            'classical dock(Force Field)' : {
                'B' : classical_dock1B,
                'T' : classical_dock1T
            },
        
            'classical dock(Force Field) With Adjuvant' : {
                'B' : classical_dock1BAdjuvant,
                'T' : classical_dock1TAdjuvant
            },

            'Machine Learning based dock' : {
                'B' : dock1B,
                'T' : dock1T
            },
        
            'Machine Learning based dock With Adjuvant' : {
                'B' : dock1BAdjuvant,
                'T' : dock1TAdjuvant
            },
            
        }

        sliced_pred = {key: pred[key] for key in list(pred.keys())[:9]}
        # Extract data for B and T cells
        B_data = {key: sliced_pred[key]['B'] for key in sliced_pred.keys() if key != 'seq'}
        T_data = {key: sliced_pred[key]['T'] for key in sliced_pred.keys() if key != 'seq'}

        # Create DataFrames
        B_result = pd.DataFrame(B_data)
        T_result = pd.DataFrame(T_data)

        print("DataFrames exported to B_result.xlsx and T_result.xlsx")

        file_path = f'{self.target_path}/{generate_filename_with_timestamp_and_random()}_eval_res_quantum.json'

        # Export to Excel files
        B_result.to_csv(f"{self.target_path}/B_result.csv", index=False)
        T_result.to_csv(f"{self.target_path}/T_result.csv", index=False)

        try:
            url = self.llm_url
            B_res_review = requests.post(url, files = {"uploaded_file": open(f'{self.target_path}/B_result.csv', 'rb')}, verify=False, timeout=6000)#.content.decode('utf8').replace("'", '"')
            T_res_review = requests.post(url, files = {"uploaded_file": open(f'{self.target_path}/T_result.csv', 'rb')}, verify=False, timeout=6000)#.content.decode('utf8').replace("'", '"')
        # print(B_res_review)
        # print(T_res_review)

        # B_res_review = json.loads(B_res_review)
        # T_res_review = json.loads(T_res_review)
        
            export_string_to_text_file(B_res_review.text, f'{self.target_path}/B_res_review.txt')
            export_string_to_text_file(T_res_review.text, f'{self.target_path}/T_res_review.txt')
        except:
            pass

        try:
            alphafold_res_dir = f'{self.target_path}/Alphafold Modelling Result'

            create_folder(alphafold_res_dir)
            create_folder(f'{alphafold_res_dir}/B')
            create_folder(f'{alphafold_res_dir}/T')

            url = self.alphafold_url
            if url[-1] == '/':
                pass
            else:
                url += '/'

            for epitope_type in list(['B', 'T']):
                for i, seq in enumerate(pred['seq'][epitope_type]):
                    # response = requests.post(url, data = {"protein_sequence": seq, "jobname":f"{epitope_type}_{i}_3D_{seq}"}, verify=False, timeout=6000)
                    response = requests.get(url+ "?protein_sequence="+seq+"&jobname="+f"{epitope_type}_{i}_3D_{seq}", verify=False, timeout=6000)
                    if response.status_code == 200:
                        response_res = json.loads(response.text)
                        print(response_res)
                        try:
                            download_file(url + response_res['result'],f"{alphafold_res_dir}/{epitope_type}/{epitope_type}_{i}_3D_{seq}.zip")
                        except:
                            print("Error/gagal download")
                    else:
                        print("Failed Protein Modelling")
                        continue

        except:
            pass
        
        # try:
        #     create_folder(f'{self.target_path}/Alphafold Modelling Result')

        # except:
        #     pass

        # Export dictionary ke file JSON
        with open(file_path, 'w') as json_file:
            json.dump(str(pred), json_file)


        return pred
    
    def predict(self):
        print("Starting Predict Epitope..")
        pred1, pred2 = self.predict_epitope()
        print("Starting Predict Evalution For Epitope..")
        pred_eval = self.predict_eval(pred2['B'], pred2['T'])
        print("Finished Predict")
        return pred1, pred2, pred_eval
        
class ProtParamClone:

    def preprocessing_begin(seq):
        seq = str(seq).upper()
        delete_char = "BJOUXZ\n\t 1234567890*&^%$#@!~()[];:',.<><?/"
        for i in range(len(delete_char)):
            seq = seq.replace(delete_char[i],'')
        return seq
    
    def __init__(self, seq):
        self.seq = ProtParamClone.preprocessing_begin(seq)
        # Dictionary untuk nilai hydropathicity dari asam amino
        self.hydropathy_values = {
            'A': 1.800,
            'R': -4.500,
            'N': -3.500,
            'D': -3.500,
            'C': 2.500,
            'E': -3.500,
            'Q': -3.500,
            'G': -0.400,
            'H': -3.200,
            'I': 4.500,
            'L': 3.800,
            'K': -3.900,
            'M': 1.900,
            'F': 2.800,
            'P': -1.600,
            'S': -0.800,
            'T': -0.700,
            'W': -0.900,
            'Y': -1.300,
            'V': 4.200
        }

        # Dictionary untuk koefisien pereduksian ekstinsi
        self.extinction_coefficients = {
            'Tyr': 1490,
            'Trp': 5500,
            'Cystine': 125
        }

        # Dictionary untuk setengah masa hidup protein
        self.half_life_values = {
            'A': {'Mammalian': 4.4, 'Yeast': 20, 'E. coli': 10},
            'R': {'Mammalian': 1, 'Yeast':2, 'E. coli':2},
            'N': {'Mammalian': 1.4, 'Yeast':3, 'E. coli': 10},
            'D': {'Mammalian': 1.1, 'Yeast':3, 'E. coli': 10},
            'C': {'Mammalian': 1.2, 'Yeast': 20,'E. coli': 10},
            'E': {'Mammalian': 0.8, 'Yeast':10, 'E. coli': 10},
            'Q': {'Mammalian': 1, 'Yeast':30, 'E. coli': 10},
            'G': {'Mammalian': 30, 'Yeast': 20, 'E. coli': 10},
            'H': {'Mammalian': 3.5, 'Yeast':10, 'E. coli': 10},
            'I': {'Mammalian': 20, 'Yeast':30, 'E. coli': 10},
            'L': {'Mammalian': 5.5, 'Yeast':3, 'E. coli':2},
            'K': {'Mammalian': 1.3, 'Yeast':3, 'E. coli':2},
            'M': {'Mammalian': 30, 'Yeast': 20,'E. coli': 10},
            'F': {'Mammalian': 1.1, 'Yeast':3, 'E. coli':2},
            'P': {'Mammalian': 20, 'Yeast': 20,'E. coli':0},
            'S': {'Mammalian': 1.9, 'Yeast': 20,'E. coli': 10},
            'T': {'Mammalian': 7.2, 'Yeast': 20,'E. coli': 10},
            'W': {'Mammalian': 2.8, 'Yeast':3, 'E. coli':2},
            'Y': {'Mammalian': 2.8, 'Yeast':10, 'E. coli':2},
            'V': {'Mammalian': 100, 'Yeast': 20, 'E. coli': 10}
        }

    # Fungsi untuk menghitung Indeks Instabilitas
    def calculate_instability_index(sequence, self):
        X = ProteinAnalysis(sequence)
        return X.instability_index()
        # L = len(sequence)
        # instability_sum = 0.0
        # for i in range(L - 1):
        #     dipeptide = sequence[i:i+2]
        #     instability_sum += self.hydropathy_values.get(dipeptide, 0)
        # instability_index = (10 / L) * instability_sum
        # return instability_index


    # Fungsi untuk menghitung Indeks Alifatik
    def calculate_aliphatic_index(sequence):
        X_Ala = (sequence.count('A') / len(sequence)) * 100
        X_Val = (sequence.count('V') / len(sequence)) * 100
        X_Ile = (sequence.count('I') / len(sequence)) * 100
        X_Leu = (sequence.count('L') / len(sequence)) * 100
        aliphatic_index = X_Ala + 2.9 * X_Val + 3.9 * (X_Ile + X_Leu)
        return aliphatic_index

    # Fungsi untuk menghitung GRAVY
    def calculate_gravy(sequence, self):
        gravy = sum(self.hydropathy_values.get(aa, 0) for aa in sequence) / len(sequence)
        return gravy

    # Fungsi untuk menghitung koefisien pereduksian ekstinsi
    def calculate_extinction_coefficient(sequence, self):
        num_Tyr = sequence.count('Y')
        num_Trp = sequence.count('W')
        num_Cystine = sequence.count('C')
        extinction_prot = (num_Tyr * self.extinction_coefficients['Tyr'] +
                        num_Trp * self.extinction_coefficients['Trp'] +
                        num_Cystine * self.extinction_coefficients['Cystine'])
        return extinction_prot

    # Fungsi untuk memprediksi setengah masa hidup protein
    def predict_half_life(self, sequence, organism='Mammalian'):
        n_terminal_residue = sequence[0]
        half_life = self.half_life_values.get(n_terminal_residue, {}).get(organism, 'N/A')
        return half_life

    # Fungsi untuk menghitung komposisi atom
    def calculate_atom_composition(molecule):
        atom_composition = {}
        atoms = molecule.GetAtoms()
        for atom in atoms:
            atom_symbol = atom.GetSymbol()
            if atom_symbol in atom_composition:
                atom_composition[atom_symbol] += 1
            else:
                atom_composition[atom_symbol] = 1
        return atom_composition

    # Fungsi untuk menghitung komposisi atom H, N, O, dan S
    def calculate_HNOSt_composition(molecule):
        composition = ProtParamClone.calculate_atom_composition(molecule)
        C_count = composition.get('C', 0)
        H_count = composition.get('H', 0)
        N_count = composition.get('N', 0)
        O_count = composition.get('O', 0)
        S_count = composition.get('S', 0)
        return C_count, H_count, N_count, O_count, S_count

    # Fungsi untuk menghitung Theoretical pI
    def calculate_theoretical_pI(protein_sequence):
        X = ProteinAnalysis(protein_sequence)
        pI = X.isoelectric_point()
        # pI = IP(protein_sequence).pi()
        return pI
        
    def MolWeight(self):
        mol = Chem.MolFromSequence(self.seq)
        mol_weight = Descriptors.MolWt(mol)
        return mol_weight

    def calculate(self):
        try:
            try:
                instability = ProtParamClone.calculate_instability_index(self.seq, self)
            except:
                print("error instability")
            aliphatic = ProtParamClone.calculate_aliphatic_index(self.seq)
            try:
                calculate_gravy = ProtParamClone.calculate_gravy(self.seq, self)
            except:
                print("error gravy")
            extinction = ProtParamClone.calculate_extinction_coefficient(self.seq, self)
            try:
                half_life = ProtParamClone.predict_half_life(self, sequence=self.seq)
            except:
                print("error half life")
            mol = Chem.MolFromSequence(self.seq)
            try:
                formula = rdMolDescriptors.CalcMolFormula(mol)
            except:
                print("error formula")
            Cn, Hn, Nn, On, Sn = ProtParamClone.calculate_HNOSt_composition(mol)
            try:
                theoretical_pI = IP.IsoelectricPoint(self.seq).pi()
            except:
                print("erro theoretical_pI")
            m_weight = ProtParamClone.MolWeight(self)

            res = {
                'seq' : self.seq,
                'instability' : instability,
                'aliphatic' : aliphatic,
                'gravy' : calculate_gravy,
                'extinction' : extinction,
                'half_life' : half_life,
                'formula' : formula,
                'C' : Cn,
                'H' : Hn,
                'N' : Nn,
                'O' : On,
                'S' : Sn,
                'theoretical_pI' : theoretical_pI,
                'mol weight' : m_weight
            }

            return res
        except:
            print("Error calculate physicochemical")

class ClassicalDocking:
    def __init__(self, bseq, tseq, base_path, target_path, n_receptor):
        self.bseq = bseq
        self.tseq = tseq
        self.base_path = base_path
        self.target_path = target_path
        self.n_receptor = n_receptor

    def ForceField1(self):
        b_cell_receptor = pd.read_csv(os.path.join(data_dir, 'b cell receptor homo sapiens.csv'))
        b_cell_receptor = b_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        b_cell_receptor = b_cell_receptor[0:self.n_receptor]
        b_epitope = self.bseq
        # print(b_epitope)

        seq1 = []
        seq2 = []
        com1 = []
        com2 = []
        seq2id = []
        Attractive = []
        Repulsive = []
        VDW_lj_force = []
        coulomb_energy = []
        force_field = []
        for b in b_epitope:
            for i in range(len(b_cell_receptor)):
                seq1.append(b)
                seq2.append(b_cell_receptor['seq'][i])
                seq2id.append(b_cell_receptor['id'][i])
                aa1 = molecular_function.calculate_amino_acid_center_of_mass(b)
                com1.append(aa1)
                aa2 = molecular_function.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                com2.append(aa2)
                dist = molecular_function.calculate_distance_between_amino_acids(aa1, aa2)
                attract = ms.attractive_energy(dist)
                Attractive.append(attract)
                repulsive = ms.repulsive_energy(dist)
                Repulsive.append(repulsive)
                vdw = ms.lj_force(dist)
                VDW_lj_force.append(vdw)
                ce = ms.coulomb_energy(e, e, dist)
                coulomb_energy.append(ce)
                force_field.append(vdw+ce)
        
        b_res = {
            'Ligand' : seq1,
            'Receptor' : seq2,
            'Receptor id' : seq2id,
            'Center Of Ligand Mass' : com1,
            'Center Of Receptor Mass' : com2,
            'Attractive' : Attractive,
            'Repulsive' : Repulsive,
            'VDW LJ Force' : VDW_lj_force,
            'Coulomb Energy' : coulomb_energy,
            'Force Field' : force_field
        }
        b_res_df = pd.DataFrame(b_res)
        print("Force Field B Cell Success")


        t_cell_receptor = pd.read_csv(os.path.join(data_dir, 't cell receptor homo sapiens.csv'))
        t_cell_receptor = t_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        t_cell_receptor = t_cell_receptor[0:self.n_receptor]
        t_epitope = self.tseq

        seq1 = []
        seq2 = []
        seq2id = []
        Attractive = []
        Repulsive = []
        VDW_lj_force = []
        coulomb_energy = []
        force_field = []
        com1 = []
        com2 = []
        for  t in  t_epitope:
            for i in range(len( t_cell_receptor)):
                seq1.append(t)
                seq2.append(t_cell_receptor['seq'][i])
                seq2id.append( t_cell_receptor['id'][i])
                aa1 = molecular_function.calculate_amino_acid_center_of_mass( t)
                com1.append(aa1)
                aa2 = molecular_function.calculate_amino_acid_center_of_mass( t_cell_receptor['seq'][i])
                com2.append(aa2)
                dist = molecular_function.calculate_distance_between_amino_acids(aa1, aa2)
                attract = ms.attractive_energy(dist)
                Attractive.append(attract)
                repulsive = ms.repulsive_energy(dist)
                Repulsive.append(repulsive)
                vdw = ms.lj_force(dist)
                VDW_lj_force.append(vdw)
                ce = ms.coulomb_energy(e, e, dist)
                coulomb_energy.append(ce)
                force_field.append(vdw+ce)

        t_res = {
            'Ligand' : seq1,
            'Receptor' : seq2,
            'Receptor id' : seq2id,
            'Center Of Ligand Mass' : com1,
            'Center Of Receptor Mass' : com2,
            'Attractive' : Attractive,
            'Repulsive' : Repulsive,
            'VDW LJ Force' : VDW_lj_force,
            'Coulomb Energy' : coulomb_energy,
            'Force Field' : force_field
        }
        t_res_df = pd.DataFrame(t_res)
        print("Force Field T Cell Success")

        b_res_df.to_excel(self.target_path+'/'+'forcefield_b_cell.xlsx', index=False)
        t_res_df.to_excel(self.target_path+'/'+'forcefield_t_cell.xlsx', index=False)

        return b_res, t_res
    
class ClassicalDockingWithAdjuvant:
    def __init__(self, bseq, tseq, base_path, target_path, n_receptor, n_adjuvant):
        self.bseq = bseq
        self.tseq = tseq
        self.base_path = base_path
        self.target_path = target_path
        self.n_receptor = n_receptor
        self.n_adjuvant = n_adjuvant

    def ForceField1(self):
        adjuvant = pd.read_csv(os.path.join(data_dir, 'PubChem_compound_text_adjuvant.csv'))
        adjuvant = adjuvant.sample(frac=1, random_state=42).reset_index(drop=True)
        adjuvant = adjuvant[0:self.n_adjuvant]
        b_cell_receptor = pd.read_csv(os.path.join(data_dir, 'b cell receptor homo sapiens.csv'))
        b_cell_receptor = b_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        b_cell_receptor = b_cell_receptor[0:self.n_receptor]
        b_epitope = self.bseq

        seq1 = []
        seq2 = []
        seq2id = []
        Attractive = []
        Repulsive = []
        VDW_lj_force = []
        coulomb_energy = []
        force_field = []
        adjuvant_list = []
        adjuvant_isosmiles = []
        com1 = []
        com2 = []
        for b in b_epitope:
            for i in range(len(b_cell_receptor)):
                for adju in range(len(adjuvant)):
                    adjuvant_list.append(adjuvant['cid'][adju])
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    adjuvant_isosmiles.append(adjuvant['isosmiles'][adju])
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    seq_plus_adjuvant = Chem.MolToSmiles(molecular_function.combine_epitope_with_adjuvant(b, adjuvant['isosmiles'][adju]))
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    seq1.append(seq_plus_adjuvant)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    seq2.append(b_cell_receptor['seq'][i])
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    seq2id.append(b_cell_receptor['id'][i])
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    aa1 = molecular_function.calculate_amino_acid_center_of_mass_smiles(seq_plus_adjuvant)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    aa2 = molecular_function.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    com1.append(aa1)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    com2.append(aa2)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    dist = molecular_function.calculate_distance_between_amino_acids(aa1, aa2)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    attract = ms.attractive_energy(dist)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    Attractive.append(attract)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    repulsive = ms.repulsive_energy(dist)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    Repulsive.append(repulsive)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    vdw = ms.lj_force(dist)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    VDW_lj_force.append(vdw)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    ce = ms.coulomb_energy(e, e, dist)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    coulomb_energy.append(ce)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    force_field.append(vdw+ce)
                    # print(f"testing_docking_with_adjuvant_{b}_{i}_{adju}")
                    print("===========================================\n\n")

        b_res = {
            'Ligand' : seq1,
            'Receptor' : seq2,
            'Receptor id' : seq2id,
            'Adjuvant CID' : adjuvant_list,
            'Adjuvant IsoSMILES' : adjuvant_isosmiles,
            'Attractive' : Attractive,
            'Repulsive' : Repulsive,
            'Center Of Ligand Mass' : com1,
            'Center Of Receptor Mass' : com2,
            'VDW LJ Force' : VDW_lj_force,
            'Coulomb Energy' : coulomb_energy,
            'Force Field' : force_field
        }
        b_res_df = pd.DataFrame(b_res)
        print("Force Field B Cell Success With Adjuvant")


        t_cell_receptor = pd.read_csv(os.path.join(data_dir, 't cell receptor homo sapiens.csv'))
        t_cell_receptor = t_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        t_cell_receptor = t_cell_receptor[0:self.n_receptor]
        t_epitope = self.tseq

        seq1 = []
        seq2 = []
        seq2id = []
        Attractive = []
        Repulsive = []
        VDW_lj_force = []
        coulomb_energy = []
        force_field = []
        adjuvant_list = []
        adjuvant_isosmiles = []
        com1 = []
        com2 = []
        for b in t_epitope:
            for i in range(len(t_cell_receptor)):
                for adju in range(len(adjuvant)):
                    adjuvant_list.append(adjuvant['cid'][adju])
                    adjuvant_isosmiles.append(adjuvant['isosmiles'][adju])
                    seq_plus_adjuvant = Chem.MolToSmiles(molecular_function.combine_epitope_with_adjuvant(b, adjuvant['isosmiles'][adju]))
                    seq1.append(seq_plus_adjuvant)
                    seq2.append(b_cell_receptor['seq'][i])
                    seq2id.append(b_cell_receptor['id'][i])
                    aa1 = molecular_function.calculate_amino_acid_center_of_mass_smiles(seq_plus_adjuvant)
                    aa2 = molecular_function.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                    com1.append(aa1)
                    com2.append(aa2)
                    dist = molecular_function.calculate_distance_between_amino_acids(aa1, aa2)
                    attract = ms.attractive_energy(dist)
                    Attractive.append(attract)
                    repulsive = ms.repulsive_energy(dist)
                    Repulsive.append(repulsive)
                    vdw = ms.lj_force(dist)
                    VDW_lj_force.append(vdw)
                    ce = ms.coulomb_energy(e, e, dist)
                    coulomb_energy.append(ce)
                    force_field.append(vdw+ce)

        t_res = {
            'Ligand' : seq1,
            'Receptor' : seq2,
            'Receptor id' : seq2id,
            'Adjuvant CID' : adjuvant_list,
            'Adjuvant IsoSMILES' : adjuvant_isosmiles,
            'Attractive' : Attractive,
            'Repulsive' : Repulsive,
            'Center Of Ligand Mass' : com1,
            'Center Of Receptor Mass' : com2,
            'VDW LJ Force' : VDW_lj_force,
            'Coulomb Energy' : coulomb_energy,
            'Force Field' : force_field
        }
        t_res_df = pd.DataFrame(t_res)
        print("Force Field T Cell Success With Adjuvant")

        b_res_df.to_excel(self.target_path+'/'+'forcefield_b_cell_with_adjuvant.xlsx', index=False)
        t_res_df.to_excel(self.target_path+'/'+'forcefield_t_cell_with_adjuvant.xlsx', index=False)

        return b_res, t_res
        
class MLDocking:
    def __init__(self, bseq, tseq, base_path, target_path, n_receptor):
        self.bseq = bseq
        self.tseq = tseq
        self.base_path = base_path
        self.target_path = target_path
        self.n_receptor = n_receptor

    def MLDock1(self):
        b_cell_receptor = pd.read_csv(os.path.join(data_dir, 'b_receptor_v2.csv'))
        b_cell_receptor = b_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        b_cell_receptor = b_cell_receptor[0:self.n_receptor]
        b_epitope = self.bseq

        seq1 = []
        seq2 = []
        seq2id = []
        smilesseq1 = []
        smilesseq2 = []
        molwseq1 = []
        molwseq2 = []
        bdist = []
        bmol_pred = []
        com1 = []
        com2 = []
        
        for b in b_epitope:
            for i in range(len(b_cell_receptor)):
                seq1.append(b)
                seq2.append(b_cell_receptor['seq'][i])
                seq2id.append(b_cell_receptor['id'][i])
                aa1 = molecular_function.calculate_amino_acid_center_of_mass(b)
                smiles1 = molecular_function.seq_to_smiles(b)
                smilesseq1.append(smiles1)
                molwt1 = molecular_function.calculate_molecular_weight(smiles1)
                molwseq1.append(molwt1)
                aa2 = molecular_function.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                smiles2 = molecular_function.seq_to_smiles(b_cell_receptor['seq'][i])
                smilesseq2.append(smiles2)
                molwt2 = molecular_function.calculate_molecular_weight(smiles2)
                molwseq2.append(molwt2)
                dist = molecular_function.calculate_distance_between_amino_acids(aa1, aa2)
                bdist.append(dist)
                com1.append(aa1)
                com2.append(aa2)
                feature = [aa1, molwt1, aa2, molwt2, dist]

                bmol_pred.append(ml_dock(feature))
        
        b_res = {
            'Ligand' : seq1,
            'Receptor' : seq2,
            'Receptor id' : seq2id,
            'Ligand Smiles' : smilesseq1,
            'Receptor Smiles' : smilesseq2,
            'Center Of Mass Ligand' : com1,
            'Center Of Mass Receptor' : com2,
            'Molecular Weight Of Ligand' : molwseq1,
            'Molecular Weight Of Receptor' : molwseq2,
            'Distance' : bdist,
            'Docking(Ki (nM))' : bmol_pred
        }

        b_res_df = pd.DataFrame(b_res)
        print("Machine Learning Based Scoring Function of B Cell Success")


        t_cell_receptor = pd.read_csv(os.path.join(data_dir, 't_receptor_v2.csv'))
        t_cell_receptor = t_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        t_cell_receptor = t_cell_receptor[0:self.n_receptor]
        t_epitope = self.tseq

        seq1 = []
        seq2 = []
        seq2id = []
        smilesseq1 = []
        smilesseq2 = []
        molwseq1 = []
        molwseq2 = []
        bdist = []
        bmol_pred = []
        com1 = []
        com2 = []
        
        for b in t_epitope:
            for i in range(len(t_cell_receptor)):
                seq1.append(b)
                seq2.append(t_cell_receptor['seq'][i])
                seq2id.append(t_cell_receptor['id'][i])
                aa1 = molecular_function.calculate_amino_acid_center_of_mass(b)
                smiles1 = molecular_function.seq_to_smiles(b)
                smilesseq1.append(smiles1)
                molwt1 = molecular_function.calculate_molecular_weight(smiles1)
                molwseq1.append(molwt1)
                aa2 = molecular_function.calculate_amino_acid_center_of_mass(t_cell_receptor['seq'][i])
                smiles2 = molecular_function.seq_to_smiles(t_cell_receptor['seq'][i])
                smilesseq2.append(smiles2)
                molwt2 = molecular_function.calculate_molecular_weight(smiles2)
                molwseq2.append(molwt2)
                dist = molecular_function.calculate_distance_between_amino_acids(aa1, aa2)
                bdist.append(dist)
                com1.append(aa1)
                com2.append(aa2)
                feature = [aa1, molwt1, aa2, molwt2, dist]

                bmol_pred.append(ml_dock(feature))
        
        t_res = {
            'Ligand' : seq1,
            'Receptor' : seq2,
            'Receptor id' : seq2id,
            'Ligand Smiles' : smilesseq1,
            'Receptor Smiles' : smilesseq2,
            'Center Of Mass Ligand' : com1,
            'Center Of Mass Receptor' : com2,
            'Molecular Weight Of Ligand' : molwseq1,
            'Molecular Weight Of Receptor' : molwseq2,
            'Distance' : bdist,
            'Docking(Ki (nM))' : bmol_pred
        }

        t_res_df = pd.DataFrame(t_res)
        print("Machine Learning Based Scoring Function of T Cell Success")

        b_res_df.to_excel(self.target_path+'/'+'ml_scoring_func_b_cell.xlsx', index=False)
        t_res_df.to_excel(self.target_path+'/'+'ml_scoring_func_t_cell.xlsx', index=False)

        return b_res, t_res
    
class MLDockingWithAdjuvant:
    def __init__(self, bseq, tseq, base_path, target_path, n_receptor, n_adjuvant):
        self.bseq = bseq
        self.tseq = tseq
        self.base_path = base_path
        self.target_path = target_path
        self.n_receptor = n_receptor
        self.n_adjuvant = n_adjuvant

    def MLDock1(self):
        adjuvant = pd.read_csv(os.path.join(data_dir, 'PubChem_compound_text_adjuvant.csv'))
        adjuvant = adjuvant.sample(frac=1, random_state=42).reset_index(drop=True)
        adjuvant = adjuvant[0:self.n_adjuvant]
        b_cell_receptor = pd.read_csv(os.path.join(data_dir, 'b_receptor_v2.csv'))
        b_cell_receptor = b_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        b_cell_receptor = b_cell_receptor[0:self.n_receptor]
        b_epitope = self.bseq

        seq1 = []
        seq2 = []
        seq2id = []
        adjuvant_list = []
        adjuvant_isosmiles = []
        seq2id = []
        smilesseq1 = []
        smilesseq2 = []
        molwseq1 = []
        molwseq2 = []
        bdist = []
        bmol_pred = []
        com1 = []
        com2 = []
        for b in b_epitope:
            for i in range(len(b_cell_receptor)):
                for adju in range(len(adjuvant)):
                    adjuvant_list.append(adjuvant['cid'][adju])
                    adjuvant_isosmiles.append(adjuvant['isosmiles'][adju])
                    seq_plus_adjuvant = Chem.MolToSmiles(molecular_function.combine_epitope_with_adjuvant(b, adjuvant['isosmiles'][adju]))
                    seq1.append(b)
                    seq2.append(b_cell_receptor['seq'][i])
                    seq2id.append(b_cell_receptor['id'][i])
                    aa1 = molecular_function.calculate_amino_acid_center_of_mass_smiles(seq_plus_adjuvant)
                    smiles1 = seq_plus_adjuvant
                    smilesseq1.append(smiles1)
                    molwt1 = molecular_function.calculate_molecular_weight(smiles1)
                    molwseq1.append(molwt1)
                    aa2 = molecular_function.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                    smiles2 = molecular_function.seq_to_smiles(b_cell_receptor['seq'][i])
                    smilesseq2.append(smiles2)
                    molwt2 = molecular_function.calculate_molecular_weight(smiles2)
                    molwseq2.append(molwt2)
                    dist = molecular_function.calculate_distance_between_amino_acids(aa1, aa2)
                    bdist.append(dist)
                    feature = [aa1, molwt1, aa2, molwt2, dist]
                    com1.append(aa1)
                    com2.append(aa2)

                    bmol_pred.append(ml_dock(feature))

        b_res = {
            'Ligand' : seq1,
            'Receptor' : seq2,
            'Receptor id' : seq2id,
            'Adjuvant CID' : adjuvant_list,
            'Adjuvant IsoSMILES' : adjuvant_isosmiles,
            'Ligand Smiles' : smilesseq1,
            'Receptor Smiles' : smilesseq2,
            'Center Of Mass Ligand' : com1,
            'Center Of Mass Receptor' : com2,
            'Molecular Weight Of Ligand' : molwseq1,
            'Molecular Weight Of Receptor' : molwseq2,
            'Distance' : bdist,
            'Docking(Ki (nM))' : bmol_pred
        }

        b_res_df = pd.DataFrame(b_res)
        print("Machine Learning Based Scoring Function of B Cell Success With Adjuvant")


        t_cell_receptor = pd.read_csv(os.path.join(data_dir, 't cell receptor homo sapiens.csv'))
        t_cell_receptor = t_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        t_cell_receptor = t_cell_receptor[0:self.n_receptor]
        t_epitope = self.tseq

        seq1 = []
        seq2 = []
        seq2id = []
        adjuvant_list = []
        adjuvant_isosmiles = []
        seq2id = []
        smilesseq1 = []
        smilesseq2 = []
        molwseq1 = []
        molwseq2 = []
        bdist = []
        bmol_pred = []
        com1 = []
        com2 = []
        for b in t_epitope:
            for i in range(len(t_cell_receptor)):
                for adju in range(len(adjuvant)):
                    adjuvant_list.append(adjuvant['cid'][adju])
                    adjuvant_isosmiles.append(adjuvant['isosmiles'][adju])
                    seq_plus_adjuvant = Chem.MolToSmiles(molecular_function.combine_epitope_with_adjuvant(b, adjuvant['isosmiles'][adju]))
                    seq1.append(b)
                    seq2.append(t_cell_receptor['seq'][i])
                    seq2id.append(t_cell_receptor['id'][i])
                    aa1 = molecular_function.calculate_amino_acid_center_of_mass_smiles(seq_plus_adjuvant)
                    smiles1 = seq_plus_adjuvant
                    smilesseq1.append(smiles1)
                    molwt1 = molecular_function.calculate_molecular_weight(smiles1)
                    molwseq1.append(molwt1)
                    aa2 = molecular_function.calculate_amino_acid_center_of_mass(t_cell_receptor['seq'][i])
                    smiles2 = molecular_function.seq_to_smiles(t_cell_receptor['seq'][i])
                    smilesseq2.append(smiles2)
                    molwt2 = molecular_function.calculate_molecular_weight(smiles2)
                    molwseq2.append(molwt2)
                    dist = molecular_function.calculate_distance_between_amino_acids(aa1, aa2)
                    bdist.append(dist)
                    feature = [aa1, molwt1, aa2, molwt2, dist]
                    com1.append(aa1)
                    com2.append(aa2)

                    bmol_pred.append(ml_dock(feature))
                    

        t_res = {
            'Ligand' : seq1,
            'Receptor' : seq2,
            'Receptor id' : seq2id,
            'Adjuvant CID' : adjuvant_list,
            'Adjuvant IsoSMILES' : adjuvant_isosmiles,
            'Ligand Smiles' : smilesseq1,
            'Receptor Smiles' : smilesseq2,
            'Center Of Mass Ligand' : com1,
            'Center Of Mass Receptor' : com2,
            'Molecular Weight Of Ligand' : molwseq1,
            'Molecular Weight Of Receptor' : molwseq2,
            'Distance' : bdist,
            'Docking(Ki (nM))' : bmol_pred
        }
        t_res_df = pd.DataFrame(t_res)
        print("Machine Learning Based Scoring Function of T Cell Success With Adjuvant")

        b_res_df.to_excel(self.target_path+'/'+'ml_b_cell_with_adjuvant.xlsx', index=False)
        t_res_df.to_excel(self.target_path+'/'+'ml_t_cell_with_adjuvant.xlsx', index=False)

        return b_res, t_res
   
class QMLDocking:
    def __init__(self, bseq, tseq, base_path, target_path, n_receptor, sampler=None):
        self.bseq = bseq
        self.tseq = tseq
        self.base_path = base_path
        self.target_path = target_path
        self.n_receptor = n_receptor
        self.sampler = sampler

    def MLDock1(self):
        b_cell_receptor = pd.read_csv(os.path.join(data_dir, 'b_receptor_v2.csv'))
        b_cell_receptor = b_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        b_cell_receptor = b_cell_receptor[0:self.n_receptor]
        b_epitope = self.bseq

        seq1 = []
        seq2 = []
        seq2id = []
        smilesseq1 = []
        smilesseq2 = []
        molwseq1 = []
        molwseq2 = []
        bdist = []
        bmol_pred = []
        com1 = []
        com2 = []
        
        for b in b_epitope:
            for i in range(len(b_cell_receptor)):
                seq1.append(b)
                seq2.append(b_cell_receptor['seq'][i])
                seq2id.append(b_cell_receptor['id'][i])
                aa1 = molecular_function.calculate_amino_acid_center_of_mass(b)
                smiles1 = molecular_function.seq_to_smiles(b)
                smilesseq1.append(smiles1)
                molwt1 = molecular_function.calculate_molecular_weight(smiles1)
                molwseq1.append(molwt1)
                aa2 = molecular_function.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                smiles2 = molecular_function.seq_to_smiles(b_cell_receptor['seq'][i])
                smilesseq2.append(smiles2)
                molwt2 = molecular_function.calculate_molecular_weight(smiles2)
                molwseq2.append(molwt2)
                dist = molecular_function.calculate_distance_between_amino_acids(aa1, aa2)
                bdist.append(dist)
                com1.append(aa1)
                com2.append(aa2)
                feature = [aa1, molwt1, aa2, molwt2, dist]

                bmol_pred.append(qml_dock(feature, self.sampler))
        
        b_res = {
            'Ligand' : seq1,
            'Receptor' : seq2,
            'Receptor id' : seq2id,
            'Ligand Smiles' : smilesseq1,
            'Receptor Smiles' : smilesseq2,
            'Center Of Mass Ligand' : com1,
            'Center Of Mass Receptor' : com2,
            'Molecular Weight Of Ligand' : molwseq1,
            'Molecular Weight Of Receptor' : molwseq2,
            'Distance' : bdist,
            'Docking(Ki (nM))' : bmol_pred
        }

        b_res_df = pd.DataFrame(b_res)
        print("Quantum Machine Learning Based Scoring Function of B Cell Success")


        t_cell_receptor = pd.read_csv(os.path.join(data_dir, 't_receptor_v2.csv'))
        t_cell_receptor = t_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        t_cell_receptor = t_cell_receptor[0:self.n_receptor]
        t_epitope = self.tseq

        seq1 = []
        seq2 = []
        seq2id = []
        smilesseq1 = []
        smilesseq2 = []
        molwseq1 = []
        molwseq2 = []
        bdist = []
        bmol_pred = []
        com1 = []
        com2 = []
        
        for b in t_epitope:
            for i in range(len(t_cell_receptor)):
                seq1.append(b)
                seq2.append(t_cell_receptor['seq'][i])
                seq2id.append(t_cell_receptor['id'][i])
                aa1 = molecular_function.calculate_amino_acid_center_of_mass(b)
                smiles1 = molecular_function.seq_to_smiles(b)
                smilesseq1.append(smiles1)
                molwt1 = molecular_function.calculate_molecular_weight(smiles1)
                molwseq1.append(molwt1)
                aa2 = molecular_function.calculate_amino_acid_center_of_mass(t_cell_receptor['seq'][i])
                smiles2 = molecular_function.seq_to_smiles(t_cell_receptor['seq'][i])
                smilesseq2.append(smiles2)
                molwt2 = molecular_function.calculate_molecular_weight(smiles2)
                molwseq2.append(molwt2)
                dist = molecular_function.calculate_distance_between_amino_acids(aa1, aa2)
                bdist.append(dist)
                com1.append(aa1)
                com2.append(aa2)
                feature = [aa1, molwt1, aa2, molwt2, dist]

                bmol_pred.append(qml_dock(feature, self.sampler))
        
        t_res = {
            'Ligand' : seq1,
            'Receptor' : seq2,
            'Receptor id' : seq2id,
            'Ligand Smiles' : smilesseq1,
            'Receptor Smiles' : smilesseq2,
            'Center Of Mass Ligand' : com1,
            'Center Of Mass Receptor' : com2,
            'Molecular Weight Of Ligand' : molwseq1,
            'Molecular Weight Of Receptor' : molwseq2,
            'Distance' : bdist,
            'Docking(Ki (nM))' : bmol_pred
        }

        t_res_df = pd.DataFrame(t_res)
        print("Machine Learning Based Scoring Function of T Cell Success")

        b_res_df.to_excel(self.target_path+'/'+'qml_scoring_func_b_cell.xlsx', index=False)
        t_res_df.to_excel(self.target_path+'/'+'qml_scoring_func_t_cell.xlsx', index=False)

        return b_res, t_res
    
class QMLDockingWithAdjuvant:
    def __init__(self, bseq, tseq, base_path, target_path, n_receptor, n_adjuvant, sampler=None):
        self.bseq = bseq
        self.tseq = tseq
        self.base_path = base_path
        self.target_path = target_path
        self.n_receptor = n_receptor
        self.n_adjuvant = n_adjuvant
        self.sampler = sampler

    def MLDock1(self):
        adjuvant = pd.read_csv(os.path.join(data_dir, 'PubChem_compound_text_adjuvant.csv'))
        adjuvant = adjuvant.sample(frac=1, random_state=42).reset_index(drop=True)
        adjuvant = adjuvant[0:self.n_adjuvant]
        b_cell_receptor = pd.read_csv(os.path.join(data_dir, 'b_receptor_v2.csv'))
        b_cell_receptor = b_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        b_cell_receptor = b_cell_receptor[0:self.n_receptor]
        b_epitope = self.bseq

        seq1 = []
        seq2 = []
        seq2id = []
        adjuvant_list = []
        adjuvant_isosmiles = []
        seq2id = []
        smilesseq1 = []
        smilesseq2 = []
        molwseq1 = []
        molwseq2 = []
        bdist = []
        bmol_pred = []
        com1 = []
        com2 = []
        for b in b_epitope:
            for i in range(len(b_cell_receptor)):
                for adju in range(len(adjuvant)):
                    adjuvant_list.append(adjuvant['cid'][adju])
                    adjuvant_isosmiles.append(adjuvant['isosmiles'][adju])
                    seq_plus_adjuvant = Chem.MolToSmiles(molecular_function.combine_epitope_with_adjuvant(b, adjuvant['isosmiles'][adju]))
                    seq1.append(b)
                    seq2.append(b_cell_receptor['seq'][i])
                    seq2id.append(b_cell_receptor['id'][i])
                    aa1 = molecular_function.calculate_amino_acid_center_of_mass_smiles(seq_plus_adjuvant)
                    smiles1 = seq_plus_adjuvant
                    smilesseq1.append(smiles1)
                    molwt1 = molecular_function.calculate_molecular_weight(smiles1)
                    molwseq1.append(molwt1)
                    aa2 = molecular_function.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                    smiles2 = molecular_function.seq_to_smiles(b_cell_receptor['seq'][i])
                    smilesseq2.append(smiles2)
                    molwt2 = molecular_function.calculate_molecular_weight(smiles2)
                    molwseq2.append(molwt2)
                    dist = molecular_function.calculate_distance_between_amino_acids(aa1, aa2)
                    bdist.append(dist)
                    feature = [aa1, molwt1, aa2, molwt2, dist]
                    com1.append(aa1)
                    com2.append(aa2)

                    bmol_pred.append(qml_dock(feature,self.sampler))

        b_res = {
            'Ligand' : seq1,
            'Receptor' : seq2,
            'Receptor id' : seq2id,
            'Adjuvant CID' : adjuvant_list,
            'Adjuvant IsoSMILES' : adjuvant_isosmiles,
            'Ligand Smiles' : smilesseq1,
            'Receptor Smiles' : smilesseq2,
            'Center Of Mass Ligand' : com1,
            'Center Of Mass Receptor' : com2,
            'Molecular Weight Of Ligand' : molwseq1,
            'Molecular Weight Of Receptor' : molwseq2,
            'Distance' : bdist,
            'Docking(Ki (nM))' : bmol_pred
        }

        b_res_df = pd.DataFrame(b_res)
        print("Machine Learning Based Scoring Function of B Cell Success With Adjuvant")


        t_cell_receptor = pd.read_csv(os.path.join(data_dir, 't cell receptor homo sapiens.csv'))
        t_cell_receptor = t_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        t_cell_receptor = t_cell_receptor[0:self.n_receptor]
        t_epitope = self.tseq

        seq1 = []
        seq2 = []
        seq2id = []
        adjuvant_list = []
        adjuvant_isosmiles = []
        seq2id = []
        smilesseq1 = []
        smilesseq2 = []
        molwseq1 = []
        molwseq2 = []
        bdist = []
        bmol_pred = []
        com1 = []
        com2 = []
        for b in t_epitope:
            for i in range(len(t_cell_receptor)):
                for adju in range(len(adjuvant)):
                    adjuvant_list.append(adjuvant['cid'][adju])
                    adjuvant_isosmiles.append(adjuvant['isosmiles'][adju])
                    seq_plus_adjuvant = Chem.MolToSmiles(molecular_function.combine_epitope_with_adjuvant(b, adjuvant['isosmiles'][adju]))
                    seq1.append(b)
                    seq2.append(t_cell_receptor['seq'][i])
                    seq2id.append(t_cell_receptor['id'][i])
                    aa1 = molecular_function.calculate_amino_acid_center_of_mass_smiles(seq_plus_adjuvant)
                    smiles1 = seq_plus_adjuvant
                    smilesseq1.append(smiles1)
                    molwt1 = molecular_function.calculate_molecular_weight(smiles1)
                    molwseq1.append(molwt1)
                    aa2 = molecular_function.calculate_amino_acid_center_of_mass(t_cell_receptor['seq'][i])
                    smiles2 = molecular_function.seq_to_smiles(t_cell_receptor['seq'][i])
                    smilesseq2.append(smiles2)
                    molwt2 = molecular_function.calculate_molecular_weight(smiles2)
                    molwseq2.append(molwt2)
                    dist = molecular_function.calculate_distance_between_amino_acids(aa1, aa2)
                    bdist.append(dist)
                    feature = [aa1, molwt1, aa2, molwt2, dist]
                    com1.append(aa1)
                    com2.append(aa2)

                    bmol_pred.append(qml_dock(feature, self.sampler))
                    

        t_res = {
            'Ligand' : seq1,
            'Receptor' : seq2,
            'Receptor id' : seq2id,
            'Adjuvant CID' : adjuvant_list,
            'Adjuvant IsoSMILES' : adjuvant_isosmiles,
            'Ligand Smiles' : smilesseq1,
            'Receptor Smiles' : smilesseq2,
            'Center Of Mass Ligand' : com1,
            'Center Of Mass Receptor' : com2,
            'Molecular Weight Of Ligand' : molwseq1,
            'Molecular Weight Of Receptor' : molwseq2,
            'Distance' : bdist,
            'Docking(Ki (nM))' : bmol_pred
        }
        t_res_df = pd.DataFrame(t_res)
        print("Machine Learning Based Scoring Function of T Cell Success With Adjuvant")

        b_res_df.to_excel(self.target_path+'/'+'qml_b_cell_with_adjuvant.xlsx', index=False)
        t_res_df.to_excel(self.target_path+'/'+'qml_t_cell_with_adjuvant.xlsx', index=False)

        return b_res, t_res
   

class molecular_function:
    def calculate_amino_acid_center_of_mass(sequence):
        try:
            amino_acid_masses = []
            for aa in sequence:
                try:
                    amino_acid_masses.append(Chem.Descriptors.MolWt(Chem.MolFromSequence(aa)))
                except:
                    return 0
                    break

            # Hitung pusat massa asam amino
            total_mass = sum(amino_acid_masses)
            center_of_mass = sum(i * mass for i, mass in enumerate(amino_acid_masses, start=1)) / total_mass

            return center_of_mass
        except:
            return 0

    def calculate_amino_acid_center_of_mass_smiles(sequence):
        try:
            amino_acid_masses = []
            for aa in sequence:
                amino_acid_masses.append(Chem.Descriptors.MolWt(Chem.MolFromSmiles(aa)))

            # Hitung pusat massa asam amino
            total_mass = sum(amino_acid_masses)
            center_of_mass = sum(i * mass for i, mass in enumerate(amino_acid_masses, start=1)) / total_mass

            return center_of_mass
        except:
            return 0

    def calculate_distance_between_amino_acids(aa1, aa2):
        # Menghitung jarak antara dua pusat massa asam amino
        distance = abs(aa1 - aa2)
        return distance

    def generate_conformer(molecule_smiles):
        mol = Chem.MolFromSmiles(molecule_smiles)
        mol = Chem.AddHs(mol)  # Add hydrogens for a more accurate 3D structure
        conformer = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)  # Generate a conformer
        return mol

    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms

    def calculate_molecular_center_of_mass(smiles):
        molecule = molecular_function.generate_conformer(smiles)
        try:
            if molecule is None:
                return None

            # Hitung pusat massa molekul
            center_of_mass = rdMolTransforms.ComputeCentroid(molecule.GetConformer())
            total_mass = Descriptors.MolWt(molecule)
            # print(center_of_mass)
            # print(total_mass)
            center_of_mass = sum([center_of_mass[i] * total_mass for i in range(len(center_of_mass))]) / total_mass
            # print(total_mass)
            
            return center_of_mass
        except Exception as e:
            print("Error:", str(e))
            return None

    def seq_to_smiles(seq):
        try:
            mol = Chem.MolFromSequence(seq)
            smiles = Chem.MolToSmiles(mol,kekuleSmiles=True)
            return str(smiles)
        except:
            return None
        
    def combine_epitope_with_adjuvant(epitope_sequence, adjuvant_smiles):
        # Konversi epitop ke molekul RDKit
        # epitope_molecule = Chem.MolFromSequence(epitope_sequence)
        epitope_molecule = molecular_function.MolFromLongSequence(epitope_sequence)
        
        # Konversi SMILES adjuvant menjadi molekul RDKit
        # adjuvant_molecule = Chem.MolFromSmiles(adjuvant_smiles)
        adjuvant_molecule = molecular_function.MolFromLongSequence(adjuvant_smiles)
        
        # Gabungkan epitop dan adjuvant
        combined_molecule = Chem.CombineMols(adjuvant_molecule, epitope_molecule)
        
        return combined_molecule
    
    def calculate_molecular_weight(molecule_smiles):
        try:
            #mol = generate_conformer(molecule_smiles)
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol is None:
                print("Gagal membaca molekul.")
                return None

            # Menghitung massa molekul
            molecular_weight = Descriptors.MolWt(mol)

            return molecular_weight
        except:
            return None
        
    def calculate_amino_acid_center_of_mass(sequence):
        try:
            amino_acid_masses = []
            for aa in sequence:
                try:
                    amino_acid_masses.append(Chem.Descriptors.MolWt(molecular_function.MolFromLongSequence(aa)))
                except:
                    return 0
                    break

            # Hitung pusat massa asam amino
            total_mass = sum(amino_acid_masses)
            center_of_mass = sum(i * mass for i, mass in enumerate(amino_acid_masses, start=1)) / total_mass

            return center_of_mass
        except:
            return 0
        
    def MolFromLongSequence(sequence, chunk_size=100):
        """Convert a long sequence into a Mol object by splitting into chunks and combining."""
        # Split the sequence into chunks of specified size
        chunks = [sequence[i:i+chunk_size] for i in range(0, len(sequence), chunk_size)]
        
        # Convert each chunk to a Mol object using MolFromSmiles
        mols = [Chem.MolFromSequence(chunk) for chunk in chunks if chunk]  # Exclude empty chunks
        
        # Combine the Mols into a single Mol
        combined_mol = Chem.Mol()
        for mol in mols:
            if mol:
                combined_mol = Chem.CombineMols(combined_mol, mol)
        
        return combined_mol
    
    def calculate_amino_acid_center_of_mass(sequence):
        try:
            amino_acid_masses = []
            for aa in sequence:
                try:
                    amino_acid_masses.append(Chem.Descriptors.MolWt(molecular_function.MolFromLongSequence(aa)))
                except:
                    return 0
                    break

            # Hitung pusat massa asam amino
            total_mass = sum(amino_acid_masses)
            center_of_mass = sum(i * mass for i, mass in enumerate(amino_acid_masses, start=1)) / total_mass

            return center_of_mass
        except:
            return 0
        
    def calculate_distance_between_amino_acids(aa1, aa2):
        # Menghitung jarak antara dua pusat massa asam amino
        distance = abs(aa1 - aa2)
        return distance
    
    def seq_to_smiles(seq):
        try:
            mol = molecular_function.MolFromLongSequence(seq)
            smiles = Chem.MolToSmiles(mol,kekuleSmiles=True)
            return str(smiles)
        except:
            return None
        
    def calculate_molecular_center_of_mass(smiles):
        molecule = molecular_function.generate_conformer(smiles)
        try:
            if molecule is None:
                return None

            # Hitung pusat massa molekul
            center_of_mass = rdMolTransforms.ComputeCentroid(molecule.GetConformer())
            total_mass = Descriptors.MolWt(molecule)
            # print(center_of_mass)
            # print(total_mass)
            center_of_mass = sum([center_of_mass[i] * total_mass for i in range(len(center_of_mass))]) / total_mass
            # print(total_mass)
            
            return center_of_mass
        except Exception as e:
            print("Error:", str(e))
            return None

    def calculate_molecular_weight(molecule_smiles):
        try:
            #mol = generate_conformer(molecule_smiles)
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol is None:
                print("Gagal membaca molekul.")
                return None

            # Menghitung massa molekul
            molecular_weight = Descriptors.MolWt(mol)

            return molecular_weight
        except:
            return None

class ms:
    def __init__(self):
        self.mass_of_argon = 39.948

    @staticmethod
    def attractive_energy(r, epsilon=0.0103, sigma=3.4):
        """
        Attractive component of the Lennard-Jones 
        interactionenergy.
        
        Parameters
        ----------
        r: float
            Distance between two particles (Å)
        epsilon: float 
            Negative of the potential energy at the 
            equilibrium bond length (eV)
        sigma: float 
            Distance at which the potential energy is 
            zero (Å)
        
        Returns
        -------
        float
            Energy of attractive component of 
            Lennard-Jones interaction (eV)
        """
        if r == 0:
            return 0
        return -4.0 * epsilon * np.power(sigma / r, 6)

    @staticmethod
    def repulsive_energy(r, epsilon=0.0103, sigma=3.4):
        """
        Repulsive component of the Lennard-Jones 
        interactionenergy.
        
        Parameters
        ----------
        r: float
            Distance between two particles (Å)
        epsilon: float 
            Negative of the potential energy at the 
            equilibrium bond length (eV)
        sigma: float 
            Distance at which the potential energy is 
            zero (Å)
        
        Returns
        -------
        float
            Energy of repulsive component of 
            Lennard-Jones interaction (eV)
        """
        if r == 0:
            return 0
        return 4 * epsilon * np.power(sigma / r, 12)
    
    @staticmethod
    def lj_energy(r, epsilon=0.0103, sigma=3.4):
        """
        Implementation of the Lennard-Jones potential 
        to calculate the energy of the interaction.
        
        Parameters
        ----------
        r: float
            Distance between two particles (Å)
        epsilon: float 
            Negative of the potential energy at the 
            equilibrium bond length (eV)
        sigma: float 
            Distance at which the potential energy is 
            zero (Å)
        
        Returns
        -------
        float
            Energy of the Lennard-Jones potential 
            model (eV)
        """
        if r == 0:
            return 0
        
        return ms.repulsive_energy(
            r, epsilon, sigma) + ms.attractive_energy(
            r, epsilon, sigma)
    
    @staticmethod
    def coulomb_energy(qi, qj, r):
        """
        Calculation of Coulomb's law.
        
        Parameters
        ----------
        qi: float
            Electronic charge on particle i
        qj: float
            Electronic charge on particle j
        r: float 
            Distance between particles i and j (Å)
            
        Returns
        -------
        float
            Energy of the Coulombic interaction (eV)
        """
        if r == 0:
            return 0
        
        energy_joules = (qi * qj * e ** 2) / (
            4 * np.pi * epsilon_0 * r * 1e-10)
        return energy_joules / 1.602e-19
    
    @staticmethod
    def bonded(kb, b0, b):
        """
        Calculation of the potential energy of a bond.
        
        Parameters
        ----------
        kb: float
            Bond force constant (units: eV/Å^2)
        b0: float 
            Equilibrium bond length (units: Å)
        b: float
            Bond length (units: Å)
        
        Returns
        float
            Energy of the bonded interaction
        """
        
        return kb / 2 * (b - b0) ** 2
    
    @staticmethod
    def lj_force(r, epsilon=0.0103, sigma=3.4):
        """
        Implementation of the Lennard-Jones potential 
        to calculate the force of the interaction.
        
        Parameters
        ----------
        r: float
            Distance between two particles (Å)
        epsilon: float 
            Potential energy at the equilibrium bond 
            length (eV)
        sigma: float 
            Distance at which the potential energy is 
            zero (Å)
        
        Returns
        -------
        float
            Force of the van der Waals interaction (eV/Å)
        """
        if r != 0:
            return 48 * epsilon * np.power(
                sigma, 12) / np.power(
                r, 13) - 24 * epsilon * np.power(
                sigma, 6) / np.power(r, 7)
        else:
            return 0
    @staticmethod
    def init_velocity(T, number_of_particles):
        """
        Initialise the velocities for a series of 
        particles.
        
        Parameters
        ----------
        T: float
            Temperature of the system at 
            initialisation (K)
        number_of_particles: int
            Number of particles in the system
        
        Returns
        -------
        ndarray of floats
            Initial velocities for a series of 
            particles (eVs/Åamu)
        """
        R = np.random.rand(number_of_particles) - 0.5
        return R * np.sqrt(Boltzmann * T / (
            ms.mass_of_argon * 1.602e-19))
    @staticmethod
    def get_accelerations(positions):
        """
        Calculate the acceleration on each particle
        as a  result of each other particle. 
        N.B. We use the Python convention of 
        numbering from 0.
        
        Parameters
        ----------
        positions: ndarray of floats
            The positions, in a single dimension, 
            for all of the particles
            
        Returns
        -------
        ndarray of floats
            The acceleration on each
            particle (eV/Åamu)
        """
        accel_x = np.zeros((positions.size, positions.size))
        for i in range(0, positions.size - 1):
            for j in range(i + 1, positions.size):
                r_x = positions[j] - positions[i]
                rmag = np.sqrt(r_x * r_x)
                force_scalar = ms.lj_force(rmag, 0.0103, 3.4)
                force_x = force_scalar * r_x / rmag
                accel_x[i, j] = force_x / ms.mass_of_argon
                accel_x[j, i] = - force_x / ms.mass_of_argon
        return np.sum(accel_x, axis=0)
    @staticmethod
    def update_pos(x, v, a, dt):
        """
        Update the particle positions.
        
        Parameters
        ----------
        x: ndarray of floats
            The positions of the particles in a 
            single dimension
        v: ndarray of floats
            The velocities of the particles in a 
            single dimension
        a: ndarray of floats
            The accelerations of the particles in a 
            single dimension
        dt: float
            The timestep length
        
        Returns
        -------
        ndarray of floats:
            New positions of the particles in a single 
            dimension
        """
        return x + v * dt + 0.5 * a * dt * dt
    
    @staticmethod
    def update_velo(v, a, a1, dt):
        """
        Update the particle velocities.
        
        Parameters
        ----------
        v: ndarray of floats
            The velocities of the particles in a 
            single dimension (eVs/Åamu)
        a: ndarray of floats
            The accelerations of the particles in a 
            single dimension at the previous 
            timestep (eV/Åamu)
        a1: ndarray of floats
            The accelerations of the particles in a
            single dimension at the current 
            timestep (eV/Åamu)
        dt: float
            The timestep length
        
        Returns
        -------
        ndarray of floats:
            New velocities of the particles in a
            single dimension (eVs/Åamu)
        """
        return v + 0.5 * (a + a1) * dt
    @staticmethod
    def run_md(dt, number_of_steps, initial_temp, x):
        """
        Run a MD simulation.
        
        Parameters
        ----------
        dt: float
            The timestep length (s)
        number_of_steps: int
            Number of iterations in the simulation
        initial_temp: float
            Temperature of the system at 
            initialisation (K)
        x: ndarray of floats
            The initial positions of the particles in a 
            single dimension (Å)
            
        Returns
        -------
        ndarray of floats
            The positions for all of the particles 
            throughout the simulation (Å)
        """
        positions = np.zeros((number_of_steps, 3))
        v = ms.init_velocity(initial_temp, 3)
        a = ms.get_accelerations(x)
        for i in range(number_of_steps):
            x = ms.update_pos(x, v, a, dt)
            a1 = ms.get_accelerations(x)
            v = ms.update_velo(v, a, a1, dt)
            a = np.array(a1)
            positions[i, :] = x
        return positions
    

    @staticmethod
    def lj_force_cutoff(r, epsilon, sigma):
        """
        Implementation of the Lennard-Jones potential 
        to calculate the force of the interaction which 
        is considerate of the cut-off.
        
        Parameters
        ----------
        r: float
            Distance between two particles (Å)
        epsilon: float 
            Potential energy at the equilibrium bond 
            length (eV)
        sigma: float 
            Distance at which the potential energy is 
            zero (Å)
        
        Returns
        -------
        float
            Force of the van der Waals interaction (eV/Å)
        """
        cutoff = 15 
        if r < cutoff:
            return 48 * epsilon * np.power(
                sigma / r, 13) - 24 * epsilon * np.power(
                sigma / r, 7)
        else:
            return 0
        
def main():
    # Mendapatkan argumen dari terminal
    arguments = sys.argv[1:]

    # Inisialisasi variabel dengan nilai default
    sequence = ""
    n_receptor = 0
    n_adjuvant = 0
    blast_activate = False
    llm_url = ""
    alphafold_url = ""

    # Mengolah argumen
    for arg in arguments:
        key, value = arg.split("=")
        if key == "sequence":
            sequence = value
        elif key == "n_receptor":
            n_receptor = int(value)
        elif key == "n_adjuvant":
            n_adjuvant = int(value)
        elif key == "blast_activate":
            blast_activate = value.lower() == "true"
        elif key == "llm_url":
            llm_url = value
        elif key == "alphafold_url":
            alphafold_url = value
        else:
            print(f"Argumen '{key}' tidak dikenali.")

    # Membuat instance kelas ReVa dan menjalankan metode run()
    target_folder = os.path.join("result", "result_"+generate_filename_with_timestamp_and_random())
    create_folder(target_folder)
    reva = ReVa(sequence, get_base_path(),os.path.join(target_folder, generate_filename_with_timestamp_and_random()), n_receptor, n_adjuvant, blast_activate, llm_url, alphafold_url)
    qreva = QReVa(sequence, get_base_path(),os.path.join(target_folder, generate_filename_with_timestamp_and_random("quantum")), n_receptor, n_adjuvant, blast_activate=blast_activate ,qibm_api="", backend_type="ibmq_qasm_simulator",llm_url=llm_url, alphafold_url=alphafold_url)
    res1, res2, res3 = reva.predict()
    qres1, qres2, qres3 = qreva.predict()

if __name__ == "__main__":
    main()