"""
ReVa - Classical Vaccine Design Library

Main class for classical machine learning-based vaccine design.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from qiskit_machine_learning.algorithms import VQC

from .utils import (
    get_base_path, get_model_dir, get_label_dir, get_data_dir,
    create_folder, generate_filename_with_timestamp_and_random,
    preprocessing_begin, invert_dict, load_pickle_model,
    perform_blastp, export_string_to_text_file,
    run_llm_review
)
from .molecular_utils import MolecularFunction
from .protein_analysis import ProtParamClone, ProteinFeatures
from .docking import (
    ClassicalDocking, ClassicalDockingWithAdjuvant,
    MLDocking, MLDockingWithAdjuvant
)

class ReVa:
    """
    ReVa - Classical Vaccine Design Class
    
    A comprehensive class for vaccine design using classical machine learning approaches.
    """
    
    def __init__(self, sequence, base_path, target_path, n_receptor, n_adjuvant, 
                 blast_activate=False, llm_model_name="microsoft/biogpt", alphafold_url=""):
        """
        Initialize ReVa class
        
        Args:
            sequence (str): Protein sequence
            base_path (str): Base path for models and data
            target_path (str): Target path for results
            n_receptor (int): Number of receptors to use
            n_adjuvant (int): Number of adjuvants to use
            blast_activate (bool): Whether to activate BLAST search
            llm_model_name (str): Name of the local LLM model
            alphafold_url (str): URL for AlphaFold service
        """
        self.sequence = preprocessing_begin(sequence)
        self.base_path = base_path
        self.blast_activate = blast_activate
        self.target_path = target_path
        self.n_receptor = n_receptor
        self.n_adjuvant = n_adjuvant
        self.llm_model_name = llm_model_name
        self.alphafold_url = alphafold_url
        create_folder(self.target_path)

        self.alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        self.num_features = len(self.alphabet)

        # Load models
        self._load_models()
        
        # Load label mappings
        self._load_label_mappings()

    def _load_models(self):
        """Load trained models"""
        try:
            model_path = os.path.join(get_model_dir(), 'BPepTree.pkl')
            self.loaded_Bmodel = load_pickle_model(model_path)
        except Exception as e:
            print(f"Error loading B-cell epitope model: {e}")
            self.loaded_Bmodel = None
        
        try:
            model_path = os.path.join(get_model_dir(), 'TPepTree.pkl')
            self.loaded_Tmodel = load_pickle_model(model_path)
        except Exception as e:
            print(f"Error loading T-cell epitope model: {e}")
            self.loaded_Tmodel = None

    def _load_label_mappings(self):
        """Load label mappings for different properties"""
        # Allergenicity labels
        try:
            with open(os.path.join(get_label_dir(), 'allergenicity_label_mapping.json'), 'r') as f:
                label_dict = json.load(f)
            self.reverse_label_mapping_allergen = {v: k for k, v in label_dict.items()}
            self.seq_length_allergen = 4857
        except Exception as e:
            print(f"Error loading allergenicity labels: {e}")
            self.reverse_label_mapping_allergen = {}
            self.seq_length_allergen = 4857

        # Toxin labels
        try:
            with open(os.path.join(get_label_dir(), 'toxin_label_mapping.json'), 'r') as f:
                label_dict = json.load(f)
            self.reverse_label_mapping_toxin = {v: k for k, v in label_dict.items()}
            self.seq_length_toxin = 35
        except Exception as e:
            print(f"Error loading toxin labels: {e}")
            self.reverse_label_mapping_toxin = {}
            self.seq_length_toxin = 35

        # Antigenicity labels
        try:
            with open(os.path.join(get_label_dir(), 'antigenicity_label_mapping.json'), 'r') as f:
                label_dict = json.load(f)
            self.reverse_label_mapping_antigen = {v: k for k, v in label_dict.items()}
            self.seq_length_antigen = 83
        except Exception as e:
            print(f"Error loading antigenicity labels: {e}")
            self.reverse_label_mapping_antigen = {}
            self.seq_length_antigen = 83

        # B-cell epitope labels
        try:
            with open(os.path.join(get_label_dir(), 'BPepTree_label.json'), 'r') as f:
                label_dict = json.load(f)
            self.Blabel = invert_dict(label_dict)
        except Exception as e:
            print(f"Error loading B-cell epitope labels: {e}")
            self.Blabel = {}

        # T-cell epitope labels
        try:
            with open(os.path.join(get_label_dir(), 'TPepTree_label.json'), 'r') as f:
                label_dict = json.load(f)
            self.Tlabel = invert_dict(label_dict)
        except Exception as e:
            print(f"Error loading T-cell epitope labels: {e}")
            self.Tlabel = {}

    def one_hot_encoding(self, sequence):
        """Convert sequence to one-hot encoding"""
        encoding = []
        for char in sequence:
            vector = [0] * self.num_features
            if char in self.alphabet:
                index = self.alphabet.index(char)
                vector[index] = 1
            encoding.append(vector)
        return encoding

    def extraction_feature(self, aa):
        """Extract features from amino acid sequence"""
        pos = [i+1 for i in range(0, len(aa))]
        scale = [self.engelman_ges_scale(aa[i]) for i in range(len(aa))]
        res = [[pos[i], scale[i], len(aa)] for i in range(len(pos))]
        return res

    @staticmethod
    def engelman_ges_scale(aa):
        """Engelman-Ges scale for amino acids"""
        scale = {
            'A': 1.60, 'C': 2.00, 'D': -9.20, 'E': -8.20, 'F': 3.70,
            'G': 1.00, 'H': -3.00, 'I': 3.10, 'K': -8.80, 'L': 2.80,
            'M': 3.40, 'N': -4.80, 'P': -0.20, 'Q': -4.10, 'R': -12.3,
            'S': 0.60, 'T': 1.20, 'V': 2.60, 'W': 1.90, 'Y': -0.70
        }
        return scale.get(aa, 0.0)

    def predict_label_and_probability_allergenicity(self, sequence):
        """Predict allergenicity"""
        try:
            model_path = os.path.join(get_model_dir(), 'allerginicity.h5')
            model = load_model(model_path)
        except Exception as e:
            print(f"Error loading allergenicity model: {e}")
            return None, None

        try:
            sequence = sequence[:self.seq_length_allergen]
            sequence = [self.one_hot_encoding(seq) for seq in [sequence]]
            sequence = [seq + [[0] * self.num_features] * (self.seq_length_allergen - len(seq)) for seq in sequence]
            sequence = np.array(sequence)
        except Exception as e:
            print(f"Error preprocessing allergenicity sequence: {e}")
            return None, None
        
        try:
            prediction = model.predict(sequence)[0]
            predicted_label_index = 1 if prediction > 0.5 else 0
            predicted_label = self.reverse_label_mapping_allergen[predicted_label_index]
            return predicted_label, prediction
        except Exception as e:
            print(f"Error predicting allergenicity: {e}")
            return None, None

    def predict_label_and_probability_toxin(self, sequence):
        """Predict toxicity"""
        try:
            model_path = os.path.join(get_model_dir(), 'toxin.h5')
            model = load_model(model_path)
        except Exception as e:
            print(f"Error loading toxin model: {e}")
            return None, None

        try:
            sequence = sequence[:self.seq_length_toxin]
            sequence = [self.one_hot_encoding(seq) for seq in [sequence]]
            sequence = [seq + [[0] * self.num_features] * (self.seq_length_toxin - len(seq)) for seq in sequence]
            sequence = np.array(sequence)
        except Exception as e:
            print(f"Error preprocessing toxin sequence: {e}")
            return None, None
        
        try:
            prediction = model.predict(sequence)[0]
            predicted_label_index = 1 if prediction > 0.5 else 0
            predicted_label = self.reverse_label_mapping_toxin[predicted_label_index]
            return predicted_label, prediction
        except Exception as e:
            print(f"Error predicting toxicity: {e}")
            return None, None
    
    def predict_label_and_probability_antigenicity(self, sequence):
        """Predict antigenicity"""
        try:
            model_path = os.path.join(get_model_dir(), 'antigenicity.h5')
            model = load_model(model_path)
        except Exception as e:
            print(f"Error loading antigenicity model: {e}")
            return None, None

        try:
            sequence = sequence[:self.seq_length_antigen]
            sequence = [self.one_hot_encoding(seq) for seq in [sequence]]
            sequence = [seq + [[0] * self.num_features] * (self.seq_length_antigen - len(seq)) for seq in sequence]
            sequence = np.array(sequence)
        except Exception as e:
            print(f"Error preprocessing antigenicity sequence: {e}")
            return None, None

        try:
            prediction = model.predict(sequence)[0]
            predicted_label_index = 1 if prediction > 0.5 else 0
            predicted_label = self.reverse_label_mapping_antigen[predicted_label_index]
            return predicted_label, prediction
        except Exception as e:
            print(f"Error predicting antigenicity: {e}")
            return None, None

    @staticmethod
    def combine_lists(list1, list2):
        """Combine lists with epitope grouping"""
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

    @staticmethod
    def process_epitope(input_list):
        """Process epitope predictions"""
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

    @staticmethod
    def filter_epitope(data):
        """Filter epitope data"""
        filtered_seq = []
        filtered_label = []

        for i in range(len(data['seq'])):
            if data['label'][i] != '.':
                filtered_seq.append(data['seq'][i])
                filtered_label.append(data['label'][i])

        filtered_data = {'seq': filtered_seq, 'label': filtered_label}
        return filtered_data
    
    @staticmethod
    def string_to_list(input_string):
        """Convert string to list"""
        return list(input_string)

    def predict_epitope(self):
        """Predict B-cell and T-cell epitopes"""
        seq = self.sequence
        seq_extra = self.extraction_feature(seq)
        
        try:
            pred_res_B = [self.Blabel[self.loaded_Bmodel.predict([seq_extra[i]])[0]] for i in range(len(seq_extra))]
            pred_res_T = [self.Tlabel[self.loaded_Tmodel.predict([seq_extra[i]])[0]] for i in range(len(seq_extra))]

            pred_proba_B = [np.max(self.loaded_Bmodel.predict_proba([seq_extra[i]])[0]) for i in range(len(seq_extra))]
            pred_proba_T = [np.max(self.loaded_Tmodel.predict_proba([seq_extra[i]])[0]) for i in range(len(seq_extra))]
        except Exception as e:
            print(f"Error predicting epitopes: {e}")
            return None, None

        seq_B = self.combine_lists(seq, pred_res_B)
        pred_B = self.process_epitope(pred_res_B)
        seq_T = self.combine_lists(seq, pred_res_T)
        pred_T = self.process_epitope(pred_res_T)

        pred_res1 = {
            'B': {'amino acid': self.string_to_list(seq), 'predictions': pred_res_B, 'probabilities': pred_proba_B},
            'T': {'amino acid': self.string_to_list(seq), 'predictions': pred_res_T, 'probabilities': pred_proba_T}
        }

        pred_res2 = {
            'B': {'seq': seq_B, 'label': pred_B},
            'T': {'seq': seq_T, 'label': pred_T}
        }

        return pred_res1, pred_res2

    def predict_eval(self, Bpred, Tpred):
        """Evaluate epitope predictions"""
        BCell = self.filter_epitope(Bpred)['seq']
        TCell = self.filter_epitope(Tpred)['seq']
        
        # Allergenicity predictions
        Ballergen, BallergenProb = [], []
        for i in range(len(BCell)):
            baller, ballerprob = self.predict_label_and_probability_allergenicity(BCell[i])
            Ballergen.append(baller)
            BallergenProb.append(ballerprob[0] if ballerprob is not None else None)

        Tallergen, TallergenProb = [], []
        for i in range(len(TCell)):
            baller, ballerprob = self.predict_label_and_probability_allergenicity(TCell[i])
            Tallergen.append(baller)
            TallergenProb.append(ballerprob[0] if ballerprob is not None else None)

        # Toxin predictions
        Btoxin, BtoxinProb = [], []
        Ttoxin, TtoxinProb = [], []

        for i in range(len(BCell)):
            baller, ballerprob = self.predict_label_and_probability_toxin(BCell[i])
            Btoxin.append(baller)
            BtoxinProb.append(ballerprob[0] if ballerprob is not None else None)

        for i in range(len(TCell)):
            baller, ballerprob = self.predict_label_and_probability_toxin(TCell[i])
            Ttoxin.append(baller)
            TtoxinProb.append(ballerprob[0] if ballerprob is not None else None)

        # Antigenicity predictions
        BAntigen, BAntigenProb = [], []
        TAntigen, TAntigenProb = [], []

        for i in range(len(BCell)):
            baller, ballerprob = self.predict_label_and_probability_antigenicity(BCell[i])
            BAntigen.append(baller)
            BAntigenProb.append(ballerprob[0] if ballerprob is not None else None)

        for i in range(len(TCell)):
            baller, ballerprob = self.predict_label_and_probability_antigenicity(TCell[i])
            TAntigen.append(baller)
            TAntigenProb.append(ballerprob[0] if ballerprob is not None else None)

        # Physicochemical properties
        Bhydrophobicity, Bkolaskar, Btangonkar, Bemini, Bsimilar, BPhysicochemical = [], [], [], [], [], []

        for i in range(len(BCell)):
            Bhydrophobicity.append(ProteinFeatures.calculate_hydrophobicity(BCell[i]))
            Bkolaskar.append(ProteinFeatures.antigenicity(BCell[i]))
            Btangonkar.append(ProteinFeatures.antigenicity(BCell[i], window_size=5))
            Bemini.append(ProteinFeatures.emini_surface_accessibility(BCell[i]))
            Bsimilar.append(perform_blastp(BCell[i], self.blast_activate))
            BPhysicochemical.append(ProtParamClone(BCell[i]).calculate())

        Thydrophobicity, Tkolaskar, Ttangonkar, Temini, Tsimilar, TPhysicochemical = [], [], [], [], [], []

        for i in range(len(TCell)):
            Thydrophobicity.append(ProteinFeatures.calculate_hydrophobicity(TCell[i]))
            Tkolaskar.append(ProteinFeatures.antigenicity(TCell[i]))
            Ttangonkar.append(ProteinFeatures.antigenicity(TCell[i], window_size=5))
            Temini.append(ProteinFeatures.emini_surface_accessibility(TCell[i]))
            Tsimilar.append(perform_blastp(TCell[i], self.blast_activate))
            TPhysicochemical.append(ProtParamClone(TCell[i]).calculate())
        
        # Docking calculations
        classical_dock1B, classical_dock1T = ClassicalDocking(
            BCell, TCell, self.base_path, self.target_path, self.n_receptor).ForceField1()
        
        classical_dock1BAdjuvant, classical_dock1TAdjuvant = ClassicalDockingWithAdjuvant(
            BCell, TCell, self.base_path, self.target_path, self.n_receptor, self.n_adjuvant).ForceField1()

        dock1B, dock1T = MLDocking(
            BCell, TCell, self.base_path, self.target_path, self.n_receptor).MLDock1()
        
        dock1BAdjuvant, dock1TAdjuvant = MLDockingWithAdjuvant(
            BCell, TCell, self.base_path, self.target_path, self.n_receptor, self.n_adjuvant).MLDock1()

        # Compile results
        pred = {
            'seq': {'B': BCell, 'T': TCell},
            'allergenicity': {
                'B': Ballergen, 'T': Tallergen,
                'B_proba': BallergenProb, 'T_proba': TallergenProb
            },
            'toxin': {
                'B': Btoxin, 'T': Ttoxin,
                'B_proba': BtoxinProb, 'T_proba': TtoxinProb
            },
            'antigenicity': {
                'B': BAntigen, 'T': TAntigen,
                'B_proba': BAntigenProb, 'T_proba': TAntigenProb
            },
            'hydrophobicity': {'B': Bhydrophobicity, 'T': Thydrophobicity},
            'kolaskar': {'B': Bkolaskar, 'T': Tkolaskar},
            'tangonkar': {'B': Btangonkar, 'T': Ttangonkar},
            'emini': {'B': Bemini, 'T': Temini},
            'similarity': {'B': Bsimilar, 'T': Tsimilar},
            'physicochemical': {'B': BPhysicochemical, 'T': TPhysicochemical},
            'classical dock(Force Field)': {'B': classical_dock1B, 'T': classical_dock1T},
            'classical dock(Force Field) With Adjuvant': {'B': classical_dock1BAdjuvant, 'T': classical_dock1TAdjuvant},
            'Machine Learning based dock': {'B': dock1B, 'T': dock1T},
            'Machine Learning based dock With Adjuvant': {'B': dock1BAdjuvant, 'T': dock1TAdjuvant},
        }

        # Export results
        self._export_results(pred)

        return pred

    def _export_results(self, pred):
        """Export results to files"""
        # Create DataFrames for B and T cells
        sliced_pred = {key: pred[key] for key in list(pred.keys())[:9]}
        B_data = {key: sliced_pred[key]['B'] for key in sliced_pred.keys() if key != 'seq'}
        T_data = {key: sliced_pred[key]['T'] for key in sliced_pred.keys() if key != 'seq'}

        B_result = pd.DataFrame(B_data)
        T_result = pd.DataFrame(T_data)

        print("DataFrames exported to B_result.csv and T_result.csv")

        # Export to CSV files
        B_result.to_csv(f"{self.target_path}/B_result.csv", index=False)
        T_result.to_csv(f"{self.target_path}/T_result.csv", index=False)

        # LLM review using local model
        try:
            B_review = run_llm_review(f'{self.target_path}/B_result.csv', model_name=self.llm_model_name)
            T_review = run_llm_review(f'{self.target_path}/T_result.csv', model_name=self.llm_model_name)
            export_string_to_text_file(B_review, f'{self.target_path}/B_res_review.txt')
            export_string_to_text_file(T_review, f'{self.target_path}/T_res_review.txt')
        except Exception as e:
            print(f"Error in LLM review: {e}")

        # AlphaFold modeling if URL provided
        if self.alphafold_url:
            try:
                self._perform_alphafold_modeling(pred)
            except Exception as e:
                print(f"Error in AlphaFold modeling: {e}")

        # Export to JSON
        file_path = f'{self.target_path}/{generate_filename_with_timestamp_and_random()}_eval_res.json'
        with open(file_path, 'w') as json_file:
            json.dump(str(pred), json_file)

    def _perform_alphafold_modeling(self, pred):
        """Perform AlphaFold protein structure modeling"""
        import requests
        from .utils import create_folder, download_file
        
        alphafold_res_dir = f'{self.target_path}/Alphafold Modelling Result'
        create_folder(alphafold_res_dir)
        create_folder(f'{alphafold_res_dir}/B')
        create_folder(f'{alphafold_res_dir}/T')

        url = self.alphafold_url
        if url[-1] != '/':
            url += '/'

        for epitope_type in ['B', 'T']:
            for i, seq in enumerate(pred['seq'][epitope_type]):
                response = requests.get(
                    url + f"?protein_sequence={seq}&jobname={epitope_type}_{i}_3D_{seq}", 
                    verify=False, timeout=6000
                )
                if response.status_code == 200:
                    response_res = json.loads(response.text)
                    print(response_res)
                    try:
                        download_file(
                            url + response_res['result'],
                            f"{alphafold_res_dir}/{epitope_type}/{epitope_type}_{i}_3D_{seq}.zip"
                        )
                    except Exception as e:
                        print(f"Error downloading AlphaFold result: {e}")
                else:
                    print("Failed Protein Modelling")
                    continue

    def predict(self):
        """Main prediction method"""
        print("Starting Predict Epitope...")
        pred1, pred2 = self.predict_epitope()
        print("Starting Predict Evaluation For Epitope...")
        pred_eval = self.predict_eval(pred2['B'], pred2['T'])
        print("Finished Predict")
        return pred1, pred2, pred_eval 