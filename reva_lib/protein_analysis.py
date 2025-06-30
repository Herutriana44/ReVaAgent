"""
Protein analysis utilities for ReVa AI Vaccine Design Library
"""

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import IsoelectricPoint as IP
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

class ProtParamClone:
    """Protein parameter analysis class"""
    
    def __init__(self, seq):
        self.seq = self.preprocessing_begin(seq)
        
        # Hydropathicity values for amino acids
        self.hydropathy_values = {
            'A': 1.800, 'R': -4.500, 'N': -3.500, 'D': -3.500, 'C': 2.500,
            'E': -3.500, 'Q': -3.500, 'G': -0.400, 'H': -3.200, 'I': 4.500,
            'L': 3.800, 'K': -3.900, 'M': 1.900, 'F': 2.800, 'P': -1.600,
            'S': -0.800, 'T': -0.700, 'W': -0.900, 'Y': -1.300, 'V': 4.200
        }

        # Extinction coefficients
        self.extinction_coefficients = {
            'Tyr': 1490, 'Trp': 5500, 'Cystine': 125
        }

        # Half-life values
        self.half_life_values = {
            'A': {'Mammalian': 4.4, 'Yeast': 20, 'E. coli': 10},
            'R': {'Mammalian': 1, 'Yeast': 2, 'E. coli': 2},
            'N': {'Mammalian': 1.4, 'Yeast': 3, 'E. coli': 10},
            'D': {'Mammalian': 1.1, 'Yeast': 3, 'E. coli': 10},
            'C': {'Mammalian': 1.2, 'Yeast': 20, 'E. coli': 10},
            'E': {'Mammalian': 0.8, 'Yeast': 10, 'E. coli': 10},
            'Q': {'Mammalian': 1, 'Yeast': 30, 'E. coli': 10},
            'G': {'Mammalian': 30, 'Yeast': 20, 'E. coli': 10},
            'H': {'Mammalian': 3.5, 'Yeast': 10, 'E. coli': 10},
            'I': {'Mammalian': 20, 'Yeast': 30, 'E. coli': 10},
            'L': {'Mammalian': 5.5, 'Yeast': 3, 'E. coli': 2},
            'K': {'Mammalian': 1.3, 'Yeast': 3, 'E. coli': 2},
            'M': {'Mammalian': 30, 'Yeast': 20, 'E. coli': 10},
            'F': {'Mammalian': 1.1, 'Yeast': 3, 'E. coli': 2},
            'P': {'Mammalian': 20, 'Yeast': 20, 'E. coli': 0},
            'S': {'Mammalian': 1.9, 'Yeast': 20, 'E. coli': 10},
            'T': {'Mammalian': 7.2, 'Yeast': 20, 'E. coli': 10},
            'W': {'Mammalian': 2.8, 'Yeast': 3, 'E. coli': 2},
            'Y': {'Mammalian': 2.8, 'Yeast': 10, 'E. coli': 2},
            'V': {'Mammalian': 100, 'Yeast': 20, 'E. coli': 10}
        }

    @staticmethod
    def preprocessing_begin(seq):
        """Preprocess protein sequence"""
        seq = str(seq).upper()
        delete_char = "BJOUXZ\n\t 1234567890*&^%$#@!~()[];:',.<><?/"
        for i in range(len(delete_char)):
            seq = seq.replace(delete_char[i], '')
        return seq

    def calculate_instability_index(self, sequence):
        """Calculate instability index"""
        try:
            X = ProteinAnalysis(sequence)
            return X.instability_index()
        except:
            print("Error calculating instability index")
            return None

    @staticmethod
    def calculate_aliphatic_index(sequence):
        """Calculate aliphatic index"""
        X_Ala = (sequence.count('A') / len(sequence)) * 100
        X_Val = (sequence.count('V') / len(sequence)) * 100
        X_Ile = (sequence.count('I') / len(sequence)) * 100
        X_Leu = (sequence.count('L') / len(sequence)) * 100
        aliphatic_index = X_Ala + 2.9 * X_Val + 3.9 * (X_Ile + X_Leu)
        return aliphatic_index

    def calculate_gravy(self, sequence):
        """Calculate GRAVY (Grand Average of Hydropathy)"""
        try:
            gravy = sum(self.hydropathy_values.get(aa, 0) for aa in sequence) / len(sequence)
            return gravy
        except:
            print("Error calculating GRAVY")
            return None

    def calculate_extinction_coefficient(self, sequence):
        """Calculate extinction coefficient"""
        num_Tyr = sequence.count('Y')
        num_Trp = sequence.count('W')
        num_Cystine = sequence.count('C')
        extinction_prot = (num_Tyr * self.extinction_coefficients['Tyr'] +
                          num_Trp * self.extinction_coefficients['Trp'] +
                          num_Cystine * self.extinction_coefficients['Cystine'])
        return extinction_prot

    def predict_half_life(self, sequence, organism='Mammalian'):
        """Predict protein half-life"""
        n_terminal_residue = sequence[0]
        half_life = self.half_life_values.get(n_terminal_residue, {}).get(organism, 'N/A')
        return half_life

    @staticmethod
    def calculate_atom_composition(molecule):
        """Calculate atom composition"""
        atom_composition = {}
        atoms = molecule.GetAtoms()
        for atom in atoms:
            atom_symbol = atom.GetSymbol()
            if atom_symbol in atom_composition:
                atom_composition[atom_symbol] += 1
            else:
                atom_composition[atom_symbol] = 1
        return atom_composition

    @staticmethod
    def calculate_HNOSt_composition(molecule):
        """Calculate H, N, O, S composition"""
        composition = ProtParamClone.calculate_atom_composition(molecule)
        C_count = composition.get('C', 0)
        H_count = composition.get('H', 0)
        N_count = composition.get('N', 0)
        O_count = composition.get('O', 0)
        S_count = composition.get('S', 0)
        return C_count, H_count, N_count, O_count, S_count

    @staticmethod
    def calculate_theoretical_pI(protein_sequence):
        """Calculate theoretical pI"""
        try:
            X = ProteinAnalysis(protein_sequence)
            pI = X.isoelectric_point()
            return pI
        except:
            print("Error calculating theoretical pI")
            return None
        
    def MolWeight(self):
        """Calculate molecular weight"""
        try:
            mol = Chem.MolFromSequence(self.seq)
            mol_weight = Descriptors.MolWt(mol)
            return mol_weight
        except:
            print("Error calculating molecular weight")
            return None

    def calculate(self):
        """Calculate all protein parameters"""
        try:
            # Calculate instability index
            try:
                instability = self.calculate_instability_index(self.seq)
            except:
                print("Error calculating instability index")
                instability = None

            # Calculate aliphatic index
            aliphatic = self.calculate_aliphatic_index(self.seq)

            # Calculate GRAVY
            try:
                calculate_gravy = self.calculate_gravy(self.seq)
            except:
                print("Error calculating GRAVY")
                calculate_gravy = None

            # Calculate extinction coefficient
            extinction = self.calculate_extinction_coefficient(self.seq)

            # Calculate half-life
            try:
                half_life = self.predict_half_life(sequence=self.seq)
            except:
                print("Error calculating half-life")
                half_life = None

            # Calculate molecular formula
            mol = Chem.MolFromSequence(self.seq)
            try:
                formula = rdMolDescriptors.CalcMolFormula(mol)
            except:
                print("Error calculating molecular formula")
                formula = None

            # Calculate atom composition
            Cn, Hn, Nn, On, Sn = self.calculate_HNOSt_composition(mol)

            # Calculate theoretical pI
            try:
                theoretical_pI = IP.IsoelectricPoint(self.seq).pi()
            except:
                print("Error calculating theoretical pI")
                theoretical_pI = None

            # Calculate molecular weight
            m_weight = self.MolWeight()

            result = {
                'seq': self.seq,
                'instability': instability,
                'aliphatic': aliphatic,
                'gravy': calculate_gravy,
                'extinction': extinction,
                'half_life': half_life,
                'formula': formula,
                'C': Cn,
                'H': Hn,
                'N': Nn,
                'O': On,
                'S': Sn,
                'theoretical_pI': theoretical_pI,
                'mol_weight': m_weight
            }

            return result
        except Exception as e:
            print(f"Error calculating physicochemical properties: {e}")
            return None

class ProteinFeatures:
    """Protein feature calculation class"""
    
    @staticmethod
    def calculate_hydrophobicity(sequence):
        """Calculate hydrophobicity score"""
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

    @staticmethod
    def antigenicity(sequence, window_size=7):
        """Calculate antigenicity scores"""
        antigenicity_scores = []
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            antigenicity_score = sum([1 if window[j] == 'A' or window[j] == 'G' else 0 for j in range(window_size)])
            antigenicity_scores.append(antigenicity_score)
        return antigenicity_scores
    
    @staticmethod
    def emini_surface_accessibility(sequence, window_size=9):
        """Calculate Emini surface accessibility scores"""
        surface_accessibility_scores = []
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            surface_accessibility_score = sum([1 if window[j] in ['S', 'T', 'N', 'Q'] else 0 for j in range(window_size)])
            surface_accessibility_scores.append(surface_accessibility_score)
        return surface_accessibility_scores 