"""
Docking utilities for ReVa AI Vaccine Design Library
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from .utils import get_data_dir
from .molecular_utils import MolecularFunction, MolecularScoring
from .protein_analysis import ProtParamClone
from scipy.constants import e

def ml_dock(data):
    """Classical ML docking function"""
    from .utils import load_pickle_model, get_model_dir
    import pickle
    
    # Load model
    model_path = os.path.join(get_model_dir(), 'Linear_Regression_Model.pkl')
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Make predictions
    predictions = model.predict([data])
    return predictions[0]

def qml_dock(data, sampler=None):
    """Quantum ML docking function"""
    from qiskit_machine_learning.algorithms import VQR
    from .utils import get_qmodel_dir
    
    try:
        if sampler is None:
            # Load model
            vqr = VQR.load(os.path.join(get_qmodel_dir(), "VQR_quantum regression-based scoring function"))
            prediction = vqr.predict([data])
            return prediction[0][0]
        else:
            # Load model
            vqr = VQR.load(os.path.join(get_qmodel_dir(), "VQR_quantum regression-based scoring function"))
            prediction = vqr.predict([data])
            return prediction[0][0]
    except:
        return None

class ClassicalDocking:
    """Classical force field-based docking"""
    
    def __init__(self, bseq, tseq, base_path, target_path, n_receptor):
        self.bseq = bseq
        self.tseq = tseq
        self.base_path = base_path
        self.target_path = target_path
        self.n_receptor = n_receptor

    def ForceField1(self):
        """Perform force field-based docking"""
        # B-cell receptor docking
        b_cell_receptor = pd.read_csv(os.path.join(get_data_dir(), 'b cell receptor homo sapiens.csv'))
        b_cell_receptor = b_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        b_cell_receptor = b_cell_receptor[0:self.n_receptor]
        b_epitope = self.bseq

        seq1, seq2, seq2id = [], [], []
        com1, com2 = [], []
        Attractive, Repulsive = [], []
        VDW_lj_force, coulomb_energy, force_field = [], [], []
        
        for b in b_epitope:
            for i in range(len(b_cell_receptor)):
                seq1.append(b)
                seq2.append(b_cell_receptor['seq'][i])
                seq2id.append(b_cell_receptor['id'][i])
                
                aa1 = MolecularFunction.calculate_amino_acid_center_of_mass(b)
                aa2 = MolecularFunction.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                com1.append(aa1)
                com2.append(aa2)
                
                dist = MolecularFunction.calculate_distance_between_amino_acids(aa1, aa2)
                attract = MolecularScoring.attractive_energy(dist)
                Attractive.append(attract)
                
                repulsive = MolecularScoring.repulsive_energy(dist)
                Repulsive.append(repulsive)
                
                vdw = MolecularScoring.lj_force(dist)
                VDW_lj_force.append(vdw)
                
                ce = MolecularScoring.coulomb_energy(e, e, dist)
                coulomb_energy.append(ce)
                force_field.append(vdw + ce)
        
        b_res = {
            'Ligand': seq1,
            'Receptor': seq2,
            'Receptor id': seq2id,
            'Center Of Ligand Mass': com1,
            'Center Of Receptor Mass': com2,
            'Attractive': Attractive,
            'Repulsive': Repulsive,
            'VDW LJ Force': VDW_lj_force,
            'Coulomb Energy': coulomb_energy,
            'Force Field': force_field
        }
        b_res_df = pd.DataFrame(b_res)
        print("Force Field B Cell Success")

        # T-cell receptor docking
        t_cell_receptor = pd.read_csv(os.path.join(get_data_dir(), 't cell receptor homo sapiens.csv'))
        t_cell_receptor = t_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        t_cell_receptor = t_cell_receptor[0:self.n_receptor]
        t_epitope = self.tseq

        seq1, seq2, seq2id = [], [], []
        com1, com2 = [], []
        Attractive, Repulsive = [], []
        VDW_lj_force, coulomb_energy, force_field = [], [], []
        
        for t in t_epitope:
            for i in range(len(t_cell_receptor)):
                seq1.append(t)
                seq2.append(t_cell_receptor['seq'][i])
                seq2id.append(t_cell_receptor['id'][i])
                
                aa1 = MolecularFunction.calculate_amino_acid_center_of_mass(t)
                aa2 = MolecularFunction.calculate_amino_acid_center_of_mass(t_cell_receptor['seq'][i])
                com1.append(aa1)
                com2.append(aa2)
                
                dist = MolecularFunction.calculate_distance_between_amino_acids(aa1, aa2)
                attract = MolecularScoring.attractive_energy(dist)
                Attractive.append(attract)
                
                repulsive = MolecularScoring.repulsive_energy(dist)
                Repulsive.append(repulsive)
                
                vdw = MolecularScoring.lj_force(dist)
                VDW_lj_force.append(vdw)
                
                ce = MolecularScoring.coulomb_energy(e, e, dist)
                coulomb_energy.append(ce)
                force_field.append(vdw + ce)

        t_res = {
            'Ligand': seq1,
            'Receptor': seq2,
            'Receptor id': seq2id,
            'Center Of Ligand Mass': com1,
            'Center Of Receptor Mass': com2,
            'Attractive': Attractive,
            'Repulsive': Repulsive,
            'VDW LJ Force': VDW_lj_force,
            'Coulomb Energy': coulomb_energy,
            'Force Field': force_field
        }
        t_res_df = pd.DataFrame(t_res)
        print("Force Field T Cell Success")

        # Save results
        b_res_df.to_excel(self.target_path + '/' + 'forcefield_b_cell.xlsx', index=False)
        t_res_df.to_excel(self.target_path + '/' + 'forcefield_t_cell.xlsx', index=False)

        return b_res, t_res

class ClassicalDockingWithAdjuvant:
    """Classical force field-based docking with adjuvant"""
    
    def __init__(self, bseq, tseq, base_path, target_path, n_receptor, n_adjuvant):
        self.bseq = bseq
        self.tseq = tseq
        self.base_path = base_path
        self.target_path = target_path
        self.n_receptor = n_receptor
        self.n_adjuvant = n_adjuvant

    def ForceField1(self):
        """Perform force field-based docking with adjuvant"""
        adjuvant = pd.read_csv(os.path.join(get_data_dir(), 'PubChem_compound_text_adjuvant.csv'))
        adjuvant = adjuvant.sample(frac=1, random_state=42).reset_index(drop=True)
        adjuvant = adjuvant[0:self.n_adjuvant]
        
        b_cell_receptor = pd.read_csv(os.path.join(get_data_dir(), 'b cell receptor homo sapiens.csv'))
        b_cell_receptor = b_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        b_cell_receptor = b_cell_receptor[0:self.n_receptor]
        b_epitope = self.bseq

        seq1, seq2, seq2id = [], [], []
        adjuvant_list, adjuvant_isosmiles = [], []
        com1, com2 = [], []
        Attractive, Repulsive = [], []
        VDW_lj_force, coulomb_energy, force_field = [], [], []
        
        for b in b_epitope:
            for i in range(len(b_cell_receptor)):
                for adju in range(len(adjuvant)):
                    adjuvant_list.append(adjuvant['cid'][adju])
                    adjuvant_isosmiles.append(adjuvant['isosmiles'][adju])
                    
                    seq_plus_adjuvant = Chem.MolToSmiles(
                        MolecularFunction.combine_epitope_with_adjuvant(b, adjuvant['isosmiles'][adju]))
                    
                    seq1.append(seq_plus_adjuvant)
                    seq2.append(b_cell_receptor['seq'][i])
                    seq2id.append(b_cell_receptor['id'][i])
                    
                    aa1 = MolecularFunction.calculate_amino_acid_center_of_mass_smiles(seq_plus_adjuvant)
                    aa2 = MolecularFunction.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                    com1.append(aa1)
                    com2.append(aa2)
                    
                    dist = MolecularFunction.calculate_distance_between_amino_acids(aa1, aa2)
                    attract = MolecularScoring.attractive_energy(dist)
                    Attractive.append(attract)
                    
                    repulsive = MolecularScoring.repulsive_energy(dist)
                    Repulsive.append(repulsive)
                    
                    vdw = MolecularScoring.lj_force(dist)
                    VDW_lj_force.append(vdw)
                    
                    ce = MolecularScoring.coulomb_energy(e, e, dist)
                    coulomb_energy.append(ce)
                    force_field.append(vdw + ce)

        b_res = {
            'Ligand': seq1,
            'Receptor': seq2,
            'Receptor id': seq2id,
            'Adjuvant CID': adjuvant_list,
            'Adjuvant IsoSMILES': adjuvant_isosmiles,
            'Attractive': Attractive,
            'Repulsive': Repulsive,
            'Center Of Ligand Mass': com1,
            'Center Of Receptor Mass': com2,
            'VDW LJ Force': VDW_lj_force,
            'Coulomb Energy': coulomb_energy,
            'Force Field': force_field
        }
        b_res_df = pd.DataFrame(b_res)
        print("Force Field B Cell Success With Adjuvant")

        # T-cell receptor docking with adjuvant
        t_cell_receptor = pd.read_csv(os.path.join(get_data_dir(), 't cell receptor homo sapiens.csv'))
        t_cell_receptor = t_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        t_cell_receptor = t_cell_receptor[0:self.n_receptor]
        t_epitope = self.tseq

        seq1, seq2, seq2id = [], [], []
        adjuvant_list, adjuvant_isosmiles = [], []
        com1, com2 = [], []
        Attractive, Repulsive = [], []
        VDW_lj_force, coulomb_energy, force_field = [], [], []
        
        for t in t_epitope:
            for i in range(len(t_cell_receptor)):
                for adju in range(len(adjuvant)):
                    adjuvant_list.append(adjuvant['cid'][adju])
                    adjuvant_isosmiles.append(adjuvant['isosmiles'][adju])
                    
                    seq_plus_adjuvant = Chem.MolToSmiles(
                        MolecularFunction.combine_epitope_with_adjuvant(t, adjuvant['isosmiles'][adju]))
                    
                    seq1.append(seq_plus_adjuvant)
                    seq2.append(t_cell_receptor['seq'][i])
                    seq2id.append(t_cell_receptor['id'][i])
                    
                    aa1 = MolecularFunction.calculate_amino_acid_center_of_mass_smiles(seq_plus_adjuvant)
                    aa2 = MolecularFunction.calculate_amino_acid_center_of_mass(t_cell_receptor['seq'][i])
                    com1.append(aa1)
                    com2.append(aa2)
                    
                    dist = MolecularFunction.calculate_distance_between_amino_acids(aa1, aa2)
                    attract = MolecularScoring.attractive_energy(dist)
                    Attractive.append(attract)
                    
                    repulsive = MolecularScoring.repulsive_energy(dist)
                    Repulsive.append(repulsive)
                    
                    vdw = MolecularScoring.lj_force(dist)
                    VDW_lj_force.append(vdw)
                    
                    ce = MolecularScoring.coulomb_energy(e, e, dist)
                    coulomb_energy.append(ce)
                    force_field.append(vdw + ce)

        t_res = {
            'Ligand': seq1,
            'Receptor': seq2,
            'Receptor id': seq2id,
            'Adjuvant CID': adjuvant_list,
            'Adjuvant IsoSMILES': adjuvant_isosmiles,
            'Attractive': Attractive,
            'Repulsive': Repulsive,
            'Center Of Ligand Mass': com1,
            'Center Of Receptor Mass': com2,
            'VDW LJ Force': VDW_lj_force,
            'Coulomb Energy': coulomb_energy,
            'Force Field': force_field
        }
        t_res_df = pd.DataFrame(t_res)
        print("Force Field T Cell Success With Adjuvant")

        # Save results
        b_res_df.to_excel(self.target_path + '/' + 'forcefield_b_cell_with_adjuvant.xlsx', index=False)
        t_res_df.to_excel(self.target_path + '/' + 'forcefield_t_cell_with_adjuvant.xlsx', index=False)

        return b_res, t_res

class MLDocking:
    """Machine learning-based docking"""
    
    def __init__(self, bseq, tseq, base_path, target_path, n_receptor):
        self.bseq = bseq
        self.tseq = tseq
        self.base_path = base_path
        self.target_path = target_path
        self.n_receptor = n_receptor

    def MLDock1(self):
        """Perform ML-based docking"""
        # B-cell receptor docking
        b_cell_receptor = pd.read_csv(os.path.join(get_data_dir(), 'b_receptor_v2.csv'))
        b_cell_receptor = b_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        b_cell_receptor = b_cell_receptor[0:self.n_receptor]
        b_epitope = self.bseq

        seq1, seq2, seq2id = [], [], []
        smilesseq1, smilesseq2 = [], []
        molwseq1, molwseq2 = [], []
        bdist, bmol_pred = [], []
        com1, com2 = [], []
        
        for b in b_epitope:
            for i in range(len(b_cell_receptor)):
                seq1.append(b)
                seq2.append(b_cell_receptor['seq'][i])
                seq2id.append(b_cell_receptor['id'][i])
                
                aa1 = MolecularFunction.calculate_amino_acid_center_of_mass(b)
                smiles1 = MolecularFunction.seq_to_smiles(b)
                smilesseq1.append(smiles1)
                molwt1 = MolecularFunction.calculate_molecular_weight(smiles1)
                molwseq1.append(molwt1)
                
                aa2 = MolecularFunction.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                smiles2 = MolecularFunction.seq_to_smiles(b_cell_receptor['seq'][i])
                smilesseq2.append(smiles2)
                molwt2 = MolecularFunction.calculate_molecular_weight(smiles2)
                molwseq2.append(molwt2)
                
                dist = MolecularFunction.calculate_distance_between_amino_acids(aa1, aa2)
                bdist.append(dist)
                com1.append(aa1)
                com2.append(aa2)
                
                feature = [aa1, molwt1, aa2, molwt2, dist]
                bmol_pred.append(ml_dock(feature))
        
        b_res = {
            'Ligand': seq1,
            'Receptor': seq2,
            'Receptor id': seq2id,
            'Ligand Smiles': smilesseq1,
            'Receptor Smiles': smilesseq2,
            'Center Of Mass Ligand': com1,
            'Center Of Mass Receptor': com2,
            'Molecular Weight Of Ligand': molwseq1,
            'Molecular Weight Of Receptor': molwseq2,
            'Distance': bdist,
            'Docking(Ki (nM))': bmol_pred
        }

        b_res_df = pd.DataFrame(b_res)
        print("Machine Learning Based Scoring Function of B Cell Success")

        # T-cell receptor docking
        t_cell_receptor = pd.read_csv(os.path.join(get_data_dir(), 't_receptor_v2.csv'))
        t_cell_receptor = t_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        t_cell_receptor = t_cell_receptor[0:self.n_receptor]
        t_epitope = self.tseq

        seq1, seq2, seq2id = [], [], []
        smilesseq1, smilesseq2 = [], []
        molwseq1, molwseq2 = [], []
        bdist, bmol_pred = [], []
        com1, com2 = [], []
        
        for t in t_epitope:
            for i in range(len(t_cell_receptor)):
                seq1.append(t)
                seq2.append(t_cell_receptor['seq'][i])
                seq2id.append(t_cell_receptor['id'][i])
                
                aa1 = MolecularFunction.calculate_amino_acid_center_of_mass(t)
                smiles1 = MolecularFunction.seq_to_smiles(t)
                smilesseq1.append(smiles1)
                molwt1 = MolecularFunction.calculate_molecular_weight(smiles1)
                molwseq1.append(molwt1)
                
                aa2 = MolecularFunction.calculate_amino_acid_center_of_mass(t_cell_receptor['seq'][i])
                smiles2 = MolecularFunction.seq_to_smiles(t_cell_receptor['seq'][i])
                smilesseq2.append(smiles2)
                molwt2 = MolecularFunction.calculate_molecular_weight(smiles2)
                molwseq2.append(molwt2)
                
                dist = MolecularFunction.calculate_distance_between_amino_acids(aa1, aa2)
                bdist.append(dist)
                com1.append(aa1)
                com2.append(aa2)
                
                feature = [aa1, molwt1, aa2, molwt2, dist]
                bmol_pred.append(ml_dock(feature))
        
        t_res = {
            'Ligand': seq1,
            'Receptor': seq2,
            'Receptor id': seq2id,
            'Ligand Smiles': smilesseq1,
            'Receptor Smiles': smilesseq2,
            'Center Of Mass Ligand': com1,
            'Center Of Mass Receptor': com2,
            'Molecular Weight Of Ligand': molwseq1,
            'Molecular Weight Of Receptor': molwseq2,
            'Distance': bdist,
            'Docking(Ki (nM))': bmol_pred
        }

        t_res_df = pd.DataFrame(t_res)
        print("Machine Learning Based Scoring Function of T Cell Success")

        # Save results
        b_res_df.to_excel(self.target_path + '/' + 'ml_scoring_func_b_cell.xlsx', index=False)
        t_res_df.to_excel(self.target_path + '/' + 'ml_scoring_func_t_cell.xlsx', index=False)

        return b_res, t_res

class MLDockingWithAdjuvant:
    """Machine learning-based docking with adjuvant"""
    
    def __init__(self, bseq, tseq, base_path, target_path, n_receptor, n_adjuvant):
        self.bseq = bseq
        self.tseq = tseq
        self.base_path = base_path
        self.target_path = target_path
        self.n_receptor = n_receptor
        self.n_adjuvant = n_adjuvant

    def MLDock1(self):
        """Perform ML-based docking with adjuvant"""
        adjuvant = pd.read_csv(os.path.join(get_data_dir(), 'PubChem_compound_text_adjuvant.csv'))
        adjuvant = adjuvant.sample(frac=1, random_state=42).reset_index(drop=True)
        adjuvant = adjuvant[0:self.n_adjuvant]
        
        b_cell_receptor = pd.read_csv(os.path.join(get_data_dir(), 'b_receptor_v2.csv'))
        b_cell_receptor = b_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        b_cell_receptor = b_cell_receptor[0:self.n_receptor]
        b_epitope = self.bseq

        seq1, seq2, seq2id = [], [], []
        adjuvant_list, adjuvant_isosmiles = [], []
        smilesseq1, smilesseq2 = [], []
        molwseq1, molwseq2 = [], []
        bdist, bmol_pred = [], []
        com1, com2 = [], []
        
        for b in b_epitope:
            for i in range(len(b_cell_receptor)):
                for adju in range(len(adjuvant)):
                    adjuvant_list.append(adjuvant['cid'][adju])
                    adjuvant_isosmiles.append(adjuvant['isosmiles'][adju])
                    
                    seq_plus_adjuvant = Chem.MolToSmiles(
                        MolecularFunction.combine_epitope_with_adjuvant(b, adjuvant['isosmiles'][adju]))
                    
                    seq1.append(b)
                    seq2.append(b_cell_receptor['seq'][i])
                    seq2id.append(b_cell_receptor['id'][i])
                    
                    aa1 = MolecularFunction.calculate_amino_acid_center_of_mass_smiles(seq_plus_adjuvant)
                    smiles1 = seq_plus_adjuvant
                    smilesseq1.append(smiles1)
                    molwt1 = MolecularFunction.calculate_molecular_weight(smiles1)
                    molwseq1.append(molwt1)
                    
                    aa2 = MolecularFunction.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                    smiles2 = MolecularFunction.seq_to_smiles(b_cell_receptor['seq'][i])
                    smilesseq2.append(smiles2)
                    molwt2 = MolecularFunction.calculate_molecular_weight(smiles2)
                    molwseq2.append(molwt2)
                    
                    dist = MolecularFunction.calculate_distance_between_amino_acids(aa1, aa2)
                    bdist.append(dist)
                    feature = [aa1, molwt1, aa2, molwt2, dist]
                    com1.append(aa1)
                    com2.append(aa2)

                    bmol_pred.append(ml_dock(feature))

        b_res = {
            'Ligand': seq1,
            'Receptor': seq2,
            'Receptor id': seq2id,
            'Adjuvant CID': adjuvant_list,
            'Adjuvant IsoSMILES': adjuvant_isosmiles,
            'Ligand Smiles': smilesseq1,
            'Receptor Smiles': smilesseq2,
            'Center Of Mass Ligand': com1,
            'Center Of Mass Receptor': com2,
            'Molecular Weight Of Ligand': molwseq1,
            'Molecular Weight Of Receptor': molwseq2,
            'Distance': bdist,
            'Docking(Ki (nM))': bmol_pred
        }

        b_res_df = pd.DataFrame(b_res)
        print("Machine Learning Based Scoring Function of B Cell Success With Adjuvant")

        # T-cell receptor docking with adjuvant
        t_cell_receptor = pd.read_csv(os.path.join(get_data_dir(), 't cell receptor homo sapiens.csv'))
        t_cell_receptor = t_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        t_cell_receptor = t_cell_receptor[0:self.n_receptor]
        t_epitope = self.tseq

        seq1, seq2, seq2id = [], [], []
        adjuvant_list, adjuvant_isosmiles = [], []
        smilesseq1, smilesseq2 = [], []
        molwseq1, molwseq2 = [], []
        bdist, bmol_pred = [], []
        com1, com2 = [], []
        
        for t in t_epitope:
            for i in range(len(t_cell_receptor)):
                for adju in range(len(adjuvant)):
                    adjuvant_list.append(adjuvant['cid'][adju])
                    adjuvant_isosmiles.append(adjuvant['isosmiles'][adju])
                    
                    seq_plus_adjuvant = Chem.MolToSmiles(
                        MolecularFunction.combine_epitope_with_adjuvant(t, adjuvant['isosmiles'][adju]))
                    
                    seq1.append(t)
                    seq2.append(t_cell_receptor['seq'][i])
                    seq2id.append(t_cell_receptor['id'][i])
                    
                    aa1 = MolecularFunction.calculate_amino_acid_center_of_mass_smiles(seq_plus_adjuvant)
                    smiles1 = seq_plus_adjuvant
                    smilesseq1.append(smiles1)
                    molwt1 = MolecularFunction.calculate_molecular_weight(smiles1)
                    molwseq1.append(molwt1)
                    
                    aa2 = MolecularFunction.calculate_amino_acid_center_of_mass(t_cell_receptor['seq'][i])
                    smiles2 = MolecularFunction.seq_to_smiles(t_cell_receptor['seq'][i])
                    smilesseq2.append(smiles2)
                    molwt2 = MolecularFunction.calculate_molecular_weight(smiles2)
                    molwseq2.append(molwt2)
                    
                    dist = MolecularFunction.calculate_distance_between_amino_acids(aa1, aa2)
                    bdist.append(dist)
                    feature = [aa1, molwt1, aa2, molwt2, dist]
                    com1.append(aa1)
                    com2.append(aa2)

                    bmol_pred.append(ml_dock(feature))

        t_res = {
            'Ligand': seq1,
            'Receptor': seq2,
            'Receptor id': seq2id,
            'Adjuvant CID': adjuvant_list,
            'Adjuvant IsoSMILES': adjuvant_isosmiles,
            'Ligand Smiles': smilesseq1,
            'Receptor Smiles': smilesseq2,
            'Center Of Mass Ligand': com1,
            'Center Of Mass Receptor': com2,
            'Molecular Weight Of Ligand': molwseq1,
            'Molecular Weight Of Receptor': molwseq2,
            'Distance': bdist,
            'Docking(Ki (nM))': bmol_pred
        }
        
        t_res_df = pd.DataFrame(t_res)
        print("Machine Learning Based Scoring Function of T Cell Success With Adjuvant")

        # Save results
        b_res_df.to_excel(self.target_path + '/' + 'ml_b_cell_with_adjuvant.xlsx', index=False)
        t_res_df.to_excel(self.target_path + '/' + 'ml_t_cell_with_adjuvant.xlsx', index=False)

        return b_res, t_res

class QMLDocking:
    """Quantum machine learning-based docking"""
    
    def __init__(self, bseq, tseq, base_path, target_path, n_receptor, sampler=None):
        self.bseq = bseq
        self.tseq = tseq
        self.base_path = base_path
        self.target_path = target_path
        self.n_receptor = n_receptor
        self.sampler = sampler

    def MLDock1(self):
        """Perform quantum ML-based docking"""
        # B-cell receptor docking
        b_cell_receptor = pd.read_csv(os.path.join(get_data_dir(), 'b_receptor_v2.csv'))
        b_cell_receptor = b_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        b_cell_receptor = b_cell_receptor[0:self.n_receptor]
        b_epitope = self.bseq

        seq1, seq2, seq2id = [], [], []
        smilesseq1, smilesseq2 = [], []
        molwseq1, molwseq2 = [], []
        bdist, bmol_pred = [], []
        com1, com2 = [], []
        
        for b in b_epitope:
            for i in range(len(b_cell_receptor)):
                seq1.append(b)
                seq2.append(b_cell_receptor['seq'][i])
                seq2id.append(b_cell_receptor['id'][i])
                
                aa1 = MolecularFunction.calculate_amino_acid_center_of_mass(b)
                smiles1 = MolecularFunction.seq_to_smiles(b)
                smilesseq1.append(smiles1)
                molwt1 = MolecularFunction.calculate_molecular_weight(smiles1)
                molwseq1.append(molwt1)
                
                aa2 = MolecularFunction.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                smiles2 = MolecularFunction.seq_to_smiles(b_cell_receptor['seq'][i])
                smilesseq2.append(smiles2)
                molwt2 = MolecularFunction.calculate_molecular_weight(smiles2)
                molwseq2.append(molwt2)
                
                dist = MolecularFunction.calculate_distance_between_amino_acids(aa1, aa2)
                bdist.append(dist)
                com1.append(aa1)
                com2.append(aa2)
                
                feature = [aa1, molwt1, aa2, molwt2, dist]
                bmol_pred.append(qml_dock(feature, self.sampler))
        
        b_res = {
            'Ligand': seq1,
            'Receptor': seq2,
            'Receptor id': seq2id,
            'Ligand Smiles': smilesseq1,
            'Receptor Smiles': smilesseq2,
            'Center Of Mass Ligand': com1,
            'Center Of Mass Receptor': com2,
            'Molecular Weight Of Ligand': molwseq1,
            'Molecular Weight Of Receptor': molwseq2,
            'Distance': bdist,
            'Docking(Ki (nM))': bmol_pred
        }

        b_res_df = pd.DataFrame(b_res)
        print("Quantum Machine Learning Based Scoring Function of B Cell Success")

        # T-cell receptor docking
        t_cell_receptor = pd.read_csv(os.path.join(get_data_dir(), 't_receptor_v2.csv'))
        t_cell_receptor = t_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        t_cell_receptor = t_cell_receptor[0:self.n_receptor]
        t_epitope = self.tseq

        seq1, seq2, seq2id = [], [], []
        smilesseq1, smilesseq2 = [], []
        molwseq1, molwseq2 = [], []
        bdist, bmol_pred = [], []
        com1, com2 = [], []
        
        for t in t_epitope:
            for i in range(len(t_cell_receptor)):
                seq1.append(t)
                seq2.append(t_cell_receptor['seq'][i])
                seq2id.append(t_cell_receptor['id'][i])
                
                aa1 = MolecularFunction.calculate_amino_acid_center_of_mass(t)
                smiles1 = MolecularFunction.seq_to_smiles(t)
                smilesseq1.append(smiles1)
                molwt1 = MolecularFunction.calculate_molecular_weight(smiles1)
                molwseq1.append(molwt1)
                
                aa2 = MolecularFunction.calculate_amino_acid_center_of_mass(t_cell_receptor['seq'][i])
                smiles2 = MolecularFunction.seq_to_smiles(t_cell_receptor['seq'][i])
                smilesseq2.append(smiles2)
                molwt2 = MolecularFunction.calculate_molecular_weight(smiles2)
                molwseq2.append(molwt2)
                
                dist = MolecularFunction.calculate_distance_between_amino_acids(aa1, aa2)
                bdist.append(dist)
                com1.append(aa1)
                com2.append(aa2)
                
                feature = [aa1, molwt1, aa2, molwt2, dist]
                bmol_pred.append(qml_dock(feature, self.sampler))
        
        t_res = {
            'Ligand': seq1,
            'Receptor': seq2,
            'Receptor id': seq2id,
            'Ligand Smiles': smilesseq1,
            'Receptor Smiles': smilesseq2,
            'Center Of Mass Ligand': com1,
            'Center Of Mass Receptor': com2,
            'Molecular Weight Of Ligand': molwseq1,
            'Molecular Weight Of Receptor': molwseq2,
            'Distance': bdist,
            'Docking(Ki (nM))': bmol_pred
        }

        t_res_df = pd.DataFrame(t_res)
        print("Quantum Machine Learning Based Scoring Function of T Cell Success")

        # Save results
        b_res_df.to_excel(self.target_path + '/' + 'qml_scoring_func_b_cell.xlsx', index=False)
        t_res_df.to_excel(self.target_path + '/' + 'qml_scoring_func_t_cell.xlsx', index=False)

        return b_res, t_res

class QMLDockingWithAdjuvant:
    """Quantum machine learning-based docking with adjuvant"""
    
    def __init__(self, bseq, tseq, base_path, target_path, n_receptor, n_adjuvant, sampler=None):
        self.bseq = bseq
        self.tseq = tseq
        self.base_path = base_path
        self.target_path = target_path
        self.n_receptor = n_receptor
        self.n_adjuvant = n_adjuvant
        self.sampler = sampler

    def MLDock1(self):
        """Perform quantum ML-based docking with adjuvant"""
        adjuvant = pd.read_csv(os.path.join(get_data_dir(), 'PubChem_compound_text_adjuvant.csv'))
        adjuvant = adjuvant.sample(frac=1, random_state=42).reset_index(drop=True)
        adjuvant = adjuvant[0:self.n_adjuvant]
        
        b_cell_receptor = pd.read_csv(os.path.join(get_data_dir(), 'b_receptor_v2.csv'))
        b_cell_receptor = b_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        b_cell_receptor = b_cell_receptor[0:self.n_receptor]
        b_epitope = self.bseq

        seq1, seq2, seq2id = [], [], []
        adjuvant_list, adjuvant_isosmiles = [], []
        smilesseq1, smilesseq2 = [], []
        molwseq1, molwseq2 = [], []
        bdist, bmol_pred = [], []
        com1, com2 = [], []
        
        for b in b_epitope:
            for i in range(len(b_cell_receptor)):
                for adju in range(len(adjuvant)):
                    adjuvant_list.append(adjuvant['cid'][adju])
                    adjuvant_isosmiles.append(adjuvant['isosmiles'][adju])
                    
                    seq_plus_adjuvant = Chem.MolToSmiles(
                        MolecularFunction.combine_epitope_with_adjuvant(b, adjuvant['isosmiles'][adju]))
                    
                    seq1.append(b)
                    seq2.append(b_cell_receptor['seq'][i])
                    seq2id.append(b_cell_receptor['id'][i])
                    
                    aa1 = MolecularFunction.calculate_amino_acid_center_of_mass_smiles(seq_plus_adjuvant)
                    smiles1 = seq_plus_adjuvant
                    smilesseq1.append(smiles1)
                    molwt1 = MolecularFunction.calculate_molecular_weight(smiles1)
                    molwseq1.append(molwt1)
                    
                    aa2 = MolecularFunction.calculate_amino_acid_center_of_mass(b_cell_receptor['seq'][i])
                    smiles2 = MolecularFunction.seq_to_smiles(b_cell_receptor['seq'][i])
                    smilesseq2.append(smiles2)
                    molwt2 = MolecularFunction.calculate_molecular_weight(smiles2)
                    molwseq2.append(molwt2)
                    
                    dist = MolecularFunction.calculate_distance_between_amino_acids(aa1, aa2)
                    bdist.append(dist)
                    feature = [aa1, molwt1, aa2, molwt2, dist]
                    com1.append(aa1)
                    com2.append(aa2)

                    bmol_pred.append(qml_dock(feature, self.sampler))

        b_res = {
            'Ligand': seq1,
            'Receptor': seq2,
            'Receptor id': seq2id,
            'Adjuvant CID': adjuvant_list,
            'Adjuvant IsoSMILES': adjuvant_isosmiles,
            'Ligand Smiles': smilesseq1,
            'Receptor Smiles': smilesseq2,
            'Center Of Mass Ligand': com1,
            'Center Of Mass Receptor': com2,
            'Molecular Weight Of Ligand': molwseq1,
            'Molecular Weight Of Receptor': molwseq2,
            'Distance': bdist,
            'Docking(Ki (nM))': bmol_pred
        }

        b_res_df = pd.DataFrame(b_res)
        print("Quantum Machine Learning Based Scoring Function of B Cell Success With Adjuvant")

        # T-cell receptor docking with adjuvant
        t_cell_receptor = pd.read_csv(os.path.join(get_data_dir(), 't cell receptor homo sapiens.csv'))
        t_cell_receptor = t_cell_receptor.sample(frac=1, random_state=42).reset_index(drop=True)
        t_cell_receptor = t_cell_receptor[0:self.n_receptor]
        t_epitope = self.tseq

        seq1, seq2, seq2id = [], [], []
        adjuvant_list, adjuvant_isosmiles = [], []
        smilesseq1, smilesseq2 = [], []
        molwseq1, molwseq2 = [], []
        bdist, bmol_pred = [], []
        com1, com2 = [], []
        
        for t in t_epitope:
            for i in range(len(t_cell_receptor)):
                for adju in range(len(adjuvant)):
                    adjuvant_list.append(adjuvant['cid'][adju])
                    adjuvant_isosmiles.append(adjuvant['isosmiles'][adju])
                    
                    seq_plus_adjuvant = Chem.MolToSmiles(
                        MolecularFunction.combine_epitope_with_adjuvant(t, adjuvant['isosmiles'][adju]))
                    
                    seq1.append(t)
                    seq2.append(t_cell_receptor['seq'][i])
                    seq2id.append(t_cell_receptor['id'][i])
                    
                    aa1 = MolecularFunction.calculate_amino_acid_center_of_mass_smiles(seq_plus_adjuvant)
                    smiles1 = seq_plus_adjuvant
                    smilesseq1.append(smiles1)
                    molwt1 = MolecularFunction.calculate_molecular_weight(smiles1)
                    molwseq1.append(molwt1)
                    
                    aa2 = MolecularFunction.calculate_amino_acid_center_of_mass(t_cell_receptor['seq'][i])
                    smiles2 = MolecularFunction.seq_to_smiles(t_cell_receptor['seq'][i])
                    smilesseq2.append(smiles2)
                    molwt2 = MolecularFunction.calculate_molecular_weight(smiles2)
                    molwseq2.append(molwt2)
                    
                    dist = MolecularFunction.calculate_distance_between_amino_acids(aa1, aa2)
                    bdist.append(dist)
                    feature = [aa1, molwt1, aa2, molwt2, dist]
                    com1.append(aa1)
                    com2.append(aa2)

                    bmol_pred.append(qml_dock(feature, self.sampler))

        t_res = {
            'Ligand': seq1,
            'Receptor': seq2,
            'Receptor id': seq2id,
            'Adjuvant CID': adjuvant_list,
            'Adjuvant IsoSMILES': adjuvant_isosmiles,
            'Ligand Smiles': smilesseq1,
            'Receptor Smiles': smilesseq2,
            'Center Of Mass Ligand': com1,
            'Center Of Mass Receptor': com2,
            'Molecular Weight Of Ligand': molwseq1,
            'Molecular Weight Of Receptor': molwseq2,
            'Distance': bdist,
            'Docking(Ki (nM))': bmol_pred
        }
        
        t_res_df = pd.DataFrame(t_res)
        print("Quantum Machine Learning Based Scoring Function of T Cell Success With Adjuvant")

        # Save results
        b_res_df.to_excel(self.target_path + '/' + 'qml_b_cell_with_adjuvant.xlsx', index=False)
        t_res_df.to_excel(self.target_path + '/' + 'qml_t_cell_with_adjuvant.xlsx', index=False)

        return b_res, t_res 