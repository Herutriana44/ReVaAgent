"""
Molecular utilities for ReVa AI Vaccine Design Library
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolTransforms, rdMolDescriptors
from scipy.constants import e, epsilon_0, Boltzmann
import os

class MolecularFunction:
    """Class containing molecular calculation functions"""
    
    @staticmethod
    def calculate_amino_acid_center_of_mass(sequence):
        """Calculate center of mass for amino acid sequence"""
        try:
            amino_acid_masses = []
            for aa in sequence:
                try:
                    amino_acid_masses.append(Chem.Descriptors.MolWt(Chem.MolFromSequence(aa)))
                except:
                    return 0
                    break

            total_mass = sum(amino_acid_masses)
            center_of_mass = sum(i * mass for i, mass in enumerate(amino_acid_masses, start=1)) / total_mass
            return center_of_mass
        except:
            return 0

    @staticmethod
    def calculate_amino_acid_center_of_mass_smiles(sequence):
        """Calculate center of mass for SMILES sequence"""
        try:
            amino_acid_masses = []
            for aa in sequence:
                amino_acid_masses.append(Chem.Descriptors.MolWt(Chem.MolFromSmiles(aa)))

            total_mass = sum(amino_acid_masses)
            center_of_mass = sum(i * mass for i, mass in enumerate(amino_acid_masses, start=1)) / total_mass
            return center_of_mass
        except:
            return 0

    @staticmethod
    def calculate_distance_between_amino_acids(aa1, aa2):
        """Calculate distance between two amino acid centers of mass"""
        distance = abs(aa1 - aa2)
        return distance

    @staticmethod
    def generate_conformer(molecule_smiles):
        """Generate molecular conformer"""
        mol = Chem.MolFromSmiles(molecule_smiles)
        mol = Chem.AddHs(mol)
        conformer = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        return mol

    @staticmethod
    def calculate_molecular_center_of_mass(smiles):
        """Calculate molecular center of mass"""
        molecule = MolecularFunction.generate_conformer(smiles)
        try:
            if molecule is None:
                return None

            center_of_mass = rdMolTransforms.ComputeCentroid(molecule.GetConformer())
            total_mass = Descriptors.MolWt(molecule)
            center_of_mass = sum([center_of_mass[i] * total_mass for i in range(len(center_of_mass))]) / total_mass
            
            return center_of_mass
        except Exception as e:
            print("Error:", str(e))
            return None

    @staticmethod
    def seq_to_smiles(seq):
        """Convert sequence to SMILES"""
        try:
            mol = Chem.MolFromSequence(seq)
            smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
            return str(smiles)
        except:
            return None
        
    @staticmethod
    def combine_epitope_with_adjuvant(epitope_sequence, adjuvant_smiles):
        """Combine epitope with adjuvant"""
        epitope_molecule = MolecularFunction.MolFromLongSequence(epitope_sequence)
        adjuvant_molecule = MolecularFunction.MolFromLongSequence(adjuvant_smiles)
        combined_molecule = Chem.CombineMols(adjuvant_molecule, epitope_molecule)
        return combined_molecule
    
    @staticmethod
    def calculate_molecular_weight(molecule_smiles):
        """Calculate molecular weight"""
        try:
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol is None:
                print("Failed to read molecule.")
                return None

            molecular_weight = Descriptors.MolWt(mol)
            return molecular_weight
        except:
            return None
        
    @staticmethod
    def MolFromLongSequence(sequence, chunk_size=100):
        """Convert long sequence to Mol object by splitting into chunks"""
        chunks = [sequence[i:i+chunk_size] for i in range(0, len(sequence), chunk_size)]
        mols = [Chem.MolFromSequence(chunk) for chunk in chunks if chunk]
        
        combined_mol = Chem.Mol()
        for mol in mols:
            if mol:
                combined_mol = Chem.CombineMols(combined_mol, mol)
        
        return combined_mol

class MolecularScoring:
    """Class containing molecular scoring functions"""
    
    def __init__(self):
        self.mass_of_argon = 39.948

    @staticmethod
    def attractive_energy(r, epsilon=0.0103, sigma=3.4):
        """Attractive component of Lennard-Jones interaction energy"""
        if r == 0:
            return 0
        return -4.0 * epsilon * np.power(sigma / r, 6)

    @staticmethod
    def repulsive_energy(r, epsilon=0.0103, sigma=3.4):
        """Repulsive component of Lennard-Jones interaction energy"""
        if r == 0:
            return 0
        return 4 * epsilon * np.power(sigma / r, 12)
    
    @staticmethod
    def lj_energy(r, epsilon=0.0103, sigma=3.4):
        """Lennard-Jones potential energy"""
        if r == 0:
            return 0
        
        return (MolecularScoring.repulsive_energy(r, epsilon, sigma) + 
                MolecularScoring.attractive_energy(r, epsilon, sigma))
    
    @staticmethod
    def coulomb_energy(qi, qj, r):
        """Coulomb's law calculation"""
        if r == 0:
            return 0
        
        energy_joules = (qi * qj * e ** 2) / (4 * np.pi * epsilon_0 * r * 1e-10)
        return energy_joules / 1.602e-19
    
    @staticmethod
    def bonded(kb, b0, b):
        """Potential energy of a bond"""
        return kb / 2 * (b - b0) ** 2
    
    @staticmethod
    def lj_force(r, epsilon=0.0103, sigma=3.4):
        """Lennard-Jones force calculation"""
        if r != 0:
            return (48 * epsilon * np.power(sigma, 12) / np.power(r, 13) - 
                   24 * epsilon * np.power(sigma, 6) / np.power(r, 7))
        else:
            return 0

    @staticmethod
    def init_velocity(T, number_of_particles):
        """Initialize velocities for particles"""
        R = np.random.rand(number_of_particles) - 0.5
        return R * np.sqrt(Boltzmann * T / (MolecularScoring().mass_of_argon * 1.602e-19))

    @staticmethod
    def get_accelerations(positions):
        """Calculate accelerations on particles"""
        accel_x = np.zeros((positions.size, positions.size))
        for i in range(0, positions.size - 1):
            for j in range(i + 1, positions.size):
                r_x = positions[j] - positions[i]
                rmag = np.sqrt(r_x * r_x)
                force_scalar = MolecularScoring.lj_force(rmag, 0.0103, 3.4)
                force_x = force_scalar * r_x / rmag
                accel_x[i, j] = force_x / MolecularScoring().mass_of_argon
                accel_x[j, i] = - force_x / MolecularScoring().mass_of_argon
        return np.sum(accel_x, axis=0)

    @staticmethod
    def update_pos(x, v, a, dt):
        """Update particle positions"""
        return x + v * dt + 0.5 * a * dt * dt
    
    @staticmethod
    def update_velo(v, a, a1, dt):
        """Update particle velocities"""
        return v + 0.5 * (a + a1) * dt

    @staticmethod
    def run_md(dt, number_of_steps, initial_temp, x):
        """Run molecular dynamics simulation"""
        positions = np.zeros((number_of_steps, 3))
        v = MolecularScoring.init_velocity(initial_temp, 3)
        a = MolecularScoring.get_accelerations(x)
        for i in range(number_of_steps):
            x = MolecularScoring.update_pos(x, v, a, dt)
            a1 = MolecularScoring.get_accelerations(x)
            v = MolecularScoring.update_velo(v, a, a1, dt)
            a = np.array(a1)
            positions[i, :] = x
        return positions

    @staticmethod
    def lj_force_cutoff(r, epsilon, sigma):
        """Lennard-Jones force with cutoff"""
        cutoff = 15 
        if r < cutoff:
            return (48 * epsilon * np.power(sigma / r, 13) - 
                   24 * epsilon * np.power(sigma / r, 7))
        else:
            return 0 