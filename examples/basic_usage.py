#!/usr/bin/env python3
"""
Basic usage example for ReVa AI Vaccine Design Library

This example demonstrates how to use both classical and quantum approaches
for vaccine design.
"""

import os
import sys

# Add the parent directory to the path to import the library
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reva_lib import ReVa, QReVa

def main():
    """Main function demonstrating basic usage"""
    
    # Example protein sequence (SARS-CoV-2 spike protein fragment)
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    # Configuration
    base_path = os.path.dirname(os.path.abspath(__file__))
    n_receptor = 5  # Number of receptors to use
    n_adjuvant = 3  # Number of adjuvants to use
    
    print("=== ReVa AI Vaccine Design Library - Basic Usage Example ===\n")
    
    # Classical vaccine design
    print("1. Running Classical Vaccine Design...")
    try:
        reva = ReVa(
            sequence=sequence,
            base_path=base_path,
            target_path="./results_classical",
            n_receptor=n_receptor,
            n_adjuvant=n_adjuvant,
            blast_activate=False  # Set to True if you want BLAST search
        )
        
        # Run predictions
        epitope_results, epitope_processed, evaluation_results = reva.predict()
        
        print("✅ Classical vaccine design completed successfully!")
        print(f"   - B-cell epitopes found: {len(evaluation_results['seq']['B'])}")
        print(f"   - T-cell epitopes found: {len(evaluation_results['seq']['T'])}")
        
    except Exception as e:
        print(f"❌ Error in classical vaccine design: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Quantum vaccine design
    print("2. Running Quantum Vaccine Design...")
    try:
        qreva = QReVa(
            sequence=sequence,
            base_path=base_path,
            target_path="./results_quantum",
            n_receptor=n_receptor,
            n_adjuvant=n_adjuvant,
            blast_activate=False,  # Set to True if you want BLAST search
            qibm_api="",  # Add your IBM Quantum API token here
            backend_type="ibmq_qasm_simulator"
        )
        
        # Run quantum predictions
        q_epitope_results, q_epitope_processed, q_evaluation_results = qreva.predict()
        
        print("✅ Quantum vaccine design completed successfully!")
        print(f"   - B-cell epitopes found: {len(q_evaluation_results['seq']['B'])}")
        print(f"   - T-cell epitopes found: {len(q_evaluation_results['seq']['T'])}")
        
    except Exception as e:
        print(f"❌ Error in quantum vaccine design: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Display some results
    print("3. Sample Results:")
    if 'evaluation_results' in locals():
        print("   Classical Results:")
        print(f"   - Allergenicity (B-cell): {evaluation_results['allergenicity']['B'][:3]}...")
        print(f"   - Toxicity (B-cell): {evaluation_results['toxin']['B'][:3]}...")
        print(f"   - Antigenicity (B-cell): {evaluation_results['antigenicity']['B'][:3]}...")
    
    if 'q_evaluation_results' in locals():
        print("   Quantum Results:")
        print(f"   - Allergenicity (B-cell): {q_evaluation_results['allergenicity']['B'][:3]}...")
        print(f"   - Toxicity (B-cell): {q_evaluation_results['toxin']['B'][:3]}...")
        print(f"   - Antigenicity (B-cell): {q_evaluation_results['antigenicity']['B'][:3]}...")
    
    print("\n" + "="*60 + "\n")
    print("🎉 Example completed! Check the results directories for detailed outputs.")
    print("   - Classical results: ./results_classical/")
    print("   - Quantum results: ./results_quantum/")

if __name__ == "__main__":
    main() 