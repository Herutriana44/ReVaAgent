"""
Command Line Interface for ReVa AI Vaccine Design Library
"""

import argparse
import sys
import os
from .reva import ReVa
from .qreva import QReVa
from .utils import get_base_path, create_folder, generate_filename_with_timestamp_and_random

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="ReVa AI Vaccine Design - Classical and Quantum ML-based vaccine design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classical vaccine design
  reva --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" --n-receptor 10 --n-adjuvant 5

  # Quantum vaccine design
  reva --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" --n-receptor 10 --n-adjuvant 5 --quantum

  # With BLAST search
  reva --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" --n-receptor 10 --n-adjuvant 5 --blast-activate

  # With LLM review
  reva --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" --n-receptor 10 --n-adjuvant 5 --llm-model-name "microsoft/biogpt"

  # With AlphaFold modeling
  reva --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" --n-receptor 10 --n-adjuvant 5 --alphafold-url "http://localhost:8001/alphafold"
        """
    )

    # Required arguments
    parser.add_argument(
        "--sequence", "-s",
        type=str,
        required=True,
        help="Protein sequence for vaccine design"
    )

    parser.add_argument(
        "--n-receptor", "-r",
        type=int,
        required=True,
        help="Number of receptors to use for docking"
    )

    parser.add_argument(
        "--n-adjuvant", "-a",
        type=int,
        required=True,
        help="Number of adjuvants to use for docking"
    )

    # Optional arguments
    parser.add_argument(
        "--quantum", "-q",
        action="store_true",
        help="Use quantum machine learning models (default: classical)"
    )

    parser.add_argument(
        "--blast-activate",
        action="store_true",
        help="Activate BLAST search for similarity analysis"
    )

    parser.add_argument(
        "--llm-model-name",
        type=str,
        default="microsoft/biogpt",
        help="HuggingFace model name for LLM review (default: microsoft/biogpt)"
    )

    parser.add_argument(
        "--alphafold-url",
        type=str,
        default="",
        help="URL for AlphaFold protein structure modeling service"
    )

    parser.add_argument(
        "--qibm-api",
        type=str,
        default="",
        help="IBM Quantum API token for quantum backend access"
    )

    parser.add_argument(
        "--backend-type",
        type=str,
        default="ibmq_qasm_simulator",
        help="Quantum backend type (default: ibmq_qasm_simulator)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for results (default: auto-generated)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate inputs
    if args.n_receptor <= 0:
        print("Error: n-receptor must be positive")
        sys.exit(1)

    if args.n_adjuvant <= 0:
        print("Error: n-adjuvant must be positive")
        sys.exit(1)

    # Set up output directory
    if args.output_dir:
        target_folder = args.output_dir
    else:
        target_folder = os.path.join("result", "result_" + generate_filename_with_timestamp_and_random())
    
    create_folder(target_folder)

    # Get base path
    base_path = get_base_path()

    try:
        if args.quantum:
            print("Starting Quantum Vaccine Design...")
            qreva = QReVa(
                sequence=args.sequence,
                base_path=base_path,
                target_path=os.path.join(target_folder, generate_filename_with_timestamp_and_random("quantum")),
                n_receptor=args.n_receptor,
                n_adjuvant=args.n_adjuvant,
                blast_activate=args.blast_activate,
                qibm_api=args.qibm_api,
                backend_type=args.backend_type,
                llm_model_name=args.llm_model_name,
                alphafold_url=args.alphafold_url
            )
            
            res1, res2, res3 = qreva.predict()
            print(f"Quantum vaccine design completed. Results saved to: {target_folder}")
            
        else:
            print("Starting Classical Vaccine Design...")
            reva = ReVa(
                sequence=args.sequence,
                base_path=base_path,
                target_path=os.path.join(target_folder, generate_filename_with_timestamp_and_random()),
                n_receptor=args.n_receptor,
                n_adjuvant=args.n_adjuvant,
                blast_activate=args.blast_activate,
                llm_model_name=args.llm_model_name,
                alphafold_url=args.alphafold_url
            )
            
            res1, res2, res3 = reva.predict()
            print(f"Classical vaccine design completed. Results saved to: {target_folder}")

        if args.verbose:
            print(f"Epitope predictions: {len(res1['B']['predictions'])} B-cell, {len(res1['T']['predictions'])} T-cell")
            print(f"Evaluation results: {len(res3['seq']['B'])} B-cell epitopes, {len(res3['seq']['T'])} T-cell epitopes")

    except Exception as e:
        print(f"Error during vaccine design: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 