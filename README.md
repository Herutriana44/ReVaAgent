# ReVa AI Vaccine Design Library

A comprehensive Python library for vaccine design using classical and quantum machine learning approaches. ReVa provides state-of-the-art tools for epitope prediction, molecular docking, and vaccine candidate evaluation.

## Features

### 🧬 Epitope Prediction
- **B-cell epitope prediction** using machine learning models
- **T-cell epitope prediction** for MHC binding
- **Quantum-enhanced epitope prediction** using VQC models

### 🔬 Protein Analysis
- **Allergenicity prediction** for safety assessment
- **Toxicity prediction** for vaccine safety
- **Antigenicity prediction** for immunogenicity
- **Physicochemical property analysis** (hydrophobicity, pI, molecular weight, etc.)

### 🧪 Molecular Docking
- **Classical force field docking** with van der Waals and Coulomb interactions
- **Machine learning-based docking** scoring functions
- **Quantum machine learning docking** for enhanced accuracy
- **Adjuvant integration** for improved immunogenicity

### 🌐 External Services Integration
- **BLAST search** for sequence similarity analysis
- **LLM review** for result interpretation
- **AlphaFold modeling** for 3D structure prediction

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Basic Installation
```bash
pip install reva-ai-vaccine-design
```

### Installation with Quantum Support
```bash
pip install reva-ai-vaccine-design[quantum]
```

### Development Installation
```bash
git clone https://github.com/reva-ai/vaccine-design.git
cd vaccine-design
pip install -e .
```

## Quick Start

### Using the Library in Python

```python
from reva_lib import ReVa, QReVa

# Classical vaccine design
reva = ReVa(
    sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    base_path="./",
    target_path="./results",
    n_receptor=10,
    n_adjuvant=5,
    blast_activate=True
)

# Run predictions
epitope_results, epitope_processed, evaluation_results = reva.predict()

# Quantum vaccine design
qreva = QReVa(
    sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    base_path="./",
    target_path="./results_quantum",
    n_receptor=10,
    n_adjuvant=5,
    blast_activate=True,
    qibm_api="your_ibm_quantum_token"
)

# Run quantum predictions
q_epitope_results, q_epitope_processed, q_evaluation_results = qreva.predict()
```

### Using the Command Line Interface

```bash
# Classical vaccine design
reva --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" \
     --n-receptor 10 \
     --n-adjuvant 5 \
     --blast-activate

# Quantum vaccine design
reva --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" \
     --n-receptor 10 \
     --n-adjuvant 5 \
     --quantum \
     --qibm-api "your_ibm_quantum_token"

# With external services
reva --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" \
     --n-receptor 10 \
     --n-adjuvant 5 \
     --llm-url "http://localhost:8000/review" \
     --alphafold-url "http://localhost:8001/alphafold"
```

## API Reference

### ReVa Class (Classical)

```python
class ReVa:
    def __init__(self, sequence, base_path, target_path, n_receptor, n_adjuvant, 
                 blast_activate=False, llm_url="", alphafold_url=""):
        """
        Initialize ReVa for classical vaccine design.
        
        Args:
            sequence (str): Protein sequence
            base_path (str): Base path for models and data
            target_path (str): Target path for results
            n_receptor (int): Number of receptors to use
            n_adjuvant (int): Number of adjuvants to use
            blast_activate (bool): Whether to activate BLAST search
            llm_url (str): URL for LLM service
            alphafold_url (str): URL for AlphaFold service
        """
    
    def predict(self):
        """
        Run complete vaccine design pipeline.
        
        Returns:
            tuple: (epitope_results, epitope_processed, evaluation_results)
        """
```

### QReVa Class (Quantum)

```python
class QReVa:
    def __init__(self, sequence, base_path, target_path, n_receptor, n_adjuvant, 
                 blast_activate=False, qibm_api="", backend_type="ibmq_qasm_simulator", 
                 llm_url="", alphafold_url=""):
        """
        Initialize QReVa for quantum vaccine design.
        
        Args:
            sequence (str): Protein sequence
            base_path (str): Base path for models and data
            target_path (str): Target path for results
            n_receptor (int): Number of receptors to use
            n_adjuvant (int): Number of adjuvants to use
            blast_activate (bool): Whether to activate BLAST search
            qibm_api (str): IBM Quantum API token
            backend_type (str): Quantum backend type
            llm_url (str): URL for LLM service
            alphafold_url (str): URL for AlphaFold service
        """
    
    def predict(self):
        """
        Run complete quantum vaccine design pipeline.
        
        Returns:
            tuple: (epitope_results, epitope_processed, evaluation_results)
        """
```

## Output Files

The library generates several output files:

### CSV Files
- `B_result.csv` - B-cell epitope evaluation results
- `T_result.csv` - T-cell epitope evaluation results

### Excel Files
- `forcefield_b_cell.xlsx` - Classical docking results for B-cells
- `forcefield_t_cell.xlsx` - Classical docking results for T-cells
- `ml_scoring_func_b_cell.xlsx` - ML docking results for B-cells
- `ml_scoring_func_t_cell.xlsx` - ML docking results for T-cells
- `qml_scoring_func_b_cell.xlsx` - Quantum ML docking results for B-cells
- `qml_scoring_func_t_cell.xlsx` - Quantum ML docking results for T-cells

### JSON Files
- `*_eval_res.json` - Complete evaluation results
- `*_eval_res_quantum.json` - Complete quantum evaluation results

### Text Files
- `B_res_review.txt` - LLM review of B-cell results
- `T_res_review.txt` - LLM review of T-cell results

### AlphaFold Results
- `Alphafold Modelling Result/` - 3D protein structure models

## Dependencies

### Core Dependencies
- numpy>=1.21.0
- pandas>=1.3.0
- tensorflow>=2.8.0
- scikit-learn>=1.0.0
- biopython>=1.79
- rdkit>=2021.9.1
- scipy>=1.7.0
- requests>=2.25.0

### Quantum Dependencies (Optional)
- qiskit>=0.40.0
- qiskit-machine-learning>=0.5.0

### Development Dependencies
- pytest>=6.0
- pytest-cov>=2.0
- black>=21.0
- flake8>=3.8
- mypy>=0.800

## Project Structure

```
reva_lib/
├── __init__.py              # Main library exports
├── reva.py                  # Classical ReVa class
├── qreva.py                 # Quantum QReVa class
├── utils.py                 # Utility functions
├── molecular_utils.py       # Molecular calculations
├── protein_analysis.py      # Protein property analysis
├── docking.py              # Docking algorithms
└── cli.py                  # Command-line interface

asset/
├── data/                   # Training and reference data
├── model/                  # Trained models
│   └── Quantum Model/      # Quantum models
├── label/                  # Label mappings
└── json/                   # Configuration files
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/reva-ai/vaccine-design.git
cd vaccine-design
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black reva_lib/
flake8 reva_lib/
```

## Citation

If you use ReVa in your research, please cite:

```bibtex
@article{reva2024,
  title={ReVa: A Comprehensive Library for Vaccine Design Using Classical and Quantum Machine Learning},
  author={ReVa AI Team},
  journal={Bioinformatics},
  year={2024},
  doi={10.1093/bioinformatics/btad123}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://reva-ai.github.io/vaccine-design/](https://reva-ai.github.io/vaccine-design/)
- **Issues**: [https://github.com/reva-ai/vaccine-design/issues](https://github.com/reva-ai/vaccine-design/issues)
- **Discussions**: [https://github.com/reva-ai/vaccine-design/discussions](https://github.com/reva-ai/vaccine-design/discussions)

## Acknowledgments

- BioPython team for bioinformatics tools
- RDKit team for cheminformatics capabilities
- Qiskit team for quantum computing framework
- TensorFlow team for machine learning framework

## Roadmap

- [ ] GPU acceleration for molecular docking
- [ ] Integration with more quantum backends
- [ ] Web-based interface
- [ ] Additional epitope prediction algorithms
- [ ] Real-time collaboration features
- [ ] Cloud deployment options 