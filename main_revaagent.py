import os
from reva_lib.reva import ReVa
from reva_lib.qreva import QReVa
from reva_lib.utils import get_base_path, generate_filename_with_timestamp_and_random, create_folder

def run_revaagent(
    sequence,
    n_receptor,
    n_adjuvant,
    blast_activate=False,
    llm_model_name="microsoft/biogpt",
    alphafold_url="",
    qibm_api="",
    backend_type="ibmq_qasm_simulator",
    output_dir="",
    use_model="reva"  # "reva" (klasik) atau "qreva" (quantum)
):
    """
    Fungsi utama untuk menjalankan ReVaAgent (klasik/quantum) secara dinamis.
    """
    base_path = get_base_path()
    if not output_dir:
        output_dir = os.path.join("result", "result_" + generate_filename_with_timestamp_and_random("quantum" if use_model=="qreva" else "classic"))
    create_folder(output_dir)

    if use_model == "qreva":
        print("Menggunakan Quantum Model (QReVa)...")
        agent = QReVa(
            sequence=sequence,
            base_path=base_path,
            target_path=output_dir,
            n_receptor=n_receptor,
            n_adjuvant=n_adjuvant,
            blast_activate=blast_activate,
            qibm_api=qibm_api,
            backend_type=backend_type,
            llm_model_name=llm_model_name,
            alphafold_url=alphafold_url
        )
    else:
        print("Menggunakan Classical Model (ReVa)...")
        agent = ReVa(
            sequence=sequence,
            base_path=base_path,
            target_path=output_dir,
            n_receptor=n_receptor,
            n_adjuvant=n_adjuvant,
            blast_activate=blast_activate,
            llm_model_name=llm_model_name,
            alphafold_url=alphafold_url
        )

    # Jalankan prediksi utama
    result = agent.predict()
    print(f"Hasil disimpan di: {output_dir}")
    return result

# =========================
# CONTOH PEMAKAIAN
# =========================
if __name__ == "__main__":
    # Ganti dengan input Anda
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    n_receptor = 10
    n_adjuvant = 5
    blast_activate = False
    llm_model_name = "microsoft/biogpt"
    alphafold_url = ""
    qibm_api = ""
    backend_type = "ibmq_qasm_simulator"
    output_dir = ""  # atau path hasil custom
    use_model = "reva"  # atau "qreva"

    run_revaagent(
        sequence=sequence,
        n_receptor=n_receptor,
        n_adjuvant=n_adjuvant,
        blast_activate=blast_activate,
        llm_model_name=llm_model_name,
        alphafold_url=alphafold_url,
        qibm_api=qibm_api,
        backend_type=backend_type,
        output_dir=output_dir,
        use_model=use_model
    )