# ComicsPAP-Qwen2.5-VL: Sequential Narrative Understanding

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/kaj04/Qwen2.5-VL-7B-ComicsPAP-QLoRA)
[![WandB Training Logs](https://img.shields.io/badge/Weights%20%26%20Biases-Logs-orange)](TUO_LINK_WANDB)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Fine-tuning of **Qwen2.5-VL-7B** on the **ComicsPAP** benchmark for the "Pick A Panel" task. 
Achieved **66.41% Validation Accuracy**, significantly outperforming zero-shot baselines.

## Repository Structure
- `src/train_qlora.py`: Training script using PEFT and QLoRA.
- `src/evaluate.py`: Evaluation pipeline on Val/Test splits.
- `official_repo/`: Integrated utilities from the official ComicsPAP repository.
  - `data_utils.py`: Contains the `SingleImagePickAPanel` processor used for training.
- `slurm/`: Job scripts for the Snellius Supercomputer.

## Project Highlights
- **Model:** Qwen2.5-VL-7B-Instruct (Vision-Language).
- **Technique:** QLoRA (Rank 16, Alpha 32) for parameter-efficient fine-tuning.
- **Compute:** Trained on 1x NVIDIA A100 (80GB) on the **Snellius Supercomputer** (SURF/TU/e).
- **Performance:**
  - **Ours (7B Fine-tuned): 66.41%**
  - Base 72B Zero-shot: 46.88% (from paper)
  - Base 7B Zero-shot: 30.53% (from paper)
 
## Getting Started
### 1. Installation
```bash
git clone [https://github.com/TUO_USERNAME/NOME_REPO.git](https://github.com/TUO_USERNAME/NOME_REPO.git)
cd NOME_REPO
pip install -r requirements.txt

### 2. Run Evaluation
You can run the evaluation using my pre-trained adapters from Hugging Face:
python src/evaluate.py --adapter_path kaj04/Qwen2.5-VL-7B-ComicsPAP-QLoRA --split val


```markdown
## Acknowledgments
This project is based on the **ComicsPAP** dataset and benchmark. Special thanks to the authors for their work in visual narrative understanding. 

- **Official Dataset:** [VLR-CVC/ComicsPAP](https://huggingface.co/datasets/VLR-CVC/ComicsPAP)
- **Official Paper:** [ComicsPAP: Understanding Comic Strips by Picking the Correct Panel](https://arxiv.org/abs/...)

### Citation
```bibtex
@InProceedings{vivoli2025comicspap,
  author="Vivoli, Emanuele and Llabr{\'e}s, Artemis and Souibgui, Mohamed Ali and Bertini, Marco and Llobet, Ernest Valveny and Karatzas, Dimosthenis",
  editor="Yin, Xu-Cheng and Karatzas, Dimosthenis and Lopresti, Daniel",
  title="ComicsPAP: Understanding Comic Strips by Picking the Correct Panel",
  booktitle="Document Analysis and Recognition -- ICDAR 2025",
  year="2026",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="337--350",
  isbn="978-3-032-04614-7"
}

```
