# Narasi: An NLP-Based Binary Clickbait Detection Model for Indonesian News Headlines

This repository contains the source code, experimental notebooks, evaluation outputs, and application files for **Narasi**, an Indonesian news headline clickbait detection system.

The project compares classical machine learning models with an IndoBERT-based classifier. The final application uses **IndoBERT Balanced with Augmentation** and integrates **Local Interpretable Model-agnostic Explanations (LIME)** to show the words that influence each prediction.

Narasi was developed as a final project for the Natural Language Processing course at Bina Nusantara University.

## Project Overview

Clickbait headlines often attract readers through curiosity gaps, sensational wording, emotional framing, or withheld information.

Narasi accepts an Indonesian news headline and classifies it into one of two categories:

- Non-Clickbait
- Clickbait

The project evaluates three classical machine learning models:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Calibrated LinearSVC

It also evaluates four IndoBERT training strategies:

1. Imbalanced
2. Balanced
3. Imbalanced with Augmentation
4. Balanced with Augmentation

The final system consists of an IndoBERT model hosted on Hugging Face, a FastAPI backend for inference and LIME explanations, and a custom frontend deployed on Vercel.

## Project Links

| Resource             | Link                                                                                                 |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| Live Web Application | [Narasi on Vercel](https://narasi-clickbait-frontend.vercel.app)                                     |
| Hugging Face Space   | [Narasi Clickbait Application](https://huggingface.co/spaces/littt24/narasi-clickbait-app)           |
| Hugging Face Model   | [IndoBERT Balanced with Augmentation](https://huggingface.co/littt24/indobert_balance_augment)       |
| CLICK-ID Dataset     | [Mendeley Data](https://data.mendeley.com/datasets/k42j7x2kpn/1)                                     |
| Project Artifacts    | [Google Drive](https://drive.google.com/drive/folders/18OUhq2FwGWABxkj4Gpi2EM5_BTZj0xYC?usp=sharing) |

The Google Drive folder contains the project source code, PowerPoint presentation, and demonstration materials.

## Repository Structure

```text
narasi-clickbait-detection/
├── backend/
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── assets/
│   │   ├── aman.png
│   │   ├── awas.png
│   │   └── logo.png
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── tentang.html
│
├── notebook/
│   ├── 01_classical_models.ipynb
│   ├── 02_indoBERT_final.ipynb
│   └── 03_compare_results.ipynb
│
├── outputs/
│   └── classical_models_all_agree/
│       └── classical_results.json
│
├── outputs_indobert/
│   └── final_results/
│       ├── comparison_bar_chart.png
│       ├── comparison_results.json
│       ├── comparison_summary_table.csv
│       ├── confusion_matrices.png
│       ├── indobert_all_experiments.csv
│       ├── indobert_balance_augment_report.txt
│       ├── indobert_balance_report.txt
│       ├── indobert_confusion_matrices_all.png
│       ├── indobert_confusion_matrices_from_checkpoints.csv
│       ├── indobert_confusion_matrices_from_checkpoints.json
│       ├── indobert_imbalance_augment_report.txt
│       ├── indobert_imbalance_report.txt
│       └── training_curve_indobert_balance.png
│
├── .gitignore
├── .vercelignore
├── Dockerfile
├── README.md
└── vercel.json
```

## Project Artifacts

| Artifact                  | Location                             | Description                                                                          |
| ------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------ |
| Classical Models Notebook | `notebook/01_classical_models.ipynb` | Classical preprocessing, feature extraction, model training, and evaluation          |
| IndoBERT Notebook         | `notebook/02_indoBERT_final.ipynb`   | IndoBERT preprocessing, four training strategies, evaluation, and final model export |
| Comparison Notebook       | `notebook/03_compare_results.ipynb`  | Comparison of classical and IndoBERT results                                         |
| Backend                   | `backend/app.py`                     | FastAPI inference service and LIME explanation logic                                 |
| Frontend                  | `frontend/`                          | HTML, CSS, JavaScript, and application assets                                        |
| Classical Results         | `outputs/`                           | Saved evaluation results for the classical models                                    |
| IndoBERT Results          | `outputs_indobert/final_results/`    | Experiment summaries, reports, figures, and confusion matrices                       |
| Model Artifacts           | Hugging Face Model Repository        | Final model weights, tokenizer, model configuration, and runtime configuration       |
| Presentation and Demo     | Google Drive                         | PowerPoint, demonstration, and additional project files                              |

Large model weights, temporary checkpoints, and the raw dataset are not duplicated in this GitHub repository.

## Dataset

This project uses **CLICK-ID: A Novel Dataset for Indonesian Clickbait Headlines**, created by Andika William and Yunita Sari.

The complete dataset is available through Mendeley Data:

[CLICK-ID Dataset](https://data.mendeley.com/datasets/k42j7x2kpn/1)

Dataset DOI:

```text
10.17632/k42j7x2kpn.1
```

CLICK-ID contains Indonesian news headlines collected from twelve local online news publishers. The complete annotated dataset contains 15,000 headlines evaluated by three annotators.

This project specifically uses the `all_agree.csv` subset, containing 8,613 headlines for which all annotators agreed on the assigned label.

| Class         | Number of Headlines | Percentage |
| ------------- | ------------------: | ---------: |
| Non-Clickbait |               5,297 |      61.5% |
| Clickbait     |               3,316 |      38.5% |
| Total         |               8,613 |       100% |

The label mapping used in this project is:

```text
0 = Non-Clickbait
1 = Clickbait
```

The dataset is not included directly in this repository. Download the dataset from Mendeley Data and adjust the dataset path in the notebooks before execution.

## Method Summary

### Classical Machine Learning

The classical pipeline uses word-level TF-IDF with unigram and bigram features.

Six additional headline features are included:

- Character length
- Word count
- Presence of an exclamation mark
- Presence of a question mark
- Number of capitalized words
- Presence of clickbait trigger words

The evaluated models are:

- Multinomial Naive Bayes
- Logistic Regression
- Calibrated LinearSVC

Hyperparameters are selected using validation macro F1-score.

### IndoBERT

The transformer experiments use:

```text
indobenchmark/indobert-base-p1
```

Four training strategies are evaluated:

| Strategy                     | Training Samples |
| ---------------------------- | ---------------: |
| Imbalanced                   |            6,889 |
| Balanced                     |            8,474 |
| Imbalanced with Augmentation |           13,778 |
| Balanced with Augmentation   |           16,948 |

Balancing is performed by oversampling the minority clickbait class.

Augmentation randomly applies either:

- Word swapping
- Word deletion

Balancing and augmentation are applied only to the training set. The validation and test sets remain unchanged.

### Fine-Tuning Configuration

| Parameter                   | Value               |
| --------------------------- | ------------------- |
| Maximum sequence length     | 96                  |
| Training batch size         | 8                   |
| Evaluation batch size       | 16                  |
| Gradient accumulation steps | 2                   |
| Effective batch size        | 16                  |
| Learning rate               | `2e-5`              |
| Weight decay                | 0.01                |
| Warm-up ratio               | 0.10                |
| Maximum epochs              | 10                  |
| Early-stopping patience     | 3                   |
| Random seed                 | 42                  |
| Checkpoint selection metric | Validation macro F1 |

The best checkpoint within each experiment is selected using validation macro F1-score.

The completed IndoBERT strategies are compared using held-out test macro F1-score. The highest-performing strategy is exported as the final Narasi model.

## Results

The strongest classical model was **Calibrated LinearSVC**, achieving:

- Accuracy: **0.9116**
- Macro F1-score: **0.9046**

Among the IndoBERT experiments, **IndoBERT Balanced** achieved the highest validation macro F1-score:

- Validation macro F1-score: **0.9570**

The best held-out test performance was achieved by **IndoBERT Balanced with Augmentation**:

- Accuracy: **0.9466**
- Macro F1-score: **0.9433**
- Total test errors: **46**

IndoBERT Balanced with Augmentation was selected as the final Narasi model and uploaded to the Hugging Face Model Repository.

Detailed metrics, classification reports, confusion matrices, and visualizations are available in:

```text
outputs_indobert/final_results/
```

## Explainability with LIME

Narasi integrates **Local Interpretable Model-agnostic Explanations** in the FastAPI backend.

For each prediction, LIME creates modified versions of the input headline and observes how the model probabilities change. It then assigns local contribution weights to the most influential words.

The frontend displays whether each word:

- Supports the predicted class
- Opposes the predicted class

LIME explanations are local and perturbation-based. Small differences in word weights may occur between requests even when the predicted class and confidence remain unchanged.

## System Implementation

Narasi consists of three main components:

1. **IndoBERT Model**

   The final Balanced with Augmentation model is stored in the Hugging Face Model Repository.

2. **FastAPI Backend**

   The backend performs text preprocessing, model inference, confidence calculation, and LIME explanation generation.

3. **Frontend Interface**

   The interface is built using HTML, CSS, and JavaScript. The standalone frontend is deployed through Vercel.

The application flow is:

```text
User Headline
      |
      v
Vercel Frontend
      |
      v
FastAPI Backend on Hugging Face Space
      |
      ├── Text preprocessing
      ├── IndoBERT prediction
      ├── Confidence calculation
      └── LIME explanation
      |
      v
Prediction displayed to the user
```

## Requirements

Install the backend dependencies using:

```bash
pip install -r backend/requirements.txt
```

The main libraries include:

- FastAPI
- Uvicorn
- PyTorch
- Hugging Face Transformers
- LIME
- scikit-learn
- pandas
- NumPy
- nlpaug
- Matplotlib

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/erlitna24/narasi-clickbait-detection.git
cd narasi-clickbait-detection
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install the Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Run the Application

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 7860
```

Open the application at:

```text
http://localhost:7860
```

Interactive FastAPI documentation is available at:

```text
http://localhost:7860/docs
```

The backend downloads the final model from Hugging Face when it is first started. An internet connection is required unless the model is already available locally.

## Run with Docker

Build the image:

```bash
docker build -t narasi-clickbait .
```

Run the container:

```bash
docker run -p 7860:7860 narasi-clickbait
```

Open:

```text
http://localhost:7860
```

## Running the Notebooks

Run the notebooks in the following order:

1. `notebook/01_classical_models.ipynb`
2. `notebook/02_indoBERT_final.ipynb`
3. `notebook/03_compare_results.ipynb`

The first notebook trains and evaluates the classical machine learning models.

The second notebook fine-tunes the four IndoBERT training strategies and exports the final model.

The third notebook loads the saved results and creates the final model comparison and visualizations.

Dataset and output paths may need to be adjusted according to the local environment or Google Drive configuration.

## Deployment

### Hugging Face Model

The final model, tokenizer, and runtime configuration are available at:

[IndoBERT Balanced with Augmentation](https://huggingface.co/littt24/indobert_balance_augment)

### Hugging Face Space

The FastAPI backend and integrated application are deployed at:

[Narasi Clickbait Application](https://huggingface.co/spaces/littt24/narasi-clickbait-app)

### Vercel

The standalone frontend is deployed at:

[Narasi Web Application](https://narasi-clickbait-frontend.vercel.app)

### Google Drive

Additional project artifacts are available at:

[Natural Language Processing Project Folder](https://drive.google.com/drive/folders/18OUhq2FwGWABxkj4Gpi2EM5_BTZj0xYC?usp=sharing)

The folder contains:

- Source Code
- PowerPoint
- Demonstration

## Limitations

The project has several limitations:

- The experiments use only the CLICK-ID `all_agree.csv` subset.
- The classical and IndoBERT pipelines use different preprocessing procedures and slightly different test samples.
- Only one random seed is used.
- The IndoBERT pipeline verifies row-level split separation but does not completely rule out duplicate or near-duplicate headline text across splits.
- Test macro F1-score is used to determine which IndoBERT strategy is exported.
- No statistical significance testing is performed.
- LIME provides local approximations rather than a complete representation of the model's internal reasoning.
- The system analyzes only the headline and does not compare it with the full article content.

## Academic Disclaimer

Narasi was developed for academic research and educational demonstration.

The prediction and confidence score should not be interpreted as an absolute judgment of journalistic quality. Ambiguous headlines may require human evaluation, and the system should be used as an assistive tool rather than a definitive decision-maker.

## Authors

- **Erlitna Natasya Debora**

- **Collyne**

## Supervisor

- **Mohammad Faisal Riftiarrasyid**
