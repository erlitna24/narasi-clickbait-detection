# Narasi: Indonesian Clickbait Detection with IndoBERT and LIME

Narasi is an NLP-based system for classifying Indonesian news headlines as **Clickbait** or **Non-Clickbait**. The project compares classical machine learning baselines with four IndoBERT training strategies and adds word-level explanations using **Local Interpretable Model-agnostic Explanations (LIME)**.

This project was developed for the Natural Language Processing course at Bina Nusantara University.

## Project Links

| Resource             | Link                                                                                                 |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| Live Web Application | [Narasi on Vercel](https://narasi-clickbait-frontend.vercel.app)                                     |
| Hugging Face Space   | [Narasi Clickbait Application](https://huggingface.co/spaces/littt24/narasi-clickbait-app)           |
| Hugging Face Model   | [IndoBERT Balanced with Augmentation](https://huggingface.co/littt24/indobert_balance_augment)       |
| CLICK-ID Dataset     | [Mendeley Data](https://data.mendeley.com/datasets/k42j7x2kpn/1)                                     |
| Project Artifacts    | [Google Drive](https://drive.google.com/drive/folders/18OUhq2FwGWABxkj4Gpi2EM5_BTZj0xYC?usp=sharing) |

The Google Drive folder contains the source code, PowerPoint presentation, and demonstration materials.

## Repository Structure

```text
narasi-clickbait-detection/
├── backend/
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── assets/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── tentang.html
│
├── notebook/
│   ├── 01_classical_models.ipynb
│   ├── 02_indoBERT_final.ipynb
│   └── 03_compare_modelresults.ipynb
│
├── outputs/
│   └── classical_models_all_agree/
│       └── classical_results.json
│
├── outputs_compare_modelresults/
│   ├── confusion_matrices/
│   ├── all_model_metrics.png
│   ├── comparison_results.json
│   ├── comparison_summary_table.csv
│   ├── macro_f1_ranking.png
│   ├── validation_f1_indobert_balance.png
│   └── validation_loss_indobert_balance.png
│
├── .gitignore
├── .vercelignore
├── Dockerfile
├── README.md
└── vercel.json
```

Raw datasets, local environment files, temporary checkpoints, serialized classical models, and large model weights are excluded from this repository.

## Dataset

This project uses **CLICK-ID: A Novel Dataset for Indonesian Clickbait Headlines** by Andika William and Yunita Sari.

- Dataset: [Mendeley Data](https://data.mendeley.com/datasets/k42j7x2kpn/1)
- DOI: `10.17632/k42j7x2kpn.1`
- Subset used: `all_agree.csv`
- Total headlines: **8,613**
- Non-Clickbait: **5,297**
- Clickbait: **3,316**

Label mapping:

```text
0 = Non-Clickbait
1 = Clickbait
```

The raw dataset is not included in this repository. Download it from Mendeley Data and adjust the dataset path in the notebooks before execution.

## Method Summary

### Classical Machine Learning

The classical pipeline evaluates:

- Multinomial Naive Bayes
- Logistic Regression
- Calibrated LinearSVC

The models use word-level TF-IDF with unigram and bigram features, together with six handcrafted headline features. Hyperparameters are selected using validation macro F1-score.

### IndoBERT

The transformer experiments use:

```text
indobenchmark/indobert-base-p1
```

Four training strategies are evaluated:

- Imbalanced
- Balanced
- Imbalanced with Augmentation
- Balanced with Augmentation

Balancing and augmentation are applied only to the training data. The validation and test sets remain unchanged. The best checkpoint in each experiment is selected using validation macro F1-score.

## Results

The strongest classical baseline was **Calibrated LinearSVC**, with:

- Accuracy: **0.9116**
- Macro F1-score: **0.9046**

**IndoBERT Balanced** achieved the highest validation macro F1-score of **0.9570**.

The best held-out test performance was achieved by **IndoBERT Balanced with Augmentation**, with:

- Accuracy: **0.9466**
- Macro F1-score: **0.9433**
- Total test errors: **46**

Therefore, IndoBERT Balanced with Augmentation was selected as the final Narasi model.

Detailed metrics, rankings, learning curves, and confusion matrices are available in:

```text
outputs_compare_modelresults/
```

## Explainability with LIME

Narasi integrates LIME to provide local word-level explanations for individual predictions. The explanation highlights words that support or oppose the predicted class.

Because LIME is perturbation-based, small differences in explanation weights may occur across requests even when the predicted class remains unchanged.

## System Implementation

Narasi consists of:

1. An IndoBERT Balanced with Augmentation model hosted on Hugging Face
2. A FastAPI backend for preprocessing, inference, confidence calculation, and LIME explanations
3. A custom HTML, CSS, and JavaScript frontend deployed on Vercel

Application flow:

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
      ├── IndoBERT inference
      ├── Confidence calculation
      └── LIME explanation
      |
      v
Prediction displayed to the user
```

## Run Locally

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

### 3. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Start the Application

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 7860
```

Open:

```text
http://localhost:7860
```

Interactive FastAPI documentation is available at:

```text
http://localhost:7860/docs
```

The backend downloads the final model from Hugging Face during the first run unless the model is already available locally.

## Run with Docker

```bash
docker build -t narasi-clickbait .
docker run -p 7860:7860 narasi-clickbait
```

Open:

```text
http://localhost:7860
```

## Running the Notebooks

Run the notebooks in this order:

1. `notebook/01_classical_models.ipynb`
2. `notebook/02_indoBERT_final.ipynb`
3. `notebook/03_compare_modelresults.ipynb`

The first notebook trains and evaluates the classical baselines. The second notebook fine-tunes the four IndoBERT strategies and exports the selected model. The third notebook combines the saved results and generates the final comparison files in `outputs_compare_modelresults/`.

Additional experiment libraries may be required to run the notebooks, including pandas, Matplotlib, SciPy, joblib, nlpaug, and Hugging Face Datasets.

## Limitations

- The study uses only the CLICK-ID `all_agree.csv` subset.
- The classical and IndoBERT pipelines apply different preprocessing procedures, so the comparison is pipeline-level rather than fully sample-identical.
- The experiments use one fixed random seed.
- No statistical significance testing is performed.
- LIME provides local approximations rather than exact descriptions of IndoBERT's internal reasoning.
- The system analyzes headlines only and does not compare them with full article content.

## Academic Disclaimer

Narasi was developed for academic research and educational demonstration. Its predictions should be treated as assistive information rather than an absolute judgment of journalistic quality.

## Contributors

- **Erlitna Natasya Debora**

- **Collyne**

## Supervisor

- **Mohammad Faisal Riftiarrasyid**
