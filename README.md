# 🧠 Brain Tumor MRI Classification Dashboard

An end-to-end Deep Learning system that detects and classifies brain tumors from MRI scans into four distinct categories with **97.10% test accuracy** and a **96.88% macro F1-score**.

Deployed with an interactive **Streamlit** dashboard featuring confidence scoring, multi-class radar visualizations, and educational tumor profiles.

---

## 📊 Key Highlights & Performance

- **Backbone**: Pretrained Deep Residual Network (**ResNet-50**) with Residual Connections.
- **Classifier Head**: Tuned regularized MLP (`Dropout(0.3)` $\to$ `Linear(2048, 512)` $\to$ `ReLU` $\to$ `Dropout(0.2)` $\to$ `Linear(512, 4)`).
- **Optimization**: AdamW optimizer with Label Smoothing ($0.05$) and Cosine Annealing learning rate schedule.
- **Dataset Size**: Trained on **5,712** full MRI scans; evaluated on **1,311** unseen test scans.

### Final Evaluation Metrics (Held-Out Test Set)

| Class | Precision | Recall | F1-Score | Support (Scans) |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | 96.84% | 92.00% | **94.36%** | 300 |
| **Meningioma** | 92.43% | 95.75% | **94.06%** | 306 |
| **Pituitary** | 99.01% | 99.67% | **99.34%** | 300 |
| **No Tumor** (Healthy) | 99.51% | 100.00% | **99.75%** | 405 |
| **Overall / Macro Avg** | **96.95%** | **96.85%** | **96.88%** | **1,311** |

---

## 📁 Project Directory Structure

```text
brain_tumor_dataset/
├── app.py                      # Main Streamlit web application & dashboard
├── train_model.py              # End-to-end training & evaluation pipeline
├── best_brain_tumor_model.pth  # Trained model weights (97.1% test accuracy)
├── requirements.txt            # Python dependencies
├── RUN_ME.bat                  # One-click Windows batch launcher
├── RUN_ME2.bat                 # Alternative Windows batch launcher
├── README.md                   # Project documentation
│
├── Training/                   # Training dataset (5,712 images)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── Testing/                    # Unseen test dataset (1,311 images)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── sample_images/              # Curated sample scans for quick testing
└── Main_MRIdata_project.ipynb  # Exploratory notebook
```

---

## 🚀 Quick Start (Running the Dashboard)

### Option 1: Double-Click Launcher (Windows)
Simply double-click **`RUN_ME.bat`** (or **`RUN_ME2.bat`**). It will install dependencies and start the app in your default browser.

### Option 2: Terminal Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the Streamlit app
streamlit run app.py
```

The web dashboard will open automatically at `http://localhost:8501`.

---

## 🛠️ Retraining the Model

To retrain the model from scratch on the full dataset:

```bash
python train_model.py
```

This will:
1. Load all 5,712 training scans and 1,311 test scans at $224 \times 224$ resolution.
2. Extract representations using the ResNet-50 backbone.
3. Optimize the classification head using AdamW and Cosine Annealing.
4. Output a full classification report.
5. Save the updated checkpoint directly to `best_brain_tumor_model.pth`.

---

## 🔬 Tumor Class Overview

- **Glioma**: Tumors originating in glial cells within the brain tissue. They are often infiltrative and require rapid diagnostic attention.
- **Meningioma**: Tumors arising from the meninges (membranes surrounding the brain and spinal cord). Mostly benign and slow-growing.
- **Pituitary Tumor**: Tumors in the pituitary gland at the base of the skull, impacting hormonal and endocrine functions.
- **No Tumor**: Healthy brain MRI showing normal bilateral symmetry and ventricular structure.

---

## ⚖️ Disclaimer
*This application is built for research, educational, and pair-programming demonstration purposes. It should not be used as a substitute for professional clinical medical diagnosis.*
