# Amazon ML Challenge 2025 — VLM-Powered-E-commerce-Price-Regression (Team Solution)

## Overview
This repository contains our team’s solution for the **Amazon Machine Learning Challenge 2025**, where the task was to **predict e-commerce product prices** using multi-modal data — combining **product images and catalog descriptions**.  
We designed an **optimized Vision-Language Model (VLM)**–based regression system built upon **LLaVA-1.5-7b-hf**, integrated with a custom trainable regression head.

---

## Approach Summary
Our pipeline combines **computer vision** and **natural language understanding** by leveraging **LLaVA (Large Language and Vision Assistant)** — a vision-language model — and fine-tuning only a small regression head to predict continuous price values.  

Key aspects of our approach:
- **Multi-modal input**: Product image + textual description (`catalog_content`)  
- **Frozen LLaVA base** for speed and memory efficiency  
- **Custom regression head** trained for price estimation  
- **Optimized for A100 GPUs** using `bfloat16`, fused optimizers, and parallel data loading  
- **SMAPE (Symmetric Mean Absolute Percentage Error)** as the primary metric  

---

## Architecture

```text
                ┌────────────────────────────────────────────┐
                │               Input Data                   │
                │  Product Image + Catalog Text (CSV + JPG)  │
                └────────────────────────────────────────────┘
                                 │
                                 ▼
                 ┌─────────────────────────────────────┐
                 │     LLaVA Base Model (Frozen)        │
                 │  Extracts Visual + Textual Features  │
                 └─────────────────────────────────────┘
                                 │
                                 ▼
                 ┌─────────────────────────────────────┐
                 │  Trainable Regression Head (Custom)  │
                 │   FC Layers + LayerNorm + Dropout    │
                 └─────────────────────────────────────┘
                                 │
                                 ▼
                    Predicted Product Price (₹)
```
## Configuration

All static parameters are centralized in the `Config` class inside `amazon_ml_challenge.py`.

```python
class Config:
    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    TRAIN_CSV_PATH = "./data/train.csv"
    IMAGE_DIR = "./data/images/train"
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-5
    USE_LOG_TRANSFORM = True
    OUTPUT_DIR = "./llava_price_regressor_output"
```

---

## Evaluation Metric

We used **Symmetric Mean Absolute Percentage Error (SMAPE)** to measure model accuracy:

```python
def smape(y_true, y_pred):
    """Calculates Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100
```

This metric is robust to outliers and handles price variability effectively.

---

## Dataset

| Field | Description |
| :---- | :----------- |
| `sample_id` | Unique product identifier |
| `catalog_content` | Product description / title |
| `image` | Product image (JPEG) |
| `price` | Target variable (float) |

The dataset was preprocessed to remove missing prices and corrupted images.

---

## Model Components

1. **Dataset Loader** (`PricePredictionDataset`)  
   Handles dynamic prompt creation and log transformation for stability.

2. **Model Class** (`LLaVARegressionModel`)  
   - Freezes LLaVA base layers  
   - Adds a multi-layer regression head with `LayerNorm`, `ReLU`, and `Dropout`  

3. **Custom Callbacks**  
   - `PerformanceMonitorCallback` for GPU speed tracking  
   - `EarlyStoppingCallback` to prevent overfitting  

---

## Training

### Prerequisites
Install required packages:
```bash
pip install torch torchvision transformers unsloth pandas numpy scikit-learn pillow
```

### Run Training
Train the regression head using the script:
```bash
python amazon_ml_challenge.py
```

During runtime, the script automatically:
- Loads and preprocesses the dataset  
- Splits it into training and validation sets  
- Fine-tunes the regression head  
- Logs performance metrics and saves the best model checkpoint  

---

## Performance

| Metric | Value |
| :------ | :---- |
| SMAPE (Validation) | ≈ 7.8 |
| MSE (Log Scale) | 0.034 |
| Training Hardware | NVIDIA A100 (80 GB) |
| Batch Size | 32 |
| Precision | BF16 |

---

## Saving & Loading the Model

After training, the regression head and config are saved automatically.

```python
torch.save({
    'model_state_dict': model.reg_head.state_dict(),
    'config': {
        'use_log_transform': Config.USE_LOG_TRANSFORM,
        'max_seq_length': Config.MAX_SEQ_LENGTH,
        'model_name': Config.MODEL_NAME,
    }
}, "./llava_price_regressor_output/best_regression_head.pth")
```

To reload for inference:
```python
checkpoint = torch.load("./llava_price_regressor_output/best_regression_head.pth")
model.reg_head.load_state_dict(checkpoint['model_state_dict'])
```

---

## Key Learnings
- Fine-tuning large VLMs efficiently by **freezing base layers** and training lightweight heads is computationally optimal.  
- **SMAPE** proved robust for skewed price distributions.  
- Leveraging **multi-modal embeddings** can outperform text-only baselines by a large margin.  

---

## Contributors
**Team Members:**
- Anurag Singh  
- Krishnakant
- Rudra Goyal
- Nalin Kumar

---

## Repository Structure
```text
amazon_ml_challenge/
│
├── amazon_ml_challenge.py          # Main training script
├── data/
│   ├── train.csv
│   └── images/train/
├── llava_price_regressor_output/   # Saved model checkpoints
└── README.md
```

---

## Future Work
- Explore quantized LLaVA (4-bit/8-bit) for cost efficiency.  
- Deploy using FastAPI or Streamlit for interactive inference.  
- Experiment with LLaVA-Next and Phi-Vision models for improved generalization.  

---

## Citation
If you find this repository useful, please cite or star it 💫

```text
@project{amazon_ml_challenge_2025,
  author = {Anurag Singh et al.},
  title  = {Amazon ML Challenge 2025 - VLM-based Price Regression},
  year   = {2025},
  url    = {https://github.com/heckur08/amazon-ml-challenge-2025}
}
```
