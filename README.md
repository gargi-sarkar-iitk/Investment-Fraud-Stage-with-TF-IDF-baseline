# Investment Fraud Detection - TF-IDF Baseline Model

A multi-label text classification system for identifying stages in investment fraud scams using machine learning. This project analyzes victim complaint texts to detect different phases of fraud schemes including recruitment, trust-building, payment triggers, lock-in tactics, and continued extraction.

## Overview

Investment fraud schemes typically follow a predictable pattern with distinct stages. This project builds a baseline classifier to automatically identify these stages from victim complaints, enabling better understanding and prevention of fraud patterns.

### Fraud Stages Detected

- **Recruitment**: Initial contact and victim acquisition
- **Trust**: Building credibility and establishing relationships
- **TriggerPayment**: Prompting the victim to make their first investment
- **LockIn**: Preventing withdrawal through various tactics
- **ContinuedExtraction**: Ongoing extraction of funds from victims

## Dataset

The project uses the `investment_fraud_translated.xlsx` dataset containing:
- Victim complaint texts (in English translation)
- Demographics (age, gender, profession)
- Financial impact (amount lost)
- Case details (police station, acknowledgement numbers)
- Fraud type classifications
- Platform information (Telegram, WhatsApp, etc.)

**Key Features:**
- Multi-label classification (complaints can belong to multiple stages)
- Real-world fraud cases from law enforcement records
- Includes metadata: victim demographics, financial loss, suspect platforms

## Methodology

### Text Preprocessing
1. Lowercasing and whitespace normalization
2. Removal of special characters and digits
3. Tokenization
4. Stopword removal (English)
5. Lemmatization using WordNet

### Feature Extraction
- **TF-IDF (Term Frequency-Inverse Document Frequency)**
  - Captures word importance across documents
  - Max features: 5000
  - N-gram range: (1, 2) - unigrams and bigrams
  - Baseline approach for text classification

### Model Architecture
- **Binary Relevance with Logistic Regression**
  - One classifier per label (5 binary classifiers)
  - Handles multi-label outputs effectively
  - Simple, interpretable baseline model
  - Class weight balancing for imbalanced data

### Evaluation Metrics
- **Per-label metrics**: Precision, Recall, F1-Score
- **Overall metrics**: 
  - Macro-average (treats all labels equally)
  - Micro-average (weighted by support)
  - Hamming Loss (label-wise accuracy)
  - Subset Accuracy (exact match)
  - ROC-AUC scores

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/investment-fraud-detection.git
cd investment-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6.0
openpyxl>=3.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Usage

### Running the Notebook

```bash
jupyter notebook Investment_Fraud_TF_IDF_baseline.ipynb
```

### Key Code Sections

**1. Data Loading**
```python
import pandas as pd
df = pd.read_excel('investment_fraud_translated.xlsx')
```

**2. Text Preprocessing**
```python
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Preprocessing pipeline applied to complaint texts
```

**3. Model Training**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression

# TF-IDF vectorization + Binary Relevance classifier
```

**4. Evaluation**
```python
from sklearn.metrics import classification_report, hamming_loss
# Comprehensive evaluation on test set
```

## Results

The baseline model provides:
- **Interpretable features**: TF-IDF scores show which words are most important
- **Per-stage performance**: Individual metrics for each fraud stage
- **Benchmark results**: Baseline for comparing advanced models

### Expected Outputs
- Classification reports per label
- Confusion matrices
- ROC curves for each fraud stage
- Feature importance analysis (top TF-IDF terms)

## Project Structure

```
investment-fraud-detection/
├── Investment_Fraud_TF_IDF_baseline.ipynb  # Main analysis notebook
├── investment_fraud_translated.xlsx         # Dataset (not included)
├── README.md                                # This file
├── requirements.txt                         # Python dependencies
└── results/                                 # Model outputs and visualizations
    ├── classification_reports/
    ├── confusion_matrices/
    └── roc_curves/
```

## Future Improvements

1. **Advanced Models**
   - BERT/transformer-based models for better context understanding
   - Deep learning architectures (LSTM, CNN)
   - Ensemble methods

2. **Feature Engineering**
   - Domain-specific features (financial terms, platform mentions)
   - Named entity recognition
   - Sentiment analysis
   - Network/graph features from suspect platforms

3. **Data Augmentation**
   - Synthetic text generation
   - Back-translation
   - Cross-lingual models

4. **Deployment**
   - REST API for real-time predictions
   - Integration with fraud reporting systems
   - Model monitoring and retraining pipeline


## Acknowledgments

- Dataset sourced from law enforcement fraud complaint records
- Built using scikit-learn and scikit-multilearn libraries
- NLTK for text preprocessing

## Contact

For questions or collaboration opportunities, please open an issue or contact [your-email@example.com]

---

**Disclaimer**: This project is for research and educational purposes. Always consult with cybersecurity and legal professionals when implementing fraud detection systems in production environments.
