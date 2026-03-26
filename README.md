# Modular Multi-Label Email Classification Architecture

**National College of Ireland**  
**Engineering and Evaluating Artificial Intelligence (CA1)**  
**Due Date:** 26th March 2026  

## Project Overview

This project implements a modular multi-label email classification system supporting two distinct design strategies:
- **Design Decision 1:** Chained Multi-Output Classification
- **Design Decision 2:** Hierarchical Modelling

The system classifies emails across three hierarchical levels (Type2, Type3, Type4) following software architecture principles: Separation of Concerns, Encapsulation, and Abstraction.

## Features

**Modular Architecture** - Independent preprocessing, embeddings, and modeling layers  
**Two Design Strategies** - Compare efficiency vs. interpretability trade-offs  
**Data Encapsulation** - Unified Data class managing train/test splits  
**Abstract Model Interface** - BaseModel enforces consistent behavior across implementations  
**RandomForest Implementation** - Concrete model with stratified splitting and class weighting  
**TF-IDF Text Representation** - 2000 features, min_df=4, max_df=0.90  

## Project Structure

```
CA_EEA_Project/
  - README.md                    # This file
  - requirements.txt             # Python dependencies
  - .gitignore                   # Git ignore file
  - src/
      - Config.py                # Shared configuration (MIN_CLASS_COUNT=3, TEST_SIZE=0.2)
      - main.py                  # Main controller
      - preprocess.py            # Data preprocessing & noise removal
      - embeddings.py            # TF-IDF feature extraction
      - modelling/
          - data_model.py        # Data encapsulation class
          - modelling.py         # Both design strategies (chained & hierarchical)
      - model/
          - base.py              # Abstract BaseModel interface
          - randomforest.py      # RandomForest concrete implementation
  - data/
      - AppGallery.csv           # Sample data file
      - Purchasing.csv           # Sample data file
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/geethikasarayun-2001/CA_EEA_Project
cd CA_EEA_Project
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run Both Design Strategies
```bash
cd src
python main.py
```

**Output:**
- Level 1 (Type2) classification results
- Level 2 (Type2+Type3) classification results
- Level 3 (Type2+Type3+Type4) classification results
- Performance metrics for both strategies

### Run Specific Strategy
Edit `src/main.py` to call either:
- `chained_multi_output_classification(X, data)` - Single model, 3 levels
- `hierarchical_modelling(X, data)` - Multiple models with filtering

## Architecture Principles

### 1. Separation of Concerns
Each module handles one responsibility:
- **preprocess.py** → Data cleaning
- **embeddings.py** → Text vectorization
- **data_model.py** → Data encapsulation
- **modelling.py** → Strategy implementation
- **model/randomforest.py** → ML model

### 2. Encapsulation
Data class encapsulates train/test splits:
```python
data = Data(X, y)
X_train, X_test = data.X_train, data.X_test
y_train, y_test = data.y_train, data.y_test
```

### 3. Abstraction
BaseModel interface enforces consistent behavior:
```python
class BaseModel(ABC):
    @abstractmethod
    def train(self, data): pass
    @abstractmethod
    def predict(self, X_test): pass
    @abstractmethod
    def print_results(self, data): pass
```

## Configuration

All parameters are managed in `src/Config.py`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| MIN_CLASS_COUNT | 3 | Filter classes with <3 instances |
| TEST_SIZE | 0.2 | 80/20 train/test split |
| RANDOM_STATE | 0 | Reproducible results |
| VECTORIZER max_features | 2000 | TF-IDF feature limit |
| VECTORIZER min_df | 4 | Minimum document frequency |
| VECTORIZER max_df | 0.90 | Maximum document frequency |
| RandomForest n_estimators | 1000 | Number of trees |
| RandomForest class_weight | 'balanced_subsample' | Handle class imbalance |

## Design Strategies Comparison

### Chained Multi-Output Classification
- **Model Count:** 1 instance
- **Approach:** Single model predicts 3 levels sequentially
- **Level 1 Accuracy:** 69.05%
- **Level 2 Accuracy:** 66.67%
- **Level 3 Accuracy:** 69.05%
- **Advantages:** Efficient, consistent, simple
- **Disadvantages:** Large label space, low explainability

### Hierarchical Modelling
- **Model Count:** 7-10 instances
- **Approach:** Filter dataset by Level 1 prediction, create separate models per branch
- **Level 1 Accuracy:** 69.05%
- **Level 2 Accuracy:** 62.50% (average per class)
- **Level 3 Accuracy:** 50-100% (varies by branch)
- **Advantages:** Per-class optimization, high explainability
- **Disadvantages:** Complex, sparse data at deeper levels

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.3 | Numerical arrays |
| pandas | 2.1.4 | Data manipulation |
| scikit-learn | 1.4.1 | ML algorithms & metrics |

## Author

**Student:** Geethika Sarayu Neelam and Venkata Naga Sri Kalyn Somisetty  
**Course:** Engineering and Evaluating Artificial Intelligence  
**Institution:** National College of Ireland  
**Due Date:** 26th March 2026

## License

This project is submitted as part of National College of Ireland assessment requirements.
