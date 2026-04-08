# Job Posting Classification System

## Live Demo
https://jobpostingclassification-gtpxazpvnje7he8vcbssim.streamlit.app/

---

## Table of Contents
- Overview
- Key Features
- Tech Stack
- Dataset
- Approach
  - Data Preprocessing
  - Feature Engineering
  - Model Architecture
- Results
- Deployment
- How to Run
- Key Learnings
- Future Work
- Author

---

## Overview

This project focuses on building an intelligent NLP based system to classify job postings into 5 categories using both traditional machine learning and deep learning techniques.

The system combines statistical and semantic features to improve classification accuracy and generalization.

- 3K plus job postings  
- 5 job categories  
- Accuracy 98.2 percent  
- F1 score 0.98  

---

## Key Features

- Ensemble NLP model using TF IDF SVM and DistilBERT  
- Combines statistical and contextual text features  
- High accuracy and robust performance  
- Explainable predictions using SHAP  
- Real time prediction system  

---

## Tech Stack

- Python  
- Pandas NumPy  
- Scikit learn  
- Transformers DistilBERT  
- SHAP  
- Streamlit  

---

## Dataset

- 3K plus job descriptions  
- Text data with labeled job categories  
- Preprocessed using cleaning tokenization and normalization  

---

## Approach

### Data Preprocessing
- Text cleaning and normalization  
- Tokenization and stopword removal  
- Vectorization using TF IDF  

### Feature Engineering
- Extracted 20K TF IDF features  
- Generated contextual embeddings using DistilBERT  
- Combined both features for better performance  

### Model Architecture
- SVM for TF IDF features  
- DistilBERT for deep contextual understanding  
- Ensemble approach combining both outputs  

---

## Results

| Metric | Value |
|--------|------|
| Accuracy | 98.2 percent |
| F1 Score | 0.98 |
| Improvement | 6 to 13 percent |

---

## Deployment

- Built a lightweight web application using Streamlit  
- Supports real time predictions  
- Displays prediction confidence scores  
- Optimized for fast inference  

---

## How to Run

```bash
git clone https://github.com/yourusername/job-classification.git
cd job-classification
pip install -r requirements.txt
streamlit run app.py
```

## Key Learnings

- Combining TF IDF and BERT improves performance  
- Feature fusion reduces prediction error  
- Model interpretability is important in NLP systems  
- Ensemble methods provide better generalization  

---

## Future Work

- Add more job categories  
- Use larger transformer models  
- Improve dataset size and diversity  
- Deploy using scalable backend  

---

## Author

Yanshu Patel  
LinkedIn https://www.linkedin.com/in/yanshu-patel-165b2b297/  
GitHub  
