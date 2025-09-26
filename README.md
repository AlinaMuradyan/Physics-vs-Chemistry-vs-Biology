# 📚 Physics vs Chemistry vs Biology Classification

## 📖 Project Overview
This project applies deep learning to classify text questions into one of three categories: **Physics**, **Chemistry**, or **Biology**.  
The dataset comes from [Kaggle: Physics vs Chemistry vs Biology](https://www.kaggle.com/datasets/vivmankar/physics-vs-chemistry-vs-biology).

The goal is to preprocess the text, train an LSTM-based model, and evaluate its performance on unseen data.

---

## 🗂 Dataset
- **Source:** Kaggle dataset: *Physics vs Chemistry vs Biology*  
- **Content:** Text samples labeled with one of three categories: Physics, Chemistry, Biology  
- **Format:** CSV file with two columns:  
  - `Comment` – the question or sentence  
  - `Topic` – label (Physics, Chemistry, Biology)  

---

## 🧰 Tools & Libraries
- **Python 3.x**  
- **TensorFlow / Keras** – deep learning framework  
- **NumPy, Pandas** – data manipulation  
- **Matplotlib / Seaborn** – visualization  
- **Scikit-learn** – preprocessing and evaluation  

---

## 🧩 Model Architecture
The model is based on an **LSTM network**:

- Embedding layer to convert words into dense vectors  
- Bidirectional LSTM (64 units)  
- Dropout layers for regularization  
- Dense hidden layer with ReLU activation  
- Softmax output layer for 3-class classification  

**Training setup:**
- Loss: categorical crossentropy  
- Optimizer: Adam (lr=0.001)  
- Metrics: accuracy  
- Regularization: Dropout + EarlyStopping  

---

## 📈 Results
- Training accuracy reached ~93%  
- Validation accuracy stabilized around **80%**  
- Validation loss started to increase after ~3 epochs, indicating some overfitting  

---

## ✅ Future Improvements
- Integrate **pretrained embeddings** (GloVe, FastText)  
- Experiment with **Transformer models** (e.g., BERT, DistilBERT)  
- Apply **text data augmentation** (synonym replacement, back translation)  
- Fine-tune hyperparameters (embedding size, LSTM units, dropout rate, learning rate)  
- Address potential class imbalance with class weighting or resampling  

---

## 🛠 How to Run
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install tensorflow keras numpy pandas matplotlib scikit-learn
    ```
3. Download the dataset from Kaggle and place it in a data/ folder
4. Preprocess the data (tokenization, padding, label encoding)
5. Train the model and evaluate on the test set
