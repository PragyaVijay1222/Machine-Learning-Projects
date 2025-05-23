# 📚 Chatbot for Online Bookstores

## Overview  
This project is an **NLP-based chatbot** designed to assist users by answering frequently asked questions about an online bookstore. The chatbot leverages **Natural Language Processing (NLP)** techniques to understand queries and generate relevant responses.
**Key Features & Implementation:**
- *Data Preparation:* Compiled frequently asked questions (FAQs) into a structured JSON dataset for model training.
- *Model Architecture:* Built a Sequential Model using TensorFlow with ReLU activation layers to ensure optimal learning.
- *Overfitting Prevention:* Incorporated dropout layers to improve model generalization and prevent overfitting.
- *Training & Optimization:* Trained the chatbot for 200 epochs, fine-tuning the accuracy and response reliability.
- *Language Processing:* Utilized NLTK for tokenization and preprocessing of user queries to enhance understanding.
- *Efficiency & Deployment:* Pickle serialization was used for fast storage and retrieval of trained model parameters, improving performance.

## 📂 Project Structure  
The project contains the following key files:

### 1️⃣ `training.ipynb` - **Chatbot Model Training**
This file is responsible for preprocessing the dataset and training the model.

#### 🔹 Key Functionalities:
- **Data Loading:** Imports the FAQ dataset (`intents.json`) for training.  
- **Text Preprocessing:** Tokenization using `nltk` and lemmatization via `WordNetLemmatizer`.  
- **Feature Extraction:** Converts words into vectors using `bag-of-words`.  
- **Model Architecture:**  
  - Implements a **Sequential Model** with **ReLU activation layers**.  
  - Includes **Dropout layers** to prevent overfitting.  
  - Uses **categorical cross-entropy loss** with **SGD optimizer**.  
- **Model Training:**  
  - Trained for **200 epochs** with **batch size = 5**.  
  - Saves the trained model as `chatbot_model.h5`.  

### 2️⃣ `chatbot.ipynb` - **Chatbot Inference and Response Generation**
This file contains functions to predict user intents and generate responses.

#### 🔹 Key Functions:
- **`clean_up_sentence(sentence)`**  
  - Tokenizes input text and applies lemmatization.  

- **`bag_of_words(sentence)`**  
  - Converts tokenized words into vectorized format using the bag-of-words approach.  

- **`predict_class(sentence)`**  
  - Uses the trained model to predict the intent of the given user query.  
  - Filters out low-probability responses using an **error threshold (0.25)**.  

- **`get_response(intents_list, intents_json)`**  
  - Retrieves the appropriate response based on predicted intent.  
  - Chooses a response from predefined **FAQ datasets** (`intents.json`).  

### 3️⃣ `intents.json` - **Predefined Dataset**
This file stores structured data containing frequently asked bookstore-related questions with their associated responses.  

---

## 🚀 Features  
✔ **Efficient NLP-based query resolution**  
✔ **Optimized training with dropout layers**  
✔ **Preprocessed text using tokenization & lemmatization**  
✔ **Response retrieval based on probability ranking**  

## 🛠 Technologies Used  
- **Python**  
- **NLTK** (`word_tokenize`, `WordNetLemmatizer`)  
- **TensorFlow & Keras (Sequential Model, Dense Layers, Dropout)**  
- **SGD Optimizer for Model Training**  
- **JSON for intent storage**  

## 🔮 Future Enhancements  
🔹 Expand the dataset with **new FAQs**  
🔹 Integrate **dynamic user input training**  
🔹 Deploy the chatbot via **Flask or FastAPI**  

---

## 📌 How to Run  
1️⃣ Install dependencies:  
   ```bash
   pip install nltk tensorflow json pickle numpy

2️⃣ **Run** `training.ipynb` **to train the model.**  
3️⃣ **Execute** `chatbot.ipynb` **to interact with the chatbot.**  
