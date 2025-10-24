# 🧠 Pragya Vijay — AI/ML & Full-Stack Projects Repository

Welcome to my project showcase repository!  
This repo brings together three of my key AI/ML projects — each focused on solving real-world problems through intelligent systems and automation.

---

## 🤖 1. Bookstore Chatbot (AI-Powered Assistant)

### 📘 Overview
A conversational chatbot built for an **online bookstore**, designed to assist users with book availability, genre-based recommendations, and general support queries.  
It uses a **Neural Network model** for intent classification trained on text data.

### ⚙️ Tech Stack
**Language:** Python  
**Libraries Used:**
```python
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
```

### 💡 Features
- Pretrained intent classification model using Keras.  
- Tokenization and lemmatization for accurate text understanding.  
- Handles bookstore-specific queries (book genres, authors, recommendations, etc.).  
- Extensible JSON structure for adding new intents.  
- Smooth user experience with dynamic and context-aware responses.

### 🚀 Future Scope
- Integrate with real-time bookstore inventory APIs.  
- Deploy with Flask and a front-end interface for customer chat.  
- Add speech-to-text and voice assistant capabilities.

---

## 🧬 2. Self-Learning Chatbot (Dynamic Intent Updater)

### 📙 Overview
A **self-improving chatbot** capable of learning new responses over time.  
Unlike traditional static bots, this one evolves by saving user inputs and mapping them to new intents dynamically.

### ⚙️ Tech Stack
**Language:** Python  
**Libraries Used:**
```python
import json
from difflib import get_close_matches
```

### 💡 Features
- Self-learning mechanism for new user phrases.  
- Uses approximate string matching (`get_close_matches`) for similar queries.  
- Updates its JSON-based knowledge base automatically.  
- Avoids single-word replies; focuses on multi-word, context-rich phrases.  
- Can be integrated easily with websites or customer-support systems.

### 🚀 Future Scope
- Incorporate a lightweight neural intent model for better generalization.  
- Add admin dashboard for reviewing and approving new learned intents.  
- Expand to domain-specific support (education, healthcare, retail, etc.).

---

## 🩺 3. Pneumonia Classification (Deep Learning — CNN Model)

### 📗 Overview
A **Deep Learning–based Pneumonia Detection System** built using **PyTorch**.  
It classifies chest X-ray images into **Normal** or **Pneumonia** categories with high accuracy.

### ⚙️ Tech Stack
**Language:** Python  
**Libraries Used:**
```python
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
import io
import base64
import json
from io import BytesIO
from flask import Flask, request, jsonify, escape
```

### 💡 Features
- End-to-end CNN model using pretrained **ResNet architectures**.  
- Data preprocessing: normalization, augmentation, and resizing for optimal accuracy.  
- Achieved ~93% accuracy on validation data.  
- Flask integration for REST API deployment and web-based image upload.  
- Model performance evaluated using confusion matrix and accuracy metrics.

### 🚀 Future Scope
- Add Grad-CAM for explainable AI visualizations.  
- Convert model to ONNX/TorchScript for lightweight deployment.  
- Build an interactive web dashboard using Streamlit or FastAPI.  

---

## 👩‍💻 About Me

I’m **Pragya Vijay**, a **B.Tech CSE student** skilled in **MERN stack**, **AI/ML**, and **Deep Learning**.  
I love working on innovative projects that combine technology, intelligence, and creativity to solve real-world challenges.

📫 **Connect With Me:**  
- **GitHub:** [github.com/PragyaVijay](https://github.com/PragyaVijay)  
- **LinkedIn:** [linkedin.com/in/pragyavijay](https://linkedin.com/in/pragyavijay)  
- **Email:** pragyavijay20318@gmail.com  

---

⭐ *If you found my projects interesting, consider starring this repository!*

