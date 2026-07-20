# 📰 Fake News Intelligence System

An AI-powered Fake News Detection platform that combines Machine Learning, Natural Language Processing (NLP), and Live News Verification to identify whether a news article is real or fake.

The system analyzes textual content using multiple machine learning models and validates results against trusted news providers to improve prediction reliability.

---

## 🚀 Features

- 🤖 Multi-model Machine Learning prediction
- 🧠 Advanced NLP preprocessing pipeline
- 📊 Feature engineering and feature selection
- 🌐 Live verification using multiple news providers
- 📈 Interactive analytics dashboard
- 👤 User authentication with JWT
- 💬 User feedback collection
- 🛠️ Admin review panel
- 🔄 Automated model retraining pipeline
- 📱 Responsive React frontend

---

## 🏗️ Tech Stack

### Frontend
- React 18
- Vite
- Tailwind CSS
- React Router
- Axios
- Recharts
- Framer Motion

### Backend
- FastAPI
- SQLAlchemy
- PostgreSQL
- Alembic
- Pydantic
- JWT Authentication

### Machine Learning
- Scikit-Learn
- XGBoost
- spaCy
- NLTK
- Textstat
- Pandas
- NumPy

---

## 🤖 Machine Learning Models

The project implements multiple ML algorithms:

- Logistic Regression
- Linear Support Vector Machine (SVM)
- Random Forest
- XGBoost

The best-performing model is selected for production while the remaining models contribute to ensemble predictions.

---

## 🧠 ML Pipeline

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Text Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Feature Selection
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Prediction
   │
   ▼
Live Verification
   │
   ▼
Final Decision
```

---

## 🌐 Live Verification

The prediction is verified using trusted news providers:

- NewsAPI
- GNews
- NewsData

The system compares live articles with the predicted result to improve overall reliability.

---

## 📂 Project Structure

```
fake-news-intelligence/
│
├── backend/
│   ├── app/
│   ├── ml/
│   ├── config/
│   ├── tests/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/fake-news-intelligence.git

cd fake-news-intelligence
```

### Backend

```bash
cd backend

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

Run the server

```bash
uvicorn app.main:app --reload
```

---

### Frontend

```bash
cd frontend

npm install

npm run dev
```

---

## 🔐 Environment Variables

Backend `.env`

```env
DATABASE_URL=your_postgresql_database_url

SECRET_KEY=your_secret_key

ALGORITHM=HS256

ACCESS_TOKEN_EXPIRE_MINUTES=30

NEWS_API_KEY=your_newsapi_key

GNEWS_API_KEY=your_gnews_key

NEWSDATA_API_KEY=your_newsdata_key
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|----------|----------|-------------|
| POST | `/api/v1/auth/signup` | Register User |
| POST | `/api/v1/auth/login` | Login |
| POST | `/api/v1/predict` | Predict News |
| POST | `/api/v1/analyze-full` | Complete Analysis |
| POST | `/api/v1/verify-news` | Live Verification |
| POST | `/api/v1/feedback` | Submit Feedback |
| GET | `/api/v1/model/info` | Model Information |
| GET | `/api/v1/health` | Health Check |

---

## 📊 Dataset

- 56,000+ News Articles
- Data Cleaning
- Duplicate Removal
- NLP Preprocessing
- Feature Engineering
- Feature Selection
- Train/Test Split

---

## 📈 Future Improvements

- Transformer-based models (BERT/RoBERTa)
- Multi-language support
- Social media verification
- Mobile application
- Explainable AI (XAI)
- Cloud deployment automation

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Neredimelli Shankar**

AI & Machine Learning Enthusiast | Full Stack Developer

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub!
