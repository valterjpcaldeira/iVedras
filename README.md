# 🎭 Torres sem Máscara

## 📱 About
Torres sem Máscara is a citizen engagement platform that empowers residents of Torres Vedras to report and track urban issues in their community. From potholes to broken street lights, this app makes it easy for citizens to contribute to making their city better while helping local authorities address concerns efficiently.

## ✨ Features
- 📝 Submit detailed complaints about urban issues
- 🗺️ Interactive map showing reported problems
- 🚨 Automatic urgency classification
- 📊 Dashboard with complaint analytics
- 🔍 Smart topic detection
- 📍 Location-based reporting
- 📱 User-friendly interface
- 📈 Real-time complaint tracking

## 🎯 Types of Issues You Can Report
- 🚧 Road and infrastructure problems
- 💡 Street lighting issues
- 🗑️ Waste management concerns
- 🌳 Parks and public spaces
- 🚰 Water and sanitation
- 🐕 Animal and environmental issues
- 🚗 Traffic and mobility
- 🏪 Commercial activities
- 🛡️ Public safety
- And more!

## 🚀 Getting Started
1. Clone the repository
2. Install dependencies
3. Set up your environment variables
4. Run the app
5. Start reporting issues!

## 🛠️ Tech Stack
- Python
- Streamlit
- MongoDB
- Folium for mapping
- Natural Language Processing
- Machine Learning for classification
- Google Maps API

## 🤝 Contributing
We welcome contributions! Feel free to submit issues and pull requests to help improve the platform.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact
Have questions or suggestions? Reach out to us!

---
Made with ❤️ for Torres Vedras

# iVedras Project

![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Railway](https://img.shields.io/badge/Deployed%20on-Railway-blue)
![Vercel](https://img.shields.io/badge/Frontend-Vercel-black)

## Project Overview
This project provides a backend API for complaint classification and urgency detection for the Torres Vedras region. It includes scripts for model training and data extraction, and is ready for deployment on Railway (backend) and Vercel (frontend).

## Table of Contents
- [Backend (FastAPI)](#backend-fastapi)
- [Model Training](#model-training)
- [Data Extraction](#data-extraction)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)

## Backend (FastAPI)
The backend is located in `ivedras-app/backend/` and provides endpoints for classifying complaints and managing complaint data.

### How to Run Locally
```bash
cd ivedras-app/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### How to Deploy on Railway
- Push your repo to Railway
- Set environment variables as described below
- Railway will automatically detect and run the FastAPI app

## Model Training
Model training scripts are in the `modeling/` folder (renamed from `auxiliar`).
- `model_train.py`: Train the main complaint classification model
- `model_train_urgency.py`: Train the urgency detection model
- `generate_pytorch_model_bin.py`: Export models to PyTorch format

## Data Extraction
Data extraction scripts are also in `modeling/`:
- `process_addresses.py`, `extract_address.py`: Extract and process address data
- `upload_addresses_to_mongodb.py`: Upload processed data to MongoDB
- Notebook: `notebooks/EstracaoMorada.ipynb`

## Model Details
- Models are based on HuggingFace Transformers.
- Training data: [Describe your dataset here, e.g., CSVs, sources, annotation process]
- Training approach: [Describe your approach, e.g., fine-tuning, hyperparameters]
- Artifacts are stored in `/iVedras_queixas/` and `/iVedras_urgencia/` (ignored by git)

## API Endpoints
- `GET /` — Health check (returns `{ "status": "ok" }`)
- `POST /classify` — Classify complaint text (returns topic and urgency)
- `GET /complaints` — List all complaints
- `POST /complaints` — Submit a new complaint

## Environment Variables
Set these in Railway or your local `.env`:
- `MONGODB_URI` — MongoDB connection string
- `TOPIC_MODEL_REPO` — HuggingFace repo for topic model
- `URGENCY_MODEL_REPO` — HuggingFace repo for urgency model
- `HF_API_TOKEN` — HuggingFace API token
- `PORT` — (Set by Railway)

## Deployment
### Backend (Railway)
- Set environment variables
- Deploy via Railway dashboard or CLI

### Frontend (Vercel)
- Set `NEXT_PUBLIC_API_URL` to your Railway backend URL (e.g., `https://your-backend.up.railway.app`)
- Deploy via Vercel dashboard or CLI

---

## License
[MIT License](LICENSE)

