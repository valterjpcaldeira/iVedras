# 🤖 iVedras – Iniciativa Torres Vedras - AI Demo for Torres Vedras

Welcome to **iVedras**!  
This is a demo project showing how Artificial Intelligence can help the city of Torres Vedras by empowering citizens to report and track urban issues easily.

✨ **All AI models were developed by me and are 100% free for everyone to use!**

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/valterjpcaldeira/iVedras.git
cd iVedras
```

### 2. Backend (API)
```bash
cd ivedras-app/backend
pip install -r requirements.txt
# Configure .env with your MongoDB URI and HuggingFace token
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Frontend (React)
```bash
cd ../frontend
npm install
npm run dev # or npm run build && npm run preview
```

### 4. Access
- Frontend: http://localhost:5173
- Backend: http://localhost:8000

---

## 🧠 What's Inside?
- 🤖 AI-powered complaint classification & urgency detection
- 🗺️ Interactive map of reported issues
- 📊 Real-time analytics dashboard

---

## ⚙️ Environment Variables
Create a `.env` file in `ivedras-app/backend/` with:
```
MONGODB_URI=your_mongodb_uri
HF_API_TOKEN=your_huggingface_token
```

---

## 🚢 Deploy on Railway
- Configure variables de ambiente no painel Railway.
- O build pode demorar devido ao download de modelos grandes. Use `.dockerignore` para acelerar.
- Certifique-se de que arquivos grandes (modelos, datasets, venvs) não estão no repositório.

---

## 💡 Why?
This project is a proof-of-concept to show how modern AI can make cities smarter, more responsive, and more connected to their citizens.

---

## 🆓 Free & Open
All models and code are open and free for anyone to use, adapt, or build upon!

