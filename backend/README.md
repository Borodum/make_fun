FastAPI Backend for Image-Based Joke Generator

This is the backend service for the Image-Based Joke Generator project. It provides APIs for uploading images. Built with FastAPI.

---

Features
- Upload images via API
- CORS enabled to allow requests from React frontend
- Modular routing with FastAPI routers

---

Requirements
- Python 3.10
- FastAPI
- Uvicorn
- python-multipart

Install dependencies using:
pip install -r requirements.txt

---

Project Structure

```
backend/
├─ app/
│  ├─ main.py              # FastAPI app entrypoint
│  ├─ routers/
│  │  └─ image.py          # Router for image upload
│  └─ ...                  # other backend files
├─ venv/                   # Python virtual environment
└─ requirements.txt        # Project dependencies
```
---

Setup & Run
1. Activate virtual environment (Windows):
```
cd backend
.\venv\Scripts\activate
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the backend server:
```
uvicorn app.main:app --reload
```

4. The backend will run at:
```
http://127.0.0.1:8000
```

---

API Endpoints
Upload Image & Generate Jokes
- POST /images/upload/
- Form Data:  
  - file: image file to upload
- Response:
```
{
  "jokes": [
        "Когда фотошоп решил, что у тебя недостаточно красивый день, он добавил немного фильтров... и твоего бывшего на задний план!",
        "Это фото настолько эпичное, что даже камера попросила автограф!",
        "Если бы эта картинка могла говорить, она бы сказала: 'Я слишком крута для этого мира!",
    ]
}
```