import random
from routers import courses
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="MescourseAI API",
    description="AI-powered study assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "https://playground-alpha-pink.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(courses.router)

@app.get("/")
def home():
    return {"status": "healthy", "message": "MesCoursAI API is running"}



if __name__ == "__name__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
