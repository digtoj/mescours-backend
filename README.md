# MesCoursAI Backend

AI-powered study assistant API built with FastAPI and Google Gemini.

## Features

- **Document Upload & Processing** 
: Extract text from PDF, DOCX, TXT, MD, and images
- **AI Summarization** - Generate summaries, key points, and audio scripts from course content
- **Q&A** - Ask questions about your course content and get AI-powered answers
- **Image OCR** - Extract text from images using Gemini Vision

## Tech Stack

- **Framework**: FastAPI
- **AI**: LangChain + Google Gemini
- **Document Processing**: PyPDF, python-docx, Pillow, pytesseract

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mescours-backend
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Add your Google Gemini API key
   ```

## Usage

### Start the server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/api/courses/supported-formats` | Get supported file formats |
| `POST` | `/api/courses/upload` | Upload and extract text from a document |
| `POST` | `/api/courses/summarize` | Generate AI summary from course content |
| `POST` | `/api/courses/ask` | Ask a question about course content |

## Supported File Formats

- **Documents**: PDF, DOCX, TXT, MD
- **Images**: PNG, JPG, JPEG, GIF, WEBP (requires API key for OCR)


## License

Free to use
