from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple
from docx import Document
from PIL import Image
import tempfile
import os
import io


class DocumentService:
    """
    Service to handle multiple document formats: PDF, TXT, DOCX, Images.
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        "pdf": [".pdf"],
        "text": [".txt", ".md", ".markdown"],
        "word": [".docx", ".doc"],
        "image": [".png", ".jpg", ".jpeg", ".gif", ".webp"]
    }
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def get_file_type(self, filename: str) -> str:
        """
        Determine file type from extension.
        
        Returns: 'pdf', 'text', 'word', 'image', or 'unsupported'
        """
        ext = os.path.splitext(filename)[1].lower()
        
        for file_type, extensions in self.SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return file_type
        
        return "unsupported"
    
    def get_supported_extensions(self) -> List[str]:
        """Return flat list of all supported extensions."""
        extensions = []
        for ext_list in self.SUPPORTED_EXTENSIONS.values():
            extensions.extend(ext_list)
        return extensions
    
    async def extract_text(self, file_content: bytes, filename: str, api_key: str = None) -> Tuple[str, int, List[str]]:
        """
        Main entry point - routes to correct extraction method.
        
        Args:
            file_content: Raw bytes of the file
            filename: Original filename (used to determine type)
            api_key: Gemini API key (needed for image extraction)
            
        Returns:
            Tuple of (full_text, page_count, chunks)
        """
        file_type = self.get_file_type(filename)
        
        if file_type == "pdf":
            return await self._extract_from_pdf(file_content, filename)
        elif file_type == "text":
            return await self._extract_from_text(file_content)
        elif file_type == "word":
            return await self._extract_from_word(file_content)
        elif file_type == "image":
            return await self._extract_from_image(file_content, filename, api_key)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    
    async def _extract_from_pdf(self, file_content: bytes, filename: str) -> Tuple[str, int, List[str]]:
        """Extract text from PDF using LangChain's PyPDFLoader."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            full_text = ""
            for page in pages:
                full_text += page.page_content + "\n\n"
            
            chunks = self.text_splitter.split_text(full_text)
            
            return full_text.strip(), len(pages), chunks
            
        finally:
            os.unlink(tmp_path)
    
    async def _extract_from_text(self, file_content: bytes) -> Tuple[str, int, List[str]]:
        """
        Extract text from plain text files (.txt, .md).
        """
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        text = None
        
        for encoding in encodings:
            try:
                text = file_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if text is None:
            raise ValueError("Could not decode text file")
        
        # Count "pages" by paragraphs (rough estimate)
        paragraphs = text.split("\n\n")
        page_count = max(1, len(paragraphs) // 5)  # ~5 paragraphs per "page"
        
        chunks = self.text_splitter.split_text(text)
        
        return text.strip(), page_count, chunks
    
    async def _extract_from_word(self, file_content: bytes) -> Tuple[str, int, List[str]]:
        """
        Extract text from Word documents (.docx).
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            doc = Document(tmp_path)
            
            full_text = ""
            
            # Extract paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text += paragraph.text + "\n\n"
            
            # Extract tables (often contains important info!)
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        full_text += row_text + "\n"
                full_text += "\n"
            
            # Estimate page count
            page_count = max(1, len(full_text) // 3000)  # ~3000 chars per page
            
            chunks = self.text_splitter.split_text(full_text)
            
            return full_text.strip(), page_count, chunks
            
        finally:
            os.unlink(tmp_path)
    
    async def _extract_from_image(self, file_content: bytes, filename: str, api_key: str = None) -> Tuple[str, int, List[str]]:
        """
        Extract text from images using Gemini Vision.
        """
        if not api_key:
            raise ValueError("API key required for image text extraction")
        
        # Import 
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Load image
        image = Image.open(io.BytesIO(file_content))
        
        # Use Gemini Vision to extract text
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """Analyze this image and extract ALL text content from it. 
        This could be:
        - Lecture slides
        - Handwritten notes  
        - Textbook pages
        - Diagrams with labels
        
        Please:
        1. Extract all visible text, maintaining the logical structure
        2. Describe any diagrams, charts, or figures
        3. If there are mathematical equations, write them out
        
        Format the output clearly with proper sections."""
        
        response = model.generate_content([prompt, image])
        
        text = response.text
        chunks = self.text_splitter.split_text(text)
        
        return text.strip(), 1, chunks  # Images = 1 page
    
    def get_chunks(self, text: str) -> List[str]:
        """Split text into chunks for RAG."""
        return self.text_splitter.split_text(text)


# Singleton instance - use this name in imports
document_service = DocumentService()  