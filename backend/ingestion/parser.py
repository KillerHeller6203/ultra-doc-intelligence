import io
from pypdf import PdfReader
from docx import Document

class DocumentParser:
    @staticmethod
    def parse_pdf(file_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    @staticmethod
    def parse_docx(file_bytes: bytes) -> str:
        doc = Document(io.BytesIO(file_bytes))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    @staticmethod
    def parse_txt(file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8")

    def parse(self, filename: str, file_bytes: bytes) -> str:
        if filename.endswith(".pdf"):
            return self.parse_pdf(file_bytes)
        elif filename.endswith(".docx"):
            return self.parse_docx(file_bytes)
        elif filename.endswith(".txt"):
            return self.parse_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
