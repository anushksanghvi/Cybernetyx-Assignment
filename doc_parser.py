import aiofiles
from pdfminer.high_level import extract_text as extract_pdf_text
import docx

async def parse_document(file):
    """Function to parse and extract text from different document types."""
    file_type = file.filename.split('.')[-1]
    
    if file_type == 'pdf':
        return await extract_pdf(file)
    elif file_type in ['doc', 'docx']:
        return await extract_docx(file)
    elif file_type == 'txt':
        return await extract_txt(file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

async def extract_pdf(file) -> str:
    """Extract text from a PDF file."""
    contents = extract_pdf_text(file.file)
    return contents

async def extract_docx(file) -> str:
    """Extract text from a DOCX file."""
    doc = docx.Document(file.file)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

async def extract_txt(file) -> str:
    """Extract text from a TXT file."""
    async with aiofiles.open(file.file, 'r') as f:
        contents = await f.read()
    return contents
