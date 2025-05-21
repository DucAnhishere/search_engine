import os
import re
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_chunk(text):
    # Bỏ chunk nếu chỉ chứa tag <!-- image -->
    if re.fullmatch(r'\s*<!--\s*image\s*-->\s*', text.strip(), flags=re.IGNORECASE):
        return None

    # Xóa tất cả các tag <!-- image --> nếu nằm giữa nội dung
    text = re.sub(r'<!--\s*image\s*-->', '', text, flags=re.IGNORECASE)

    # Xóa dấu # markdown tiêu đề 
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Làm sạch ký tự lạ, giữ lại chữ, số và dấu câu cơ bản
    cleaned_text = re.sub(r'[^a-zA-Z0-9À-Ỹà-ỹ\s.,!?]', '', text).strip()

    return cleaned_text if cleaned_text else None

def chunking(directory_path, chunk_size=500, chunk_overlap=100):
    """Đọc tất cả các file trong thư mục (kể cả thư mục con), xử lý thành chunks"""

    file_list = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    all_chunks, resume_paths, resume_ids = [], [], []

    for idx, file_path in enumerate(file_list):
        try:
            doc = DocumentConverter().convert(source=file_path).document.export_to_markdown().replace("\n\n", "\n")

            text_splitter = RecursiveCharacterTextSplitter(
                separators=["##", "#", "\n"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )

            chunks = text_splitter.split_text(doc)

            cleaned_chunks = [clean_chunk(chunk) for chunk in chunks]
            cleaned_chunks = [chunk for chunk in cleaned_chunks if chunk is not None]

            all_chunks.extend(cleaned_chunks)
            resume_paths.extend([file_path] * len(cleaned_chunks))
            resume_ids.extend([f"resume_{idx:03}"] * len(cleaned_chunks))

        except Exception as e:
            print(f"❌ Lỗi khi xử lý file {file_path}: {e}")

    return all_chunks, resume_paths, resume_ids

