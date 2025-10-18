# data_processor.py
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# --- 1. Hàm làm sạch text (MỚI) ---
def clean_text(text: str) -> str:
    """
    Làm sạch text thô trích xuất từ PDF/DOCX.
    - Gộp các từ/dòng bị ngắt.
    - Xóa các khoảng trắng thừa.
    """
    # Gộp các từ bị ngắt dòng (ví dụ: "chuyển \n động" -> "chuyển động")
    text = re.sub(r'(\w+)\s*-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Gộp các dòng bị ngắt (nối các dòng không kết thúc bằng dấu chấm, chữ hoa...)
    # Regex này tìm một ký tự thường, theo sau là \n, theo sau là chữ cái
    text = re.sub(r'([a-z0-9vxyàáâãèéêìíòóôõùúýỳỹđ])\n([a-zA-Zvxyàáâãèéêìíòóôõùúýỳỹđ])', r'\1 \2', text)
    
    # Xóa các khoảng trắng/tab/newline thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # (Bạn có thể thêm các luật (rule) làm sạch khác ở đây)
    
    return text

# --- 2. Hàm khởi tạo Embedding ---
def get_embedding_function():
    """Khởi tạo và trả về embedding function."""
    model_name = "bkai-foundation-models/vietnamese-bi-encoder"
    model_kwargs = {'device': 'cpu'} # Thay 'cpu' bằng 'cuda' nếu có GPU
    encode_kwargs = {'normalize_embeddings': True}
    
    print("Đang tải model embedding (BKAI)...")
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("Tải model embedding thành công.")
    return embedding_function

# --- 3. Hàm khởi tạo Vector Store ---
def get_vectorstore(embedding_function):
    """Khởi tạo và trả về vector store (Chroma)."""
    print("Khởi tạo cơ sở dữ liệu vector (Chroma)...")
    vectorstore = Chroma(
        persist_directory="./db", 
        embedding_function=embedding_function
    )
    print("Khởi tạo DB thành công.")
    return vectorstore

# --- 4. Hàm xử lý file (đã cập nhật) ---
# --- 4. Hàm xử lý file (ĐÃ CẬP NHẬT) ---
def process_document(file_path: str, vectorstore: Chroma):
    """
    Đọc file, LÀM SẠCH, chia nhỏ và lưu vào vector database.
    (Phiên bản có thêm LOG DEBUG)
    """
    print(f"\n--- BẮT ĐẦU XỬ LÝ FILE: {file_path} ---")
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    # --- THÊM XỬ LÝ CHO .doc ---
    elif file_path.endswith(".doc"):
        print(f"Phát hiện file .doc, sử dụng UnstructuredFileLoader...")
        loader = UnstructuredFileLoader(file_path)
    # -----------------------------
    else:
        print(f"LỖI: File type not supported: {file_path}")
        raise ValueError("Unsupported file type")

    # --- Tải tài liệu ---
    try:
        documents = loader.load()
        print(f"[LOG 1] Đã trích xuất {len(documents)} trang/phần từ file.")

        if not documents:
            print("LỖI: Loader.load() trả về danh sách rỗng. File có thể bị hỏng, trống, hoặc là PDF dạng ảnh.")
            return  # Thoát sớm

        print(f"      Nội dung thô (500 ký tự đầu trang 1): {documents[0].page_content[:500]}...")

    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG KHI LOAD FILE: {e}")
        if file_path.endswith(".pdf"):
            print(
                "      GỢI Ý: Nếu đây là PDF dạng ảnh (scan), bạn cần dùng OCR (như PyTesseractLoader) thay vì PyPDFLoader.")
        raise e

    # --- BƯỚC LÀM SẠCH ---
    print("\n[LOG 2] Bắt đầu làm sạch text...")
    cleaned_content_exists = False
    for i, doc in enumerate(documents):
        original_len = len(doc.page_content)

        # Áp dụng hàm làm sạch
        doc.page_content = clean_text(doc.page_content)

        cleaned_len = len(doc.page_content)

        print(f"      Phần {i}: Độ dài gốc={original_len}, Độ dài sau clean={cleaned_len}")

        if not doc.page_content.strip():
            print(f"      CẢNH BÁO: Phần {i} bị rỗng SAU KHI làm sạch!")
        else:
            cleaned_content_exists = True

    print("Làm sạch text hoàn tất.")

    if not cleaned_content_exists:
        print("LỖI: Tất cả tài liệu đều rỗng sau khi làm sạch. Dừng xử lý.")
        return  # Thoát sớm

    # --- Chia văn bản ---
    print("\n[LOG 3] Bắt đầu chia văn bản (splitting)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"      Đã chia thành {len(docs)} chunks.")

    if not docs:
        print("LỖI: Text splitter không tạo ra chunks nào (docs rỗng).")
        return  # Thoát sớm

    # --- Lọc chunks rỗng (fix lỗi) ---
    valid_docs = [d for d in docs if d.page_content.strip()]

    print(f"      Số chunks ban đầu: {len(docs)}")
    print(f"      Số chunks RỖNG đã bị lọc bỏ: {len(docs) - len(valid_docs)}")
    print(f"      Số chunks HỢP LỆ (không rỗng): {len(valid_docs)}")

    if not valid_docs:
        print("LỖI: Tất cả các chunks đều rỗng sau khi split. Không có gì để thêm vào DB.")
        return  # Thoát sớm

    print(f"      Nội dung chunk hợp lệ đầu tiên (500 ký tự): {valid_docs[0].page_content[:500]}...")

    # --- Thêm vào vector store ---
    print("\n[LOG 4] Chuẩn bị thêm chunks hợp lệ vào vector store...")
    try:
        # Chỉ thêm các docs hợp lệ
        vectorstore.add_documents(valid_docs)
        print(f"--- HOÀN TẤT: Đã thêm {len(valid_docs)} chunks từ {file_path} vào DB. ---")
    except Exception as e:
        print(f"LỖI TRỰC TIẾP TỪ VECTORSTORE.ADD_DOCUMENTS: {e}")
        print("      Dù đã lọc, vẫn có lỗi. Kiểm tra xem model embedding (BKAI) có đang chạy đúng không.")
        raise e


# --- Phần test (Giữ nguyên) ---
if __name__ == '__main__':
    # File này có thể được chạy riêng để test
    print("Chạy data_processor.py ở chế độ test...")

    # Khi chạy riêng, nó tự khởi tạo mọi thứ
    test_emb = get_embedding_function()
    test_vs = get_vectorstore(test_emb)

    pass