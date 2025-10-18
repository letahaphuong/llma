# check_db.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- PHẢI DÙNG CHÍNH XÁC MODEL EMBEDDING MÀ BẠN ĐÃ DÙNG TRƯỚC ĐÓ ---
model_name = "bkai-foundation-models/vietnamese-bi-encoder"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_function = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# ---------------------------------------------------------------------

print("Đang kết nối tới cơ sở dữ liệu vector...")
# Trỏ tới thư mục 'db' nơi bạn đã lưu vector
vectorstore = Chroma(persist_directory="./db", embedding_function=embedding_function)

# Từ khóa để tìm kiếm (thay bằng một thuật ngữ có trong tài liệu của bạn)
query = "máy biến áp"
print(f"\nĐang tìm kiếm các tài liệu liên quan đến: '{query}'")

# Thực hiện tìm kiếm
search_results = vectorstore.similarity_search(query, k=3) # Tìm 3 kết quả gần nhất

# In kết quả
if search_results:
    print(f"\n✅ Đã tìm thấy {len(search_results)} kết quả liên quan:\n")
    for i, doc in enumerate(search_results):
        print(f"--- Kết quả {i+1} ---")
        # In một phần nội dung và nguồn
        print(f"Nội dung: '{doc.page_content[:200]}...'")
        print(f"Nguồn: {doc.metadata.get('source', 'N/A')}\n")
else:
    print("\n❌ LỖI: Không tìm thấy bất kỳ tài liệu nào trong cơ sở dữ liệu khớp với truy vấn.")
    print("Có thể bạn chưa chạy file data_processor.py hoặc quá trình xử lý đã thất bại.")