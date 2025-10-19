# app.py
import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from werkzeug.utils import secure_filename

# Sửa đổi import: Tách riêng các hàm khởi tạo
from data_processor import (
    process_document,
    get_vectorstore,
    get_embedding_function
)

# --- CẤU HÌNH DATABASE LOG ---
DB_NAME = 'qa_log.db'

def init_db():
    """Khởi tạo cơ sở dữ liệu SQLite để lưu log hỏi đáp."""
    print(f"Đang khởi tạo database log tại: {DB_NAME}")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS qa_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            email TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("Khởi tạo database log thành công.")

# --- KHỞI TẠO CÁC THÀNH PHẦN CHÍNH (CHỈ 1 LẦN KHI APP START) ---

print("--- Khởi tạo ứng dụng AI ---")

# 1. Khởi tạo Embedding
embedding_function = get_embedding_function()

# 2. Khởi tạo Vector Store
vectorstore = get_vectorstore(embedding_function)

# 3. Khởi tạo LLM
print("Đang tải model LLM (LlamaCpp)...")
llm = LlamaCpp(
    model_path="C:/Users/DevLife/PycharmProjects/PythonProject/models/vinallama-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_batch=512,
    n_ctx=4096,
    verbose=True,
    streaming=False
    # Dựa trên thông tin tôi biết, bạn muốn streaming data.
    # Khi bạn sẵn sàng nâng cấp, hãy:
    # 1. Đặt streaming=True ở đây.
    # 2. Sửa lại route /ask để trả về một Response(generator)
    # 3. Sửa lại JavaScript (chat.html) để nhận stream (dùng Fetch API + ReadableStream)
)
print("Tải model LLM thành công.")

# 4. Tạo Prompt Template
prompt_template = """Sử dụng những thông tin được cung cấp dưới đây để trả lời câu hỏi.
Nếu không tìm thấy câu trả lời trong thông tin này, hãy nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
Hãy trả lời bằng tiếng Việt, chi tiết và trích dẫn nguồn nếu có thể.

Context: {context}

Question: {question}

Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# 5. Tạo chuỗi xử lý QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

print("--- Khởi tạo hoàn tất. Máy chủ sẵn sàng. ---")

# --- CẤU HÌNH MÁY CHỦ FLASK ---

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Khởi tạo DB khi app chạy
init_db()


# --- CÁC API ROUTES (ĐÃ CẬP NHẬT) ---

@app.route('/')
def home():
    """Phục vụ file index.html (trang đăng nhập)."""
    # File này phải nằm trong thư mục /templates
    return render_template('index.html')

@app.route('/chat')
def chat_page():
    """Phục vụ file chat.html (trang hỏi đáp chính)."""
    # File này phải nằm trong thư mục /templates
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    email = data.get('email') # Nhận thêm email

    if not question or not email:
        return jsonify({"error": "Question and email are required"}), 400

    try:
        print(f"Nhận câu hỏi từ [{email}]: {question}")

        # 1. Gọi chuỗi QA để lấy kết quả
        result = qa_chain({"query": question})
        answer_text = result['result'].strip()

        # 2. Lấy nguồn tài liệu
        sources = [doc.metadata.get('source', 'N/A') for doc in result['source_documents']]
        unique_sources = list(set(sources))

        # 3. LƯU VÀO DATABASE
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO qa_logs (timestamp, email, question, answer) VALUES (?, ?, ?, ?)",
                      (datetime.now(), email, question, answer_text))
            conn.commit()
            conn.close()
            print(f"Đã lưu log cho [{email}] vào database.")
        except Exception as db_error:
            print(f"LỖI: Không thể lưu log vào DB: {db_error}")

        # 4. Trả kết quả về client
        print("Đã tạo câu trả lời, trả về client.")
        return jsonify({
            "answer": answer_text,
            "sources": unique_sources
        })

    except Exception as e:
        print(f"Lỗi khi xử lý câu hỏi: {e}")
        return jsonify({"error": "Đã có lỗi xảy ra trên máy chủ."}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('file')

    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected files"}), 400

    processed_files = []
    errors = []

    for file in files:
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            try:
                file.save(filepath)
                process_document(filepath, vectorstore)
                processed_files.append(filename)
            except Exception as e:
                print(f"Lỗi khi xử lý file '{filename}': {e}")
                errors.append(f"Error processing '{filename}': {str(e)}")

    if not processed_files and not errors:
         return jsonify({"error": "No valid files to process"}), 400

    return jsonify({
        "message": f"Processing complete. {len(processed_files)} file(s) processed successfully.",
        "processed": processed_files,
        "errors": errors
    }), 200


# --- API LẤY LỊCH SỬ (MỚI) ---

@app.route('/history', methods=['GET'])
def get_history():
    """Lấy lịch sử hỏi đáp dựa trên email."""

    # Lấy email từ query parameter (ví dụ: /history?email=user@example.com)
    email = request.args.get('email')

    if not email:
        return jsonify({"error": "Email parameter is required"}), 400

    print(f"Nhận yêu cầu xem lịch sử cho: {email}")

    try:
        conn = sqlite3.connect(DB_NAME)
        # Cài đặt row_factory để trả về kết quả dạng dictionary
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Truy vấn các cột cần thiết, sắp xếp theo thời gian mới nhất
        c.execute(
            "SELECT question, answer, timestamp FROM qa_logs WHERE email = ? ORDER BY timestamp DESC",
            (email,)
        )

        rows = c.fetchall()
        conn.close()

        # Chuyển đổi danh sách các đối tượng Row thành danh sách dictionary
        history_list = [dict(row) for row in rows]

        print(f"Đã tìm thấy {len(history_list)} bản ghi lịch sử cho {email}.")

        return jsonify({
            "email": email,
            "history": history_list
        })

    except Exception as e:
        print(f"Lỗi khi truy vấn lịch sử DB: {e}")
        # Đảm bảo đóng kết nối nếu có lỗi xảy ra
        if 'conn' in locals() and conn:
            conn.close()
        return jsonify({"error": "Đã có lỗi xảy ra khi lấy lịch sử."}), 500

# --- CHẠY MÁY CHỦ ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)