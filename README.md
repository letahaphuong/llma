# 🧠 Flask + LangChain + Local LLM Demo

Dự án này minh họa cách xây dựng API sử dụng **Flask** làm backend, **LangChain** để quản lý mô hình ngôn ngữ, và **llama-cpp-python** để chạy mô hình LLM local (offline, không cần OpenAI API).

---

## 🚀 1. Yêu cầu hệ thống

- Python 3.13 hoặc cao hơn  
- pip (đã cài sẵn trong Python)  
- Máy có ít nhất **8GB RAM** nếu muốn chạy LLM local  
- (Tùy chọn) GPU hỗ trợ tăng tốc inference (nếu dùng llama.cpp bản có CUDA)

---

## 📦 2. Cài đặt môi trường

Clone hoặc tải project về:
```bash
git clone https://github.com/<your-repo>/flask-langchain-llm.git
cd flask-langchain-llm
Windows:: .venv\Scripts\activate
macOS / Linux:: source venv/bin/activate
pip install -r requirements.txt
python app.py
