# TalentLens AI — CV Parser Pro

Hệ thống phân tích CV thông minh bằng AI, hỗ trợ nhà tuyển dụng trích xuất thông tin, chấm điểm ứng viên, so sánh với Job Description và hỏi đáp trực tiếp về CV.

## Tính năng chính

### 1. Phân tích CV đơn lẻ
- Upload file CV (PDF / DOCX)
- AI tự động trích xuất: họ tên, email, SĐT, học vấn, kinh nghiệm, kỹ năng, chứng chỉ, dự án
- Chấm điểm ứng viên 0–100 theo tiêu chí: Kỹ năng (30%), Kinh nghiệm (30%), Học vấn (20%), Chứng chỉ & Dự án (20%)
- Xuất báo cáo CSV

### 2. So sánh CV với Job Description
- Nhập JD vào sidebar, hệ thống tự động so sánh khi phân tích CV
- Hiển thị % phù hợp, kỹ năng đã có, kỹ năng còn thiếu
- Nhận xét tổng quan cho nhà tuyển dụng

### 3. Batch Processing
- Upload nhiều CV cùng lúc
- Phân tích hàng loạt với progress bar
- Bảng xếp hạng ứng viên theo điểm từ cao xuống thấp
- Xem chi tiết từng ứng viên

### 4. Chatbot Q&A
- Hỏi đáp trực tiếp về ứng viên sau khi đã phân tích CV
- Gợi ý câu hỏi nhanh: "Ứng viên phù hợp vị trí nào?", "Điểm mạnh?", "Cần cải thiện gì?"
- Lưu lịch sử hội thoại trong phiên làm việc

### 5. Keyword Extraction (Logic ẩn)
- Trích xuất keyword từ CV bằng TF-IDF (Scikit-learn)
- So sánh keyword CV vs JD bằng set intersection
- Logic chạy ngầm, không hiển thị trên giao diện

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Frontend | Streamlit |
| AI / LLM | Groq API (LLaMA 3.3 70B, LLaMA 3.1 8B, LLaMA 4 Scout, Qwen3 32B) |
| Trích xuất PDF | pdfplumber |
| Trích xuất DOCX | python-docx |
| NLP / Keyword | Scikit-learn (TF-IDF) |
| Dữ liệu | Pandas, openpyxl |

## Cấu trúc dự án

```
TalentLens-AI/
├── app.py                      # Giao diện Streamlit (3 tabs)
├── core/
│   ├── __init__.py
│   ├── ai_engine.py            # Xử lý AI: parse CV, so sánh JD, chatbot
│   ├── extractor.py            # Trích xuất text từ PDF/DOCX
│   └── keyword_extractor.py    # TF-IDF keyword extraction
├── requirements.txt
└── README.md
```

## Cài đặt & Chạy

```bash
# Clone project
git clone <repo-url>
cd TalentLens-AI

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy ứng dụng
streamlit run app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`.

## Hướng dẫn sử dụng

1. **Cài đặt API Key**: Mở sidebar > "Cài đặt API" > nhập Groq API Key (đã có key mặc định)
2. **Phân tích CV**: Tab "Phân tích CV" > Upload file PDF/DOCX > chờ AI xử lý
3. **So sánh JD**: Nhập Job Description vào sidebar trước khi upload CV
4. **Batch Processing**: Tab "Batch Processing" > Upload nhiều file > bấm "Phân tích tất cả"
5. **Chatbot**: Tab "Chatbot" > hỏi đáp về ứng viên (cần phân tích CV trước)

## AI Model Fallback

Hệ thống sử dụng chuỗi model dự phòng qua Groq API. Nếu model đầu tiên lỗi (rate limit, quá tải), tự động chuyển sang model tiếp theo:

1. `llama-3.3-70b-versatile` (ưu tiên cao nhất)
2. `llama-3.1-8b-instant`
3. `meta-llama/llama-4-scout-17b-16e-instruct`
4. `qwen/qwen3-32b`

## Yêu cầu hệ thống

- Python 3.9+
- Groq API Key (miễn phí tại [console.groq.com](https://console.groq.com))
- Kết nối internet

## Tác giả

Dự án thực tập sinh AI — Global AI Platform
