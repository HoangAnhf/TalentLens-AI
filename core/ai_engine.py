from groq import Groq
import json

MODELS_TO_TRY = [
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'qwen/qwen3-32b'
]


def _call_groq(api_key, messages, json_mode=True, temperature=0.0, max_tokens=2048):
    client = Groq(api_key=api_key)
    kwargs = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error = ""
    for model_id in MODELS_TO_TRY:
        try:
            response = client.chat.completions.create(model=model_id, **kwargs)
            content = response.choices[0].message.content
            if json_mode:
                return json.loads(content)
            return content
        except Exception as e:
            last_error = f"[{model_id}] {str(e)}"
            continue

    if json_mode:
        return {"error": f"Tất cả model đều báo lỗi. Lỗi cuối: {last_error}"}
    return f"Lỗi: Không thể kết nối AI. {last_error}"


def process_cv_with_ai(api_key, raw_text):
    try:
        system_prompt = """Bạn là hệ thống phân tích CV chuyên nghiệp. Bạn CHỈ trả về JSON thuần túy, KHÔNG kèm bất kỳ văn bản giải thích, markdown, hay nội dung CV nào.

QUAN TRỌNG:
- KHÔNG lặp lại nội dung CV trong response
- KHÔNG thêm bất kỳ text nào ngoài JSON
- Response phải bắt đầu bằng { và kết thúc bằng }
- Trường "Diem" phải là số nguyên (không phải chuỗi)
- Trường "KinhNghiem" phải là chuỗi

Cấu trúc JSON bắt buộc:
{
  "HoTen": "Tên đầy đủ",
  "KinhNghiem": "Số năm kinh nghiệm",
  "HocVan": "Trình độ học vấn",
  "GPA": "Điểm GPA hoặc N/A",
  "Email": "Email",
  "DienThoai": "Số điện thoại",
  "DiaChi": "Địa chỉ",
  "TomTat": "Tóm tắt ngắn gọn năng lực",
  "Diem": 75,
  "ChiTietDiem": "Lý do chấm điểm ngắn gọn",
  "KyNang": ["Kỹ năng 1", "Kỹ năng 2"],
  "ChungChi": ["Chứng chỉ 1"],
  "KinhNghiemLamViec": [
    {
      "CongTy": "Tên công ty",
      "ViTri": "Vị trí",
      "ThoiGian": "Thời gian",
      "MoTa": "Mô tả ngắn"
    }
  ],
  "DuAn": [
    {
      "TenDuAn": "Tên dự án",
      "VaiTro": "Vai trò trong dự án",
      "MucDich": "Mục đích / mục tiêu dự án",
      "CongNghe": ["Công nghệ 1", "Công nghệ 2"],
      "MoTa": "Mô tả ngắn gọn về dự án và kết quả đạt được",
      "Link": "Link demo/github nếu có, không có thì null"
    }
  ]
}

Tiêu chí chấm điểm "Diem" (0-100):
- Kỹ năng: 30%, Kinh nghiệm: 30%, Học vấn: 20%, Chứng chỉ & dự án: 20%
- 0-40: Yếu, 41-60: Trung bình, 61-80: Khá, 81-100: Xuất sắc"""

        user_prompt = f"Phân tích CV sau và trả về JSON:\n\n{raw_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = _call_groq(api_key, messages, json_mode=True)

        if "error" not in result:
            if isinstance(result.get("Diem"), str):
                try:
                    result["Diem"] = int(result["Diem"])
                except (ValueError, TypeError):
                    result["Diem"] = 50

            for field in ["HoTen", "KinhNghiem", "HocVan", "GPA", "Email", "DienThoai", "DiaChi", "TomTat"]:
                if field not in result:
                    result[field] = "N/A"
            if "Diem" not in result:
                result["Diem"] = 50
            if "KyNang" not in result:
                result["KyNang"] = []
            if "ChungChi" not in result:
                result["ChungChi"] = []
            if "KinhNghiemLamViec" not in result:
                result["KinhNghiemLamViec"] = []
            if "DuAn" not in result:
                result["DuAn"] = []

        return result

    except Exception as e:
        return {"error": str(e)}


def compare_cv_with_jd(api_key, cv_text, jd_text):
    try:
        system_prompt = """Bạn là hệ thống so sánh CV với Job Description. Bạn CHỈ trả về JSON thuần túy.

QUAN TRỌNG:
- KHÔNG lặp lại nội dung CV hay JD trong response
- KHÔNG thêm bất kỳ text nào ngoài JSON
- Response phải bắt đầu bằng { và kết thúc bằng }
- Trường "PhuHop" phải là số nguyên (không phải chuỗi)

Cấu trúc JSON bắt buộc:
{
  "PhuHop": 75,
  "KyNangPhuHop": ["Kỹ năng phù hợp 1", "Kỹ năng phù hợp 2"],
  "KyNangThieu": ["Kỹ năng còn thiếu 1", "Kỹ năng còn thiếu 2"],
  "NhanXet": "Nhận xét tổng quan và đề xuất cho nhà tuyển dụng"
}

Lưu ý:
- "PhuHop": số nguyên 0-100, mức độ phù hợp tổng thể
- "KyNangPhuHop": kỹ năng/yêu cầu mà ứng viên ĐÃ CÓ
- "KyNangThieu": kỹ năng/yêu cầu mà ứng viên CHƯA CÓ
- "NhanXet": nhận xét ngắn gọn cho nhà tuyển dụng"""

        user_prompt = f"""So sánh CV với JD sau và trả về JSON:

=== CV ===
{cv_text}

=== JOB DESCRIPTION ===
{jd_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = _call_groq(api_key, messages, json_mode=True)

        if "error" not in result:
            if isinstance(result.get("PhuHop"), str):
                try:
                    result["PhuHop"] = int(result["PhuHop"])
                except (ValueError, TypeError):
                    result["PhuHop"] = 50
            if "KyNangPhuHop" not in result:
                result["KyNangPhuHop"] = []
            if "KyNangThieu" not in result:
                result["KyNangThieu"] = []
            if "NhanXet" not in result:
                result["NhanXet"] = "Không có nhận xét."

        return result

    except Exception as e:
        return {"error": str(e)}


def chat_about_cv(api_key, cv_text, cv_result, chat_history, user_question):
    try:
        system_prompt = """Bạn là chuyên gia tư vấn nhân sự (HR Consultant) cấp cao với hơn 10 năm kinh nghiệm tuyển dụng trong ngành công nghệ tại Việt Nam.

NHIỆM VỤ:
- Trả lời câu hỏi về ứng viên dựa trên CV và kết quả phân tích đã cung cấp
- Đưa ra nhận xét sâu sắc, có phân tích, có lập luận — KHÔNG trả lời máy móc hay liệt kê lại CV
- Khi được hỏi về lương, hãy dựa vào kiến thức thị trường IT Việt Nam để đưa ra mức lương tham khảo hợp lý (ví dụ: thực tập AI/ML tại Hà Nội/HCM thường 5-15 triệu/tháng tùy năng lực)
- Khi được hỏi về vị trí phù hợp, hãy phân tích kỹ năng + kinh nghiệm để gợi ý cụ thể
- Khi được hỏi về điểm mạnh/yếu, hãy đánh giá khách quan và đưa ra lời khuyên thực tế

PHONG CÁCH TRẢ LỜI:
- Trả lời bằng tiếng Việt, tự nhiên như đang tư vấn trực tiếp
- Có cấu trúc rõ ràng (dùng gạch đầu dòng nếu cần)
- Đưa ra con số cụ thể khi được hỏi (mức lương, % phù hợp, số năm cần bổ sung...)
- KHÔNG lặp lại nội dung CV nguyên văn, hãy phân tích và tổng hợp
- KHÔNG trả về JSON, chỉ trả về văn bản thuần
- Nếu không chắc chắn, nói rõ đây là ước tính dựa trên thị trường chung"""

        work_exp = cv_result.get('KinhNghiemLamViec', [])
        work_summary = ""
        for exp in work_exp:
            work_summary += f"\n  + {exp.get('ViTri', '')} tại {exp.get('CongTy', '')} ({exp.get('ThoiGian', '')})"
        if not work_summary:
            work_summary = "\n  Chưa có kinh nghiệm làm việc"

        projects = cv_result.get('DuAn', [])
        project_summary = ""
        for p in projects:
            techs = ', '.join(p.get('CongNghe', []))
            project_summary += f"\n  + {p.get('TenDuAn', '')} ({p.get('VaiTro', '')}) — Công nghệ: {techs}"
        if not project_summary:
            project_summary = "\n  Không có thông tin dự án"

        context = f"""=== HỒ SƠ ỨNG VIÊN ===
Họ tên: {cv_result.get('HoTen', 'N/A')}
Học vấn: {cv_result.get('HocVan', 'N/A')} | GPA: {cv_result.get('GPA', 'N/A')}
Kinh nghiệm: {cv_result.get('KinhNghiem', 'N/A')}
Điểm đánh giá: {cv_result.get('Diem', 'N/A')}/100
Kỹ năng: {', '.join(cv_result.get('KyNang', []))}
Chứng chỉ: {', '.join(cv_result.get('ChungChi', []))}
Kinh nghiệm làm việc:{work_summary}
Dự án:{project_summary}
Tóm tắt: {cv_result.get('TomTat', 'N/A')}

=== NỘI DUNG CV GỐC ===
{cv_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Đây là thông tin ứng viên tôi cần bạn tư vấn:\n\n{context}"},
            {"role": "assistant", "content": f"Tôi đã nắm rõ hồ sơ của ứng viên {cv_result.get('HoTen', '')}. Bạn muốn hỏi gì về ứng viên này?"}
        ]

        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_question})

        return _call_groq(api_key, messages, json_mode=False, temperature=0.5, max_tokens=1500)

    except Exception as e:
        return f"Lỗi: {str(e)}"
