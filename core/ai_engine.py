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
- Tất cả trường điểm phải là số nguyên (không phải chuỗi)
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
  "Diem": 42,
  "DiemChiTiet": {
    "KyNangKyThuat": {"diem": 55, "nhanxet": "Nhận xét ngắn về kỹ năng kỹ thuật"},
    "KinhNghiemLV": {"diem": 15, "nhanxet": "Chưa có kinh nghiệm làm việc chính thức"},
    "DuAnThucTe": {"diem": 40, "nhanxet": "Nhận xét ngắn về dự án thực tế"},
    "HocVanBangCap": {"diem": 65, "nhanxet": "Nhận xét ngắn về học vấn"},
    "ChungChiNgoaiNgu": {"diem": 15, "nhanxet": "Không có chứng chỉ chuyên môn"},
    "KyNangMem": {"diem": 35, "nhanxet": "Nhận xét ngắn về kỹ năng mềm & hoạt động"}
  },
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

TIÊU CHÍ CHẤM ĐIỂM CHI TIẾT — CHẤM NGHIÊM KHẮC NHƯ NHÀ TUYỂN DỤNG THỰC TẾ (mỗi tiêu chí 0-100):

1. "KyNangKyThuat" (25%):
   - 80-100: Nhiều kỹ năng chuyên sâu, hot skill (AI/ML, Cloud, DevOps), có bằng chứng thực hành
   - 60-79: Kỹ năng ổn nhưng chưa chuyên sâu hoặc thiếu hot skill
   - 40-59: Chỉ có kỹ năng cơ bản, chưa nổi bật
   - 0-39: Rất ít kỹ năng hoặc chỉ liệt kê không có bằng chứng

2. "KinhNghiemLV" (25%):
   - 80-100: 3+ năm kinh nghiệm thực tế, vị trí tốt, có thăng tiến
   - 60-79: 1-3 năm kinh nghiệm thực tế
   - 30-50: Chỉ có thực tập hoặc part-time
   - 0-29: CHƯA CÓ kinh nghiệm làm việc → BẮT BUỘC cho 0-20 điểm, KHÔNG ĐƯỢC cho cao hơn

3. "DuAnThucTe" (15%):
   - 80-100: 3+ dự án chất lượng, có link github/demo, công nghệ phức tạp, kết quả rõ ràng
   - 60-79: 1-2 dự án tốt với mô tả chi tiết
   - 30-50: Chỉ có bài tập lớn/đồ án môn học cơ bản
   - 0-29: KHÔNG CÓ dự án nào → BẮT BUỘC cho 0-15 điểm

4. "HocVanBangCap" (15%):
   - 80-100: Thạc sĩ/Tiến sĩ ngành liên quan, GPA > 3.5
   - 60-79: Đại học ngành liên quan, GPA 3.0-3.5
   - 40-59: Đại học ngành liên quan, GPA < 3.0 hoặc không rõ GPA
   - 20-39: Cao đẳng hoặc ngành không liên quan

5. "ChungChiNgoaiNgu" (10%):
   - 80-100: Có chứng chỉ quốc tế uy tín (AWS, GCP, IELTS 7+, TOEIC 800+)
   - 50-70: Có 1-2 chứng chỉ cơ bản
   - 10-30: KHÔNG CÓ chứng chỉ nào → BẮT BUỘC cho 10-20 điểm, KHÔNG cho cao hơn

6. "KyNangMem" (10%):
   - 80-100: Nhiều hoạt động ngoại khóa, cuộc thi, vai trò lãnh đạo
   - 50-70: Có tham gia 1-2 hoạt động
   - 10-30: Không có hoạt động nào nổi bật

NGUYÊN TẮC CHẤM QUAN TRỌNG:
- Sinh viên mới ra trường, chưa có kinh nghiệm: tổng điểm KHÔNG NÊN vượt quá 45-55
- Chỉ có thực tập, chưa đi làm chính thức: KinhNghiemLV tối đa 30-50
- Không có chứng chỉ = ChungChiNgoaiNgu tối đa 20
- Không có dự án = DuAnThucTe tối đa 15
- Ứng viên xuất sắc (80+): phải có kinh nghiệm thực tế + dự án nổi bật + chứng chỉ tốt
- KHÔNG BAO GIỜ cho điểm "làm tròn" hay "ước chừng cao". Chấm thực tế, khách quan

"Diem" = KyNangKyThuat*0.25 + KinhNghiemLV*0.25 + DuAnThucTe*0.15 + HocVanBangCap*0.15 + ChungChiNgoaiNgu*0.1 + KyNangMem*0.1
Làm tròn thành số nguyên."""

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
            if "DiemChiTiet" not in result:
                result["DiemChiTiet"] = {}
            # Ensure all 6 sub-scores exist
            default_subs = ["KyNangKyThuat", "KinhNghiemLV", "DuAnThucTe", "HocVanBangCap",
                            "ChungChiNgoaiNgu", "KyNangMem"]
            for sub in default_subs:
                if sub not in result["DiemChiTiet"]:
                    result["DiemChiTiet"][sub] = {"diem": 50, "nhanxet": "Không có dữ liệu"}
                elif isinstance(result["DiemChiTiet"][sub], dict):
                    if isinstance(result["DiemChiTiet"][sub].get("diem"), str):
                        try:
                            result["DiemChiTiet"][sub]["diem"] = int(result["DiemChiTiet"][sub]["diem"])
                        except (ValueError, TypeError):
                            result["DiemChiTiet"][sub]["diem"] = 50
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
- Tất cả trường điểm phải là số nguyên (không phải chuỗi)

Cấu trúc JSON bắt buộc:
{
  "PhuHop": 75,
  "DiemJD": {
    "KyNangPhuHopJD": {"diem": 80, "nhanxet": "Nhận xét về mức độ kỹ năng đáp ứng JD"},
    "KinhNghiemPhuHopJD": {"diem": 60, "nhanxet": "Nhận xét về kinh nghiệm so với yêu cầu JD"},
    "DuAnLienQuan": {"diem": 70, "nhanxet": "Nhận xét về dự án liên quan đến JD"},
    "HocVanChungChiYC": {"diem": 75, "nhanxet": "Nhận xét về bằng cấp/chứng chỉ so với yêu cầu"},
    "NgoaiNguYeuCau": {"diem": 65, "nhanxet": "Nhận xét về ngoại ngữ & yêu cầu khác"},
    "TiemNangPhatTrien": {"diem": 70, "nhanxet": "Nhận xét về khả năng học hỏi, bổ sung kỹ năng thiếu"}
  },
  "KyNangPhuHop": ["Kỹ năng phù hợp 1", "Kỹ năng phù hợp 2"],
  "KyNangThieu": ["Kỹ năng còn thiếu 1", "Kỹ năng còn thiếu 2"],
  "NhanXet": "Nhận xét tổng quan và đề xuất cho nhà tuyển dụng"
}

TIÊU CHÍ CHẤM ĐIỂM PHÙ HỢP JD (mỗi tiêu chí 0-100):

1. "KyNangPhuHopJD" (25%): % kỹ năng yêu cầu trong JD mà ứng viên đã có
2. "KinhNghiemPhuHopJD" (25%): Số năm kinh nghiệm và lĩnh vực có đáp ứng JD không
3. "DuAnLienQuan" (15%): Dự án có công nghệ/lĩnh vực trùng với yêu cầu JD
4. "HocVanChungChiYC" (15%): Đáp ứng yêu cầu bằng cấp, chứng chỉ bắt buộc trong JD
5. "NgoaiNguYeuCau" (10%): Đáp ứng yêu cầu ngoại ngữ, địa điểm, sẵn sàng đi công tác
6. "TiemNangPhatTrien" (10%): Khả năng học hỏi nhanh, bổ sung kỹ năng còn thiếu

"PhuHop" = Tổng điểm gia quyền = KyNangPhuHopJD*0.25 + KinhNghiemPhuHopJD*0.25 + DuAnLienQuan*0.15 + HocVanChungChiYC*0.15 + NgoaiNguYeuCau*0.1 + TiemNangPhatTrien*0.1
Làm tròn thành số nguyên."""

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
            if "DiemJD" not in result:
                result["DiemJD"] = {}
            default_jd_subs = ["KyNangPhuHopJD", "KinhNghiemPhuHopJD", "DuAnLienQuan",
                               "HocVanChungChiYC", "NgoaiNguYeuCau", "TiemNangPhatTrien"]
            for sub in default_jd_subs:
                if sub not in result["DiemJD"]:
                    result["DiemJD"][sub] = {"diem": 50, "nhanxet": "Không có dữ liệu"}
                elif isinstance(result["DiemJD"][sub], dict):
                    if isinstance(result["DiemJD"][sub].get("diem"), str):
                        try:
                            result["DiemJD"][sub]["diem"] = int(result["DiemJD"][sub]["diem"])
                        except (ValueError, TypeError):
                            result["DiemJD"][sub]["diem"] = 50
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
        # Build structured candidate profile (no raw CV to reduce noise)
        work_exp = cv_result.get('KinhNghiemLamViec', [])
        work_lines = []
        for exp in work_exp:
            work_lines.append(f"- {exp.get('ViTri', '?')} tại {exp.get('CongTy', '?')} ({exp.get('ThoiGian', '?')}): {exp.get('MoTa', '')}")
        work_text = "\n".join(work_lines) if work_lines else "Chưa có kinh nghiệm làm việc chính thức"

        projects = cv_result.get('DuAn', [])
        project_lines = []
        for p in projects:
            techs = ', '.join(p.get('CongNghe', []))
            project_lines.append(f"- {p.get('TenDuAn', '?')} (Vai trò: {p.get('VaiTro', '?')}): {p.get('MoTa', '')} | Công nghệ: {techs}")
        project_text = "\n".join(project_lines) if project_lines else "Không có dự án"

        skills = ', '.join(cv_result.get('KyNang', []))
        certs = ', '.join(cv_result.get('ChungChi', []))

        candidate_profile = f"""HỌ TÊN: {cv_result.get('HoTen', 'N/A')}
HỌC VẤN: {cv_result.get('HocVan', 'N/A')} | GPA: {cv_result.get('GPA', 'N/A')}
KINH NGHIỆM: {cv_result.get('KinhNghiem', 'N/A')}
ĐIỂM ĐÁNH GIÁ CV: {cv_result.get('Diem', 'N/A')}/100
KỸ NĂNG: {skills}
CHỨNG CHỈ: {certs if certs else 'Không có'}
TÓM TẮT: {cv_result.get('TomTat', 'N/A')}

KINH NGHIỆM LÀM VIỆC:
{work_text}

DỰ ÁN:
{project_text}"""

        system_prompt = f"""Bạn là chuyên gia tư vấn nhân sự cấp cao, chuyên tuyển dụng ngành công nghệ tại Việt Nam.

HỒ SƠ ỨNG VIÊN ĐANG PHÂN TÍCH:
{candidate_profile}

QUY TẮC BẮT BUỘC:
1. ĐỌC KỸ câu hỏi của người dùng, TRẢ LỜI ĐÚNG TRỌNG TÂM câu hỏi đó
2. KHÔNG BAO GIỜ copy/paste nội dung CV vào câu trả lời. Hãy PHÂN TÍCH và TƯ VẤN
3. Trả lời ngắn gọn, có cấu trúc (bullet points), đi thẳng vào vấn đề
4. Khi hỏi "Ứng viên phù hợp vị trí nào?" → phân tích kỹ năng + kinh nghiệm → gợi ý 2-3 vị trí cụ thể với lý do
5. Khi hỏi "Điểm mạnh?" → liệt kê 3-4 điểm mạnh nổi bật, mỗi điểm giải thích ngắn gọn
6. Khi hỏi "Cần cải thiện gì?" → chỉ ra 2-3 điểm yếu cụ thể và cách khắc phục
7. Khi hỏi về lương → tham khảo thị trường IT Việt Nam 2024-2025:
   - Thực tập sinh: 3-8 triệu/tháng
   - Fresher (0-1 năm): 8-15 triệu/tháng
   - Junior (1-2 năm): 12-25 triệu/tháng
   - Mid-level (2-4 năm): 20-40 triệu/tháng
   - Senior (5+ năm): 35-70 triệu/tháng
   Điều chỉnh theo kỹ năng, lĩnh vực (AI/ML thường cao hơn 20-30%), và địa điểm
8. Luôn trả lời bằng tiếng Việt
9. KHÔNG trả về JSON hay code, chỉ văn bản thuần"""

        messages = [{"role": "system", "content": system_prompt}]

        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_question})

        return _call_groq(api_key, messages, json_mode=False, temperature=0.3, max_tokens=1500)

    except Exception as e:
        return f"Lỗi: {str(e)}"
