import streamlit as st
import pandas as pd
import time
import base64
from core.extractor import extract_text
from core.ai_engine import process_cv_with_ai, compare_cv_with_jd, chat_about_cv
from core.keyword_extractor import extract_keywords_tfidf, compare_keywords

# --- SESSION STATE INITIALIZATION ---
# Initialize session state variables to store data across reruns
if "parsed_cv_data" not in st.session_state:
    st.session_state.parsed_cv_data = None  # Stores data of the single parsed CV
if "job_description" not in st.session_state:
    st.session_state.job_description = ""  # Stores the Job Description input from sidebar
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores the chatbot conversation history
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None  # Stores results from batch processing
if "last_batch_files" not in st.session_state:
    st.session_state.last_batch_files = set()  # To detect changes in uploaded batch files
if "raw_cv_text" not in st.session_state:
    st.session_state.raw_cv_text = ""  # Raw text extracted from CV file
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# --- UI/UX SPECIFICATION (CSS and HTML Components) ---

# Define global CSS styles based on the provided UI/UX specification.
# These styles are injected into the Streamlit app using st.markdown with unsafe_allow_html=True.
CSS_STYLES = """
<style>
    /* Global Styles: Defines consistent color palette, typography, and common sizing/spacing. */
    :root {
        --bg-light-gray: rgb(245, 246, 250); /* Neutral light grey for page background */
        --card-bg: white; /* White for card and container backgrounds */
        --border-light: #e0e0e0; /* Light grey for borders */
        --accent-blue: #2980b9; /* Primary accent blue for professional elements */
        --accent-blue-light: #e0f2f7; /* Lighter shade of accent blue */
        --text-dark: #333333; /* Dark grey for main text, high contrast */
        --text-medium: #666666; /* Medium grey for labels, captions */
        --text-light: #999999; /* Light grey for subtle text like empty states */
        --green-success: #27ae60; /* Green for success states, high scores */
        --green-success-light: #e6faed; /* Light green for badges/backgrounds */
        --red-error: #c0392b; /* Red for error states, low scores */
        --red-error-light: #fdecec; /* Light red for badges/backgrounds */
        --orange-warning: #e67e22; /* Orange for warning states, average scores */
        --orange-warning-light: #fff0e0; /* Light orange for badges/backgrounds */
        --font-family: 'Inter', 'IBM Plex Sans', system-ui, sans-serif; /* Clean, readable fonts */
        --border-radius-card: 8px; /* Standard border-radius for cards */
        --border-radius-badge: 6px; /* Standard border-radius for badges/buttons */
        --shadow-card: 0 1px 3px rgba(0,0,0,0.05); /* Subtle shadow for cards */
    }

    body {
        font-family: var(--font-family);
        color: var(--text-dark);
    }

    /* Streamlit Overrides: Customizes default Streamlit components to match the design. */
    .stApp {
        background-color: var(--bg-light-gray); /* Page background */
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--card-bg); /* White sidebar background */
        box-shadow: var(--shadow-card);
    }
    [data-testid="stSidebar"] h1 { /* "CV Parser Pro" title in sidebar */
        font-weight: 700;
        font-size: 1.5rem;
        color: var(--text-dark);
        margin-bottom: 0.25rem;
    }
    [data-testid="stSidebar"] p { /* Subtitle & footer text in sidebar */
        font-size: 0.9rem;
        color: var(--text-medium);
        margin-top: 0;
    }
    .sidebar-divider { /* Horizontal separator in sidebar */
        height: 1px;
        background-color: var(--border-light);
        margin: 1rem 0;
    }
    .sidebar-status-badge { /* "Hệ thống sẵn sàng" badge */
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        background-color: var(--green-success-light);
        color: var(--green-success);
        border-radius: var(--border-radius-badge);
        font-size: 0.85rem;
        font-weight: 600;
        width: fit-content;
        margin-bottom: 1rem;
    }
    .sidebar-status-dot { /* Small green dot for status indicator */
        width: 8px;
        height: 8px;
        background-color: var(--green-success);
        border-radius: 50%;
    }
    .sidebar-label { /* Custom labels for inputs (e.g., Job Description) */
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-medium);
        margin-bottom: 0.5rem;
        display: block;
        font-weight: 600;
    }
    /* Specific Streamlit input overrides within sidebar */
    [data-testid="stSidebar"] .stTextArea label { display: none; } /* Hide default label */
    [data-testid="stSidebar"] .stTextArea { margin-top: 1rem; }
    [data-testid="stSidebar"] .stTextArea > div > label { display: block; margin-bottom: 0.5rem; }
    [data-testid="stSidebar"] .stTextArea textarea,
    [data-testid="stSidebar"] .stTextInput input {
        background-color: var(--bg-light-gray);
        border: 1px solid var(--border-light);
        border-radius: var(--border-radius-badge);
        color: var(--text-dark);
        font-family: var(--font-family);
    }


    /* Main Content Area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Card Styling: General style for all content cards */
    .card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-light);
        border-radius: var(--border-radius-card);
        box-shadow: var(--shadow-card);
        padding: 1.5rem;
        margin-bottom: 1.25rem;
    }
    .card-title { /* Title within each card */
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-dark);
        margin-bottom: 1rem;
    }

    /* Header Styling */
    .header-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.5rem;
        background-color: var(--card-bg);
        border: 1px solid var(--border-light);
        border-radius: var(--border-radius-card);
        box-shadow: var(--shadow-card);
        margin-bottom: 1.5rem;
    }
    .header-icon-box { /* Icon box in header */
        width: 48px;
        height: 48px;
        background-color: var(--accent-blue-light);
        border-radius: var(--border-radius-badge);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.75rem;
        color: var(--accent-blue);
    }
    .header-title-main { /* "CV Parser Pro" title in header */
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-dark);
        margin: 0;
        line-height: 1.2;
    }
    .header-subtitle { /* Subtitle in header */
        font-size: 0.9rem;
        color: var(--text-medium);
        margin: 0;
        line-height: 1.2;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem; /* Space between tabs */
        margin-bottom: 1.5rem;
        background-color: var(--card-bg);
        border-radius: var(--border-radius-card);
        padding: 0.5rem;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-card);
    }
    .stTabs [data-baseweb="tab"] { /* Individual tab button */
        border-radius: var(--border-radius-card);
        padding: 0.75rem 1.25rem;
        font-weight: 600;
        font-size: 1rem;
        color: var(--text-medium);
        background-color: transparent;
        transition: all 0.2s ease-in-out;
        flex: 1; /* Make tabs equally wide */
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    .stTabs [aria-selected="true"] { /* Active tab styling */
        background-color: var(--accent-blue-light);
        color: var(--accent-blue) !important;
        border-color: transparent !important;
        box-shadow: none;
    }
    .stTabs [aria-selected="false"]:hover { /* Hover effect for inactive tabs */
        background-color: var(--bg-light-gray);
        color: var(--text-dark);
    }
    .stTabs [data-baseweb="tab"] > div > div { /* Target text inside tab for padding adjustment */
        padding: 0 !important;
    }

    /* Empty State Styling */
    .empty-state-card {
        text-align: center;
        padding: 3rem 1.5rem;
        background-color: var(--card-bg);
        border: 1px solid var(--border-light);
        border-radius: var(--border-radius-card);
        box-shadow: var(--shadow-card);
        margin-top: 2rem;
    }
    .empty-state-icon { /* Large icon in empty state */
        font-size: 4rem;
        color: var(--text-light);
        margin-bottom: 1rem;
    }
    .empty-state-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-dark);
        margin-bottom: 0.5rem;
    }
    .empty-state-subtitle {
        font-size: 1rem;
        color: var(--text-medium);
    }

    /* Score Card Styling */
    .score-container {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .score-value { /* Large score number */
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1;
        display: flex;
        align-items: baseline;
    }
    .score-value small { /* "/ 100 điểm" part of the score */
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-medium);
        margin-left: 5px;
    }
    .score-description { /* Contains progress bar and detailed text */
        flex-grow: 1;
    }
    .progress-bar-container { /* Outer container for progress bar */
        width: 100%;
        height: 10px;
        background-color: var(--bg-light-gray);
        border-radius: 5px;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }
    .progress-bar-fill { /* Inner fill of the progress bar, color changes dynamically */
        height: 100%;
        width: 0%; /* Will be set dynamically by Python */
        border-radius: 5px;
        transition: width 0.5s ease-in-out;
    }
    .score-text-detail { /* Text describing the score */
        font-size: 0.9rem;
        color: var(--text-medium);
    }

    /* Conditional Score Colors: Applied to score value and progress bar fill */
    .score-color-green { color: var(--green-success); }
    .score-color-blue { color: var(--accent-blue); }
    .score-color-orange { color: var(--orange-warning); }
    .score-color-red { color: var(--red-error); }
    .progress-fill-green { background-color: var(--green-success); }
    .progress-fill-blue { background-color: var(--accent-blue); }
    .progress-fill-orange { background-color: var(--orange-warning); }
    .progress-fill-red { background-color: var(--red-error); }

    /* Info Cells: For displaying candidate details in a grid layout */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
    }
    .info-cell {
        background-color: var(--bg-light-gray);
        border-radius: var(--border-radius-badge);
        padding: 1rem 1.25rem;
        display: flex;
        flex-direction: column;
    }
    .info-label { /* Label for each info cell (e.g., "HỌ VÀ TÊN") */
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-medium);
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    .info-value { /* Actual data value in each info cell */
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-dark);
        word-break: break-all;
    }

    /* Badges / Tags Styling */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px; /* Margin between badges */
        margin-top: 0.5rem;
    }
    .badge { /* Base style for all badges */
        display: inline-flex;
        align-items: center;
        padding: 5px 12px;
        border-radius: var(--border-radius-badge);
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid;
        white-space: nowrap;
    }
    /* Specific badge types as per specification */
    .badge-default { /* For general skills, certifications */
        background-color: var(--bg-light-gray);
        color: var(--text-dark);
        border-color: var(--border-light);
    }
    .badge-green { /* For matching skills, success indicators */
        background-color: var(--green-success-light);
        color: var(--green-success);
        border-color: var(--green-success);
    }
    .badge-red { /* For missing skills, error indicators */
        background-color: var(--red-error-light);
        color: var(--red-error);
        border-color: var(--red-error);
    }
    .badge-blue { /* For project technologies, highlights */
        background-color: var(--accent-blue-light);
        color: var(--accent-blue);
        border-color: var(--accent-blue);
    }

    /* Expander Overrides: Customizes Streamlit's native expander component */
    .streamlit-expanderHeader {
        background-color: var(--bg-light-gray) !important;
        border-radius: var(--border-radius-badge) !important;
        border: 1px solid var(--border-light) !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.5rem;
        font-weight: 600 !important;
        color: var(--text-dark) !important;
    }
    .streamlit-expanderContent { /* Content area of the expander */
        padding-left: 1rem;
        padding-right: 1rem;
        padding-bottom: 0.5rem;
        border-left: 2px solid var(--border-light);
        margin-left: 0.5rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
    .streamlit-expanderHeader:hover { /* Hover effect for expander header */
        background-color: var(--border-light) !important;
    }

    /* Project details styling within expanders */
    .project-detail-label {
        font-weight: 600;
        color: var(--text-dark);
        margin-top: 1rem;
        margin-bottom: 0.25rem;
        display: block;
    }
    .project-detail-text {
        color: var(--text-dark);
        margin-bottom: 0.5rem;
    }
    .project-detail-link {
        color: var(--accent-blue);
        text-decoration: none;
        font-weight: 500;
    }
    .project-detail-link:hover {
        text-decoration: underline;
    }

    /* Chatbot UI Styling */
    .stChatInputContainer { /* The input bar at the bottom of the chatbot */
        border-top: 1px solid var(--border-light);
        padding-top: 1rem;
        background-color: var(--card-bg);
        margin-left: -1rem;
        margin-right: -1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        position: sticky;
        bottom: 0;
        z-index: 999;
    }
    .stChatInputContainer > div > div > div > div > div > input { /* The actual text input field */
        border-radius: var(--border-radius-card);
        background-color: var(--bg-light-gray);
        border: 1px solid var(--border-light);
    }
    .stChatMessage { /* Individual chat message container */
        background-color: transparent !important;
        padding-left: 0;
        padding-right: 0;
        margin-bottom: 1rem;
    }
    .stChatMessage.st-ai .stChatMessageContent { /* AI message bubble */
        background-color: var(--accent-blue-light) !important;
        border: 1px solid var(--accent-blue) !important;
        color: var(--text-dark) !important;
        border-radius: var(--border-radius-card) !important;
        padding: 1rem !important;
    }
    .stChatMessage.st-user .stChatMessageContent { /* User message bubble */
        background-color: var(--accent-blue) !important;
        color: white !important;
        border-radius: var(--border-radius-card) !important;
        padding: 1rem !important;
    }
    .stChatMessage .stChatMessageAvatar { /* Avatar icon for chat messages */
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        color: white;
        background-color: var(--accent-blue); /* Default for AI */
    }
    .stChatMessage.st-user .stChatMessageAvatar { /* Specific color for user avatar */
        background-color: var(--text-dark);
    }

    /* Chat suggestion buttons: Styling Streamlit's button to match the design */
    .chat-suggestion-button-container .stButton > button {
        background-color: var(--bg-light-gray);
        color: var(--accent-blue);
        border: 1px solid var(--accent-blue);
        border-radius: var(--border-radius-badge);
        font-size: 0.9rem;
        font-weight: 600;
        padding: 0.6rem 1rem;
        width: 100%; /* Make button fill its column */
        margin-bottom: 0;
        transition: all 0.2s ease-in-out;
    }
    .chat-suggestion-button-container .stButton > button:hover {
        background-color: var(--accent-blue);
        color: var(--card-bg);
    }
    h4 { /* General h4 styling for sub-sections */
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-dark);
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    /* Streamlit DataFrame Styling */
    div.stDataFrame > div:first-child {
        border: 1px solid var(--border-light);
        border-radius: var(--border-radius-card);
        overflow: hidden;
    }
    div.stDataFrame thead th {
        background-color: var(--bg-light-gray);
        color: var(--text-dark);
        font-weight: 600;
        border-bottom: 1px solid var(--border-light);
    }
    div.stDataFrame tbody tr:hover {
        background-color: var(--accent-blue-light) !important; /* Row hover effect */
    }
    div.stDataFrame tbody td {
        color: var(--text-dark);
    }

    /* General Button Styling (for primary buttons like "Analyze", "Download") */
    .stButton > button {
        background-color: var(--accent-blue);
        color: white;
        border-radius: var(--border-radius-badge);
        font-weight: 600;
        padding: 0.6rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #2c77a9; /* Slightly darker blue on hover */
        color: white;
    }
    .stDownloadButton > button { /* Styling for Streamlit's download button */
        background-color: var(--accent-blue);
        color: white;
        border-radius: var(--border-radius-badge);
        font-weight: 600;
        padding: 0.6rem 1rem;
        border: none;
    }
    .stDownloadButton > button:hover {
        background-color: #2c77a9;
        color: white;
    }
    .stFileUploader label {
        color: var(--text-medium);
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .stFileUploader > div > button { /* Styling for the "Browse files" button in file_uploader */
        background-color: var(--accent-blue);
        color: white;
        border-radius: var(--border-radius-badge);
        font-weight: 600;
        padding: 0.6rem 1rem;
        border: none;
    }
    .stFileUploader > div > button:hover {
        background-color: #2c77a9;
        color: white;
    }

</style>
"""

# Configure Streamlit page layout to wide mode
st.set_page_config(layout="wide")
# Inject the custom CSS styles into the Streamlit app
st.markdown(CSS_STYLES, unsafe_allow_html=True)


# --- HELPER FUNCTIONS FOR RENDERING HTML COMPONENTS ---

# Function to start a custom HTML card with optional title and class
def render_card_start(title=None, class_name="card"):
    html = f'<div class="{class_name}">'
    if title:
        html += f'<h2 class="card-title">{title}</h2>'
    return html


# Function to close a custom HTML card
def render_card_end():
    return '</div>'


# Function to render a badge (tag) with specified text and type
def render_badge(text, badge_type="default"):
    return f'<span class="badge badge-{badge_type}">{text}</span>'


# Function to determine CSS classes for score value and progress bar fill based on score
def get_score_color_class(score):
    if score >= 81:
        return "score-color-green", "progress-fill-green"
    elif score >= 61:
        return "score-color-blue", "progress-fill-blue"
    elif score >= 41:
        return "score-color-orange", "progress-fill-orange"
    else:
        return "score-color-red", "progress-fill-red"


# --- SIDEBAR CONTENT ---
with st.sidebar:
    # Branding section
    st.markdown(f"<h1>CV Parser Pro</h1><p>Global AI Platform</p>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    # System Status badge
    st.markdown(f'<div class="sidebar-status-badge"><div class="sidebar-status-dot"></div>Hệ thống sẵn sàng</div>',
                unsafe_allow_html=True)

    # API Settings
    api_input = st.text_input("Groq API Key", value=st.session_state.api_key, type="password",
                              placeholder="Nhập API Key tại đây...",
                              help="Lấy API Key miễn phí tại console.groq.com")
    if api_input:
        st.session_state.api_key = api_input

    # Job Description input area
    st.markdown(f'<label class="sidebar-label">Job Description</label>', unsafe_allow_html=True)
    st.session_state.job_description = st.text_area(
        "placeholder for JD",  # This label is hidden by CSS, custom label used above
        height=180,
        placeholder="Dán JD vào đây để so sánh với CV...",
        key="jd_input"
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    # Footer section
    st.markdown(
        f'<p style="text-align: center; font-size: 0.8rem; color: var(--text-light);">v1.0.0 © 2024 Global AI</p>',
        unsafe_allow_html=True)

# --- HEADER CONTENT ---
st.markdown(
    f"""
    <div class="header-container">
        <div class="header-icon-box">📄</div>
        <div>
            <h1 class="header-title-main">CV Parser Pro</h1>
            <p class="header-subtitle">Phân tích CV thông minh bằng AI — Trích xuất, Chấm điểm, So sánh JD</p>
        </div>
    </div>
    """, unsafe_allow_html=True
)

# --- MAIN CONTENT AREA WITH TABS ---
tab1, tab2, tab3 = st.tabs(["📄 Phân tích CV", "📚 Batch Processing", "💬 Chatbot"])

# --- TAB 1: SINGLE CV PARSING ---
with tab1:
    st.markdown(render_card_start(), unsafe_allow_html=True)
    col_upload_left, col_upload_center, col_upload_right = st.columns([1, 2, 1])
    with col_upload_center:
        uploaded_file = st.file_uploader("Chọn file CV (PDF / DOCX)", type=["pdf", "docx"], key="single_cv_uploader")
    st.markdown(render_card_end(), unsafe_allow_html=True)

    # Display empty state if no file is uploaded
    if uploaded_file is None:
        st.markdown(
            f"""
            <div class="empty-state-card">
                <div class="empty-state-icon">📄</div>
                <h3 class="empty-state-title">Tải lên file CV để bắt đầu phân tích</h3>
                <p class="empty-state-subtitle">Hỗ trợ định dạng PDF và DOCX</p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        # Parse CV if a new file is uploaded or no data exists
        if st.session_state.parsed_cv_data is None or st.session_state.parsed_cv_data.get(
                "filename") != uploaded_file.name:
            with st.spinner(f"Đang phân tích CV: {uploaded_file.name}..."):
                raw_text = extract_text(uploaded_file)
                if not raw_text.strip():
                    st.error("Không thể trích xuất nội dung từ file. Vui lòng kiểm tra lại file CV.")
                else:
                    st.session_state.raw_cv_text = raw_text
                    result = process_cv_with_ai(st.session_state.api_key, raw_text)
                    if "error" in result:
                        st.error(f"Lỗi phân tích CV: {result['error']}")
                    else:
                        result["filename"] = uploaded_file.name
                        # JD comparison if JD is provided
                        jd_text = st.session_state.job_description
                        if jd_text and jd_text.strip():
                            jd_result = compare_cv_with_jd(st.session_state.api_key, raw_text, jd_text)
                            if "error" not in jd_result:
                                result["JD_Comparison"] = jd_result
                            else:
                                result["JD_Comparison"] = None
                        else:
                            result["JD_Comparison"] = None
                        # Final score: 100% JD if JD provided, else 100% CV
                        if result["JD_Comparison"] and "PhuHop" in result["JD_Comparison"]:
                            result["DiemTongHop"] = result["JD_Comparison"]["PhuHop"]
                        else:
                            result["DiemTongHop"] = result.get("Diem", 50)
                        st.session_state.parsed_cv_data = result
                        st.session_state.chat_history = []
                        st.success("Phân tích CV hoàn tất!")

        cv_data = st.session_state.parsed_cv_data
        if cv_data:
            cv_score = cv_data.get("Diem", 50)
            has_jd = cv_data.get("JD_Comparison") is not None
            final_score = cv_data.get("DiemTongHop", cv_score)

            score_class, progress_fill_class = get_score_color_class(final_score)
            score_title = "Độ phù hợp với Job Description" if has_jd else "Điểm đánh giá ứng viên"

            # 7.3. Main Score Card
            st.markdown(render_card_start(score_title), unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="score-container">
                    <div class="score-value {score_class}">
                        {final_score}<small>/ 100 điểm</small>
                    </div>
                    <div class="score-description">
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill {progress_fill_class}" style="width: {final_score}%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            st.markdown(render_card_end(), unsafe_allow_html=True)

            # 7.3a. JD Detail (shown FIRST if JD provided, since JD score is the main score)
            if has_jd:
                jd_compare = cv_data["JD_Comparison"]

                # JD skills matched/missing
                st.markdown(render_card_start("Phân tích phù hợp JD"), unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.9rem;color:var(--text-medium);margin-bottom:0.8rem;">{jd_compare.get("NhanXet", "")}</div>', unsafe_allow_html=True)
                col_jd_matched, col_jd_missing = st.columns(2)
                with col_jd_matched:
                    st.markdown("<h4>Kỹ năng phù hợp</h4>", unsafe_allow_html=True)
                    if jd_compare["KyNangPhuHop"]:
                        matched_skills_html = "<div class='badge-container'>"
                        for skill in jd_compare["KyNangPhuHop"]:
                            matched_skills_html += render_badge(skill, "green")
                        matched_skills_html += "</div>"
                        st.markdown(matched_skills_html, unsafe_allow_html=True)
                    else:
                        st.markdown("<p>Không có kỹ năng phù hợp.</p>", unsafe_allow_html=True)
                with col_jd_missing:
                    st.markdown("<h4>Kỹ năng còn thiếu</h4>", unsafe_allow_html=True)
                    if jd_compare["KyNangThieu"]:
                        missing_skills_html = "<div class='badge-container'>"
                        for skill in jd_compare["KyNangThieu"]:
                            missing_skills_html += render_badge(skill, "red")
                        missing_skills_html += "</div>"
                        st.markdown(missing_skills_html, unsafe_allow_html=True)
                    else:
                        st.markdown("<p>Không có kỹ năng còn thiếu.</p>", unsafe_allow_html=True)
                st.markdown(render_card_end(), unsafe_allow_html=True)

                # JD detailed sub-scores
                diem_jd = jd_compare.get("DiemJD", {})
                jd_sub_labels = {
                    "KyNangPhuHopJD": ("Kỹ năng phù hợp JD", "25%"),
                    "KinhNghiemPhuHopJD": ("Kinh nghiệm phù hợp JD", "25%"),
                    "DuAnLienQuan": ("Dự án liên quan", "15%"),
                    "HocVanChungChiYC": ("Học vấn & Chứng chỉ yêu cầu", "15%"),
                    "NgoaiNguYeuCau": ("Ngoại ngữ & Yêu cầu khác", "10%"),
                    "TiemNangPhatTrien": ("Tiềm năng phát triển", "10%"),
                }
                st.markdown(render_card_start(f"Chi tiết điểm phù hợp JD — {jd_compare['PhuHop']}/100"), unsafe_allow_html=True)
                for key, (label, weight) in jd_sub_labels.items():
                    sub = diem_jd.get(key, {"diem": 50, "nhanxet": ""})
                    sub_score = sub.get("diem", 50) if isinstance(sub, dict) else 50
                    sub_note = sub.get("nhanxet", "") if isinstance(sub, dict) else ""
                    sub_class, sub_fill = get_score_color_class(sub_score)
                    st.markdown(
                        f"""
                        <div style="margin-bottom:0.8rem;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;">
                                <span style="font-weight:600;color:var(--text-dark);font-size:0.9rem;">{label} <span style="color:var(--text-medium);font-weight:400;">({weight})</span></span>
                                <span class="{sub_class}" style="font-weight:700;font-size:0.95rem;">{sub_score}</span>
                            </div>
                            <div class="progress-bar-container" style="height:8px;">
                                <div class="progress-bar-fill {sub_fill}" style="width:{sub_score}%;"></div>
                            </div>
                            <div style="font-size:0.8rem;color:var(--text-medium);margin-top:0.2rem;">{sub_note}</div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                st.markdown(render_card_end(), unsafe_allow_html=True)

            # 7.3b. Detailed CV Score Breakdown
            diem_ct = cv_data.get("DiemChiTiet", {})
            cv_sub_labels = {
                "KyNangKyThuat": ("Kỹ năng kỹ thuật", "25%"),
                "KinhNghiemLV": ("Kinh nghiệm làm việc", "25%"),
                "DuAnThucTe": ("Dự án thực tế", "15%"),
                "HocVanBangCap": ("Học vấn & Bằng cấp", "15%"),
                "ChungChiNgoaiNgu": ("Chứng chỉ & Ngoại ngữ", "10%"),
                "KyNangMem": ("Kỹ năng mềm & Hoạt động", "10%"),
            }

            st.markdown(render_card_start(f"Chi tiết điểm CV — {cv_score}/100"), unsafe_allow_html=True)
            for key, (label, weight) in cv_sub_labels.items():
                sub = diem_ct.get(key, {"diem": 50, "nhanxet": ""})
                sub_score = sub.get("diem", 50) if isinstance(sub, dict) else 50
                sub_note = sub.get("nhanxet", "") if isinstance(sub, dict) else ""
                sub_class, sub_fill = get_score_color_class(sub_score)
                st.markdown(
                    f"""
                    <div style="margin-bottom:0.8rem;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;">
                            <span style="font-weight:600;color:var(--text-dark);font-size:0.9rem;">{label} <span style="color:var(--text-medium);font-weight:400;">({weight})</span></span>
                            <span class="{sub_class}" style="font-weight:700;font-size:0.95rem;">{sub_score}</span>
                        </div>
                        <div class="progress-bar-container" style="height:8px;">
                            <div class="progress-bar-fill {sub_fill}" style="width:{sub_score}%;"></div>
                        </div>
                        <div style="font-size:0.8rem;color:var(--text-medium);margin-top:0.2rem;">{sub_note}</div>
                    </div>
                    """, unsafe_allow_html=True
                )
            st.markdown(render_card_end(), unsafe_allow_html=True)

            # 7.4. Candidate Information Card
            st.markdown(render_card_start("Thông tin ứng viên"), unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="info-grid">
                    <div class="info-cell"><div class="info-label">HỌ VÀ TÊN</div><div class="info-value">{cv_data["HoTen"]}</div></div>
                    <div class="info-cell"><div class="info-label">HỌC VẤN</div><div class="info-value">{cv_data["HocVan"]}</div></div>
                    <div class="info-cell"><div class="info-label">KINH NGHIỆM</div><div class="info-value">{cv_data["KinhNghiem"]}</div></div>
                    <div class="info-cell"><div class="info-label">EMAIL</div><div class="info-value">{cv_data["Email"]}</div></div>
                    <div class="info-cell"><div class="info-label">ĐIỆN THOẠI</div><div class="info-value">{cv_data["DienThoai"]}</div></div>
                    <div class="info-cell"><div class="info-label">ĐỊA CHỈ</div><div class="info-value">{cv_data["DiaChi"]}</div></div>
                    <div class="info-cell"><div class="info-label">GPA</div><div class="info-value">{cv_data["GPA"]}</div></div>
                </div>
                """, unsafe_allow_html=True
            )
            st.markdown(render_card_end(), unsafe_allow_html=True)

            # 7.5. Skills and Overall Summary Cards (side-by-side)
            col_skills, col_summary = st.columns(2)
            with col_skills:
                st.markdown(render_card_start("Kỹ năng"), unsafe_allow_html=True)
                skills_html = "<div class='badge-container'>"
                for skill in cv_data["KyNang"]:
                    skills_html += render_badge(skill, "default")
                skills_html += "</div>"
                st.markdown(skills_html, unsafe_allow_html=True)
                st.markdown(render_card_end(), unsafe_allow_html=True)
            with col_summary:
                st.markdown(render_card_start("Đánh giá tổng quan"), unsafe_allow_html=True)
                st.markdown(f"<p>{cv_data['TomTat']}</p>", unsafe_allow_html=True)
                st.markdown(render_card_end(), unsafe_allow_html=True)

            # 7.6. Certifications Card
            st.markdown(render_card_start("Chứng chỉ"), unsafe_allow_html=True)
            if cv_data["ChungChi"]:
                certs_html = "<div class='badge-container'>"
                for cert in cv_data["ChungChi"]:
                    certs_html += render_badge(cert, "default")
                certs_html += "</div>"
                st.markdown(certs_html, unsafe_allow_html=True)
            else:
                st.markdown("<p>Không có thông tin</p>", unsafe_allow_html=True)
            st.markdown(render_card_end(), unsafe_allow_html=True)

            # 7.7. Work Experience Card (using Streamlit expanders)
            st.markdown(render_card_start("Kinh nghiệm làm việc"), unsafe_allow_html=True)
            if cv_data["KinhNghiemLamViec"]:
                for exp in cv_data["KinhNghiemLamViec"]:
                    with st.expander(f"{exp['ViTri']} — {exp['CongTy']}"):
                        st.markdown(f"**Thời gian:** {exp['ThoiGian']}", unsafe_allow_html=True)
                        st.markdown(f"<p>{exp['MoTa']}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p>Không có thông tin kinh nghiệm làm việc.</p>", unsafe_allow_html=True)
            st.markdown(render_card_end(), unsafe_allow_html=True)

            # 7.8. Projects Card (using Streamlit expanders with blue tech badges)
            st.markdown(render_card_start("Dự án"), unsafe_allow_html=True)
            if cv_data["DuAn"]:
                for project in cv_data["DuAn"]:
                    with st.expander(f"{project['TenDuAn']} — {project['VaiTro']}"):
                        st.markdown(
                            f'<div class="project-detail-label">Mục đích:</div><p class="project-detail-text">{project["MucDich"]}</p>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="project-detail-label">Mô tả:</div><p class="project-detail-text">{project["MoTa"]}</p>',
                            unsafe_allow_html=True)
                        if project["CongNghe"]:
                            tech_badges = "<div class='project-detail-label'>Công nghệ:</div><div class='badge-container'>"
                            for tech in project["CongNghe"]:
                                tech_badges += render_badge(tech, "blue")
                            tech_badges += "</div>"
                            st.markdown(tech_badges, unsafe_allow_html=True)
                        if project["Link"]:
                            st.markdown(
                                f'<div class="project-detail-label">Link:</div><a href="https://{project["Link"]}" target="_blank" class="project-detail-link">{project["Link"]}</a>',
                                unsafe_allow_html=True)
            else:
                st.markdown("<p>Không có thông tin dự án.</p>", unsafe_allow_html=True)
            st.markdown(render_card_end(), unsafe_allow_html=True)

            # 7.10. Export Report Card
            st.markdown(render_card_start("Xuất báo cáo"), unsafe_allow_html=True)
            # Create a dummy Excel file content for download
            dummy_excel_data = {
                "Họ tên": [cv_data["HoTen"]],
                "Email": [cv_data["Email"]],
                "Điểm": [cv_data["Diem"]],
                "Học vấn": [cv_data["HocVan"]],
                "Kinh nghiệm": [cv_data["KinhNghiem"]],
                "Kỹ năng": [", ".join(cv_data["KyNang"])]
            }
            df = pd.DataFrame(dummy_excel_data)
            csv = df.to_csv(index=False,
                            encoding='utf-8-sig')  # Use utf-8-sig for proper Vietnamese characters in Excel

            st.download_button(
                label="📥 Tải xuống Excel",
                data=csv,
                file_name=f"bao_cao_{cv_data['HoTen'].replace(' ', '_')}.csv",
                mime="text/csv",
                key="download_single_cv_excel"
            )
            st.markdown(render_card_end(), unsafe_allow_html=True)

# --- TAB 2: BATCH PROCESSING ---
with tab2:
    st.markdown(render_card_start("Batch Processing — Phân tích & Xếp hạng hàng loạt"), unsafe_allow_html=True)
    uploaded_batch_files = st.file_uploader("Chọn nhiều file CV (PDF / DOCX)", type=["pdf", "docx"],
                                            accept_multiple_files=True, key="batch_cv_uploader")
    st.markdown(render_card_end(), unsafe_allow_html=True)

    # Display empty state if no files are uploaded for batch processing
    if not uploaded_batch_files:
        st.markdown(
            f"""
            <div class="empty-state-card">
                <div class="empty-state-icon">📚</div>
                <h3 class="empty-state-title">Tải lên nhiều CV để phân tích hàng loạt</h3>
                <p class="empty-state-subtitle">Hỗ trợ định dạng PDF và DOCX</p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        # Check if the list of uploaded files has changed to clear previous batch results
        current_uploaded_file_names = {f.name for f in uploaded_batch_files}
        if current_uploaded_file_names != st.session_state.last_batch_files:
            st.session_state.batch_results = []
            st.session_state.last_batch_files = current_uploaded_file_names

        # Button to trigger batch analysis
        if st.button("Phân tích tất cả", key="analyze_batch_button"):
            st.session_state.batch_results = []  # Clear previous results on new analysis click
            progress_text = "Đang xử lý CV..."
            my_bar = st.progress(0, text=progress_text)

            # Process each file with real AI backend
            for i, file in enumerate(uploaded_batch_files):
                raw_text = extract_text(file)
                if raw_text.strip():
                    result = process_cv_with_ai(st.session_state.api_key, raw_text)
                    if "error" not in result:
                        result["filename"] = file.name
                        # JD comparison for batch if JD provided
                        jd_text = st.session_state.job_description
                        if jd_text and jd_text.strip():
                            jd_result = compare_cv_with_jd(st.session_state.api_key, raw_text, jd_text)
                            if "error" not in jd_result:
                                result["JD_Comparison"] = jd_result
                                result["DiemTongHop"] = jd_result["PhuHop"]
                            else:
                                result["DiemTongHop"] = result.get("Diem", 50)
                        else:
                            result["DiemTongHop"] = result.get("Diem", 50)
                        st.session_state.batch_results.append(result)
                    else:
                        st.session_state.batch_results.append({
                            "filename": file.name,
                            "HoTen": file.name.split('.')[0].replace('_', ' '),
                            "Email": "N/A", "Diem": 0, "DiemTongHop": 0, "HocVan": "N/A",
                            "KinhNghiem": "N/A", "KyNang": [],
                            "TomTat": f"Lỗi phân tích: {result['error']}"
                        })
                else:
                    st.session_state.batch_results.append({
                        "filename": file.name,
                        "HoTen": file.name.split('.')[0].replace('_', ' '),
                        "Email": "N/A", "Diem": 0, "DiemTongHop": 0, "HocVan": "N/A",
                        "KinhNghiem": "N/A", "KyNang": [],
                        "TomTat": "Không thể trích xuất nội dung từ file."
                    })
                my_bar.progress((i + 1) / len(uploaded_batch_files),
                                text=f"Đang xử lý: {file.name} ({i + 1}/{len(uploaded_batch_files)})")

            my_bar.empty()
            st.success("Hoàn tất phân tích hàng loạt!")

        # Display results if batch processing has been done
        if st.session_state.batch_results:
            # Create a DataFrame for ranking table
            df_results = pd.DataFrame([
                {
                    "Hạng": i + 1,
                    "Tên": r["HoTen"],
                    "Email": r.get("Email", "N/A"),
                    "Điểm tổng hợp": r.get("DiemTongHop", r.get("Diem", 0)),
                    "Điểm CV": r.get("Diem", 0),
                    "Học vấn": r.get("HocVan", "N/A"),
                    "Kinh nghiệm": r.get("KinhNghiem", "N/A")
                } for i, r in enumerate(sorted(st.session_state.batch_results, key=lambda x: x.get("DiemTongHop", x.get("Diem", 0)), reverse=True))
            ])

            # 8.4. Candidate Ranking Table
            st.markdown(render_card_start("Bảng xếp hạng ứng viên"), unsafe_allow_html=True)
            st.dataframe(df_results, hide_index=True, use_container_width=True)
            st.markdown(render_card_end(), unsafe_allow_html=True)

            # 8.5. Detailed information for each candidate in batch (using expanders)
            st.markdown(render_card_start("Chi tiết từng ứng viên"), unsafe_allow_html=True)
            for r_index, r_row in df_results.iterrows():
                # Find the original data using the name (unique enough for dummy data)
                original_data = next((item for item in st.session_state.batch_results if item["HoTen"] == r_row["Tên"]),
                                     None)
                if original_data:
                    with st.expander(f"{r_row.Tên} — {r_row['Điểm tổng hợp']} điểm"):
                        col_details_left, col_details_right = st.columns(2)
                        with col_details_left:
                            st.markdown(
                                f'<div class="info-label">EMAIL</div><div class="info-value">{original_data["Email"]}</div>',
                                unsafe_allow_html=True)
                            st.markdown(
                                f'<div class="info-label" style="margin-top:1rem;">HỌC VẤN</div><div class="info-value">{original_data["HocVan"]}</div>',
                                unsafe_allow_html=True)
                        with col_details_right:
                            st.markdown(
                                f'<div class="info-label">ĐIỆN THOẠI</div><div class="info-value">Đang cập nhật</div>',
                                unsafe_allow_html=True)
                            st.markdown(
                                f'<div class="info-label" style="margin-top:1rem;">KINH NGHIỆM</div><div class="info-value">{original_data["KinhNghiem"]}</div>',
                                unsafe_allow_html=True)

                        st.markdown(f'<div class="info-label" style="margin-top:1rem;">KỸ NĂNG</div>',
                                    unsafe_allow_html=True)
                        skills_html_batch = "<div class='badge-container'>"
                        for skill in original_data["KyNang"]:
                            skills_html_batch += render_badge(skill, "default")
                        skills_html_batch += "</div>"
                        st.markdown(skills_html_batch, unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="info-label" style="margin-top:1rem;">TÓM TẮT</div><p>{original_data["TomTat"]}</p>',
                            unsafe_allow_html=True)
            st.markdown(render_card_end(), unsafe_allow_html=True)

            # 8.6. Export Batch Report
            st.markdown(render_card_start("Xuất báo cáo tổng hợp"), unsafe_allow_html=True)
            csv_batch = df_results.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Tải xuống Excel",
                data=csv_batch,
                file_name="bao_cao_tong_hop_ung_vien.csv",
                mime="text/csv",
                key="download_batch_excel"
            )
            st.markdown(render_card_end(), unsafe_allow_html=True)

# --- TAB 3: CHATBOT ---
with tab3:
    st.markdown(render_card_start("AI Assistant — Hỏi đáp về CV"), unsafe_allow_html=True)
    st.markdown(render_card_end(), unsafe_allow_html=True)

    # Display empty state if no CV has been parsed in Tab 1
    if st.session_state.parsed_cv_data is None:
        st.markdown(
            f"""
            <div class="empty-state-card">
                <div class="empty-state-icon">💬</div>
                <h3 class="empty-state-title">Phân tích CV trước để sử dụng Chatbot</h3>
                <p class="empty-state-subtitle">Vào tab "Phân tích CV" → Upload file → Quay lại đây</p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        # Status bar showing which CV is being analyzed
        st.markdown(
            f'<div class="sidebar-status-badge badge-blue" style="background-color: var(--accent-blue-light); color: var(--accent-blue);"><div class="sidebar-status-dot" style="background-color: var(--accent-blue);"></div>Đang phân tích CV của: **{st.session_state.parsed_cv_data["HoTen"]}**</div>',
            unsafe_allow_html=True)

        st.markdown("<h4>Gợi ý câu hỏi:</h4>", unsafe_allow_html=True)
        questions = [
            "Ứng viên phù hợp vị trí nào?",
            "Điểm mạnh của ứng viên?",
            "Cần cải thiện điều gì?"
        ]

        # Display suggested questions as clickable buttons
        cols_q = st.columns(len(questions))
        for i, q in enumerate(questions):
            with cols_q[i]:
                # Wrap st.button in a div with the custom class for styling
                st.markdown('<div class="chat-suggestion-button-container">', unsafe_allow_html=True)
                if st.button(q, key=f"chat_suggest_btn_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    with st.spinner("Đang tạo phản hồi..."):
                        ai_response = chat_about_cv(
                            st.session_state.api_key,
                            st.session_state.raw_cv_text,
                            st.session_state.parsed_cv_data,
                            st.session_state.chat_history[:-1],
                            q
                        )
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        # 9.5. Chat UI: Display chat messages and input field
        for message in st.session_state.chat_history:
            avatar = "🤖" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Chat input field
        if prompt := st.chat_input("Nhập câu hỏi về ứng viên..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Đang suy nghĩ..."):
                    ai_response = chat_about_cv(
                        st.session_state.api_key,
                        st.session_state.raw_cv_text,
                        st.session_state.parsed_cv_data,
                        st.session_state.chat_history[:-1],
                        prompt
                    )
                    st.markdown(ai_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})