IKEA_STORE_DB="data\ikea_store_database.db"
MARKDOWN_IMAGE_REGEX = r"!\[.*?\]\(.*?\)"

# Refined color theme for light mode - IKEA inspired
PRIMARY_COLOR = "#0058A3"      # IKEA blue
SECONDARY_COLOR = "#F5F5F5"    # Light gray background
TEXT_COLOR = "#2C2C2C"         # Dark text for readability
ACCENT_COLOR = "#FFDA1A"       # IKEA yellow
HOVER_COLOR = "#005C98"        # Darker blue for hover
BORDER_COLOR = "#D1D5DB"       # Light border color
INPUT_BG_COLOR = "#FFFFFF"     # White input background
SIDEBAR_BG_COLOR = "#FAFAFA"   # Very light gray sidebar
GRADIENT_PRIMARY = "linear-gradient(135deg, #0058A3 0%, #005C98 100%)"  # IKEA gradient

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, .stApp {{
        background-color: #FFFFFF;
        color: #2C2C2C;
        font-family: 'Inter', sans-serif;
    }}

    /* Layout spacing */
    .main .block-container {{
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }}

    /* Headers */
    h1, h2, h3, h4, h5 {{
        color: #0058A3;
        font-weight: 700;
    }}

    /* Buttons - Enhanced styling with consistent padding */
    .stButton > button {{
        background-color: #0058A3;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 8px rgba(0, 88, 163, 0.2);
        transition: all 0.2s ease;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        font-size: 14px !important;
        line-height: 1.4 !important;
        margin: 4px 0 !important;
    }}

    .stButton > button:hover {{
        background-color: #005C98;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0, 88, 163, 0.3);
    }}

    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(0, 88, 163, 0.2);
    }}

    /* Specific button container fixes */
    .stButton {{
        margin-bottom: 8px !important;
    }}

    /* Column button alignment fix */
    div[data-testid="column"] .stButton {{
        width: 100% !important;
    }}

    div[data-testid="column"] .stButton > button {{
        width: 100% !important;
    }}

    /* Sidebar - Enhanced styling with consistent padding */
    [data-testid="stSidebar"] {{
        background-color: #FAFAFA;
        border-right: 1px solid #D1D5DB;
        padding: 1rem !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: #2C2C2C;
        padding: 0 !important;
        margin-bottom: 1rem !important;
    }}

    /* Sidebar button container fixes */
    [data-testid="stSidebar"] .stButton {{
        margin-bottom: 0.5rem !important;
    }}

    [data-testid="stSidebar"] .stButton > button {{
        padding: 0.5rem 1rem !important;
        font-size: 13px !important;
        min-height: 38px !important;
        width: 100% !important;
    }}

    /* Sidebar columns fix */
    [data-testid="stSidebar"] div[data-testid="column"] {{
        padding: 0 4px !important;
    }}

    /* Remove top padding from sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 0 !important;
    }}

    /* Input fields - Enhanced styling */
    .stTextInput > div > div > input {{
        border-radius: 12px !important;
        padding: 12px 16px !important;
        background-color: #FFFFFF !important;
        color: #2C2C2C !important;
        border: 1px solid #D1D5DB !important;
        font-size: 14px !important;
        min-height: 44px !important;
        box-sizing: border-box !important;
        transition: all 0.2s ease !important;
    }}

    .stTextInput > div > div > input:focus {{
        border-color: #0058A3 !important;
        box-shadow: 0 0 0 3px rgba(0, 88, 163, 0.1) !important;
        outline: none !important;
    }}

    /* Text input label styling */
    .stTextInput > label {{
        font-weight: 600 !important;
        color: #2C2C2C !important;
        margin-bottom: 0.5rem !important;
        font-size: 14px !important;
    }}

    /* Container spacing fixes */
    .stTextInput {{
        margin-bottom: 1rem !important;
    }}

    /* Chat input container - Clean background */
    /* Multiple selectors to catch different Streamlit versions */
    [data-testid="stChatInput"],
    .stChatInput,
    div[data-testid="stChatInput"] {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 8px !important;
        margin: 0 auto !important;
    }}

    /* Target textarea with multiple approaches */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input,
    .stChatInput textarea,
    .stChatInput input,
    textarea[placeholder*="message"],
    input[placeholder*="message"] {{
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 16px !important;
        color: #2C2C2C !important;
        padding: 14px 16px !important;
        font-size: 14px !important;
        line-height: 1.4 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease !important;
        resize: none !important;
        min-height: 44px !important;
        width: 100% !important;
    }}

    /* Hover and focus states - Multiple selectors */
    [data-testid="stChatInput"] textarea:hover,
    [data-testid="stChatInput"] input:hover,
    .stChatInput textarea:hover,
    .stChatInput input:hover,
    textarea[placeholder*="message"]:hover,
    input[placeholder*="message"]:hover {{
        border-color: #0058A3 !important;
        box-shadow: 0 4px 12px rgba(0, 88, 163, 0.1) !important;
    }}

    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] input:focus,
    .stChatInput textarea:focus,
    .stChatInput input:focus,
    textarea[placeholder*="message"]:focus,
    input[placeholder*="message"]:focus {{
        border-color: #0058A3 !important;
        box-shadow: 0 0 0 3px rgba(0, 88, 163, 0.1), 0 4px 12px rgba(0, 88, 163, 0.15) !important;
        outline: none !important;
    }}

    /* Placeholder text styling - Multiple selectors */
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder,
    .stChatInput textarea::placeholder,
    .stChatInput input::placeholder,
    textarea[placeholder*="message"]::placeholder,
    input[placeholder*="message"]::placeholder {{
        color: #6B7280 !important;
        opacity: 0.8 !important;
    }}

    /* Remove surrounding container styling - More comprehensive */
    div:has([data-testid="stChatInput"]),
    div:has(.stChatInput),
    .stChatInputContainer,
    section:has([data-testid="stChatInput"]) {{
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 8px 0 !important;
    }}

    /* Nuclear option - Force styling on all textareas in chat area */
    .main textarea,
    .stApp textarea {{
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 16px !important;
        color: #2C2C2C !important;
        padding: 14px 16px !important;
    }}

    /* Optional: Style the send button if it exists */
    [data-testid="stChatInput"] button,
    [data-testid="stChatInput"] .stButton button,
    .stChatInput button,
    div[data-testid="stChatInput"] button {{
        background-color: #0058A3 !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        padding: 8px 10px !important;
        transition: background-color 0.2s ease !important;
        margin-bottom: 2px !important;
        height: 36px !important;
        align-self: flex-end !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}

    [data-testid="stChatInput"] button:hover,
    [data-testid="stChatInput"] .stButton button:hover,
    .stChatInput button:hover {{
        background-color: #005C98 !important;
    }}

    /* Chat input container flexbox alignment */
    [data-testid="stChatInput"],
    .stChatInput {{
        display: flex !important;
        align-items: flex-end !important;
        gap: 8px !important;
    }}

    /* Ensure the input and button are properly aligned */
    [data-testid="stChatInput"] > div,
    .stChatInput > div {{
        display: flex !important;
        align-items: flex-end !important;
        gap: 8px !important;
        width: 100% !important;
    }}

    /* Chat container */
    [data-testid="stChatMessage"] {{
        border-radius: 14px !important;
        padding: 12px 16px !important;
        margin-bottom: 12px !important;
        background-color: #F8F9FA !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #2C2C2C !important;
        border: 1px solid #E5E7EB;
    }}

    [data-testid="stChatMessage"]:has([data-testid="user"]) {{
        background-color: #E3F2FD !important;
        border: 1px solid #BBDEFB;
    }}

    /* Chat header */
    .chat-header {{
        margin-bottom: 1rem;
        border-bottom: 1px solid #D1D5DB;
        padding-bottom: 0.5rem;
        color: #0058A3;
        font-size: 1.6rem;
    }}

    /* Source container */
    .source-container {{
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0058A3;
        margin-top: 1rem;
        border: 1px solid #E5E7EB;
    }}

    .citation-badge {{
        background: #E3F2FD;
        color: #0058A3;
        font-weight: bold;
        width: 22px;
        height: 22px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        margin-right: 6px;
        border: 1px solid #BBDEFB;
    }}

    .citation-link {{
        color: #0058A3;
        text-decoration: none;
    }}

    .citation-link:hover {{
        text-decoration: underline;
        color: #005C98;
    }}

    /* Login card and image */
    .login-container {{
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        margin: 0 auto;
        max-width: 400px;
        color: #2C2C2C;
        border: 1px solid #E5E7EB;
    }}

    div[data-testid="stImage"] {{
        text-align: center;
        margin-bottom: 1rem;
    }}

    /* Remove top padding in sidebar image */
    .css-1kyxreq.e115fcil1 {{
        padding-top: 0 !important;
    }}

    /* Additional light theme adjustments */
    .stSelectbox > div > div {{
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
    }}

    .stMultiSelect > div > div {{
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
    }}

    /* Metric containers */
    [data-testid="metric-container"] {{
        background-color: #F8F9FA;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
    }}

    /* Column spacing fixes */
    div[data-testid="column"] {{
        padding: 0 0.5rem !important;
    }}

    /* Main container spacing */
    .main .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1000px !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}

    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {{
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.5rem 0 !important;
    }}

    /* Chat message spacing */
    [data-testid="stChatMessage"] {{
        margin: 0.75rem 0 !important;
    }}

    /* Remove extra padding from markdown containers */
    [data-testid="stMarkdownContainer"] {{
        padding-top: 0 !important;
    }}

    /* Spinner styling */
    .stSpinner > div {{
        border-color: #0058A3 !important;
    }}
</style>
"""