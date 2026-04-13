import logging
import sys
import json
import traceback
import boto3
import os
import asyncio
import re
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-basic")

aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
aws_session_token = os.environ.get('AWS_SESSION_TOKEN')
aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')

workingDir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(workingDir, "config.json")

def load_config():
    config = None
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
        projectName = "strands"
        session = boto3.Session()
        region = session.region_name
        config['region'] = region
        config['projectName'] = projectName
        
        sts = boto3.client("sts")
        response = sts.get_caller_identity()
        accountId = response["Account"]
        config['accountId'] = accountId
        config['s3_bucket'] = f'storage-for-{projectName}-{accountId}-{region}'
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)    
    return config

config = load_config()

bedrock_region = config['region']
projectName = config['projectName']
        
def get_contents_type(file_name):
    if file_name.lower().endswith((".jpg", ".jpeg")):
        content_type = "image/jpeg"
    elif file_name.lower().endswith((".pdf")):
        content_type = "application/pdf"
    elif file_name.lower().endswith((".txt")):
        content_type = "text/plain"
    elif file_name.lower().endswith((".csv")):
        content_type = "text/csv"
    elif file_name.lower().endswith((".ppt", ".pptx")):
        content_type = "application/vnd.ms-powerpoint"
    elif file_name.lower().endswith((".doc", ".docx")):
        content_type = "application/msword"
    elif file_name.lower().endswith((".xls")):
        content_type = "application/vnd.ms-excel"
    elif file_name.lower().endswith((".py")):
        content_type = "text/x-python"
    elif file_name.lower().endswith((".js")):
        content_type = "application/javascript"
    elif file_name.lower().endswith((".md")):
        content_type = "text/markdown"
    elif file_name.lower().endswith((".png")):
        content_type = "image/png"
    else:
        content_type = "no info"    
    return content_type

def load_mcp_env():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_env_path = os.path.join(script_dir, "mcp.env")
    
    with open(mcp_env_path, "r", encoding="utf-8") as f:
        mcp_env = json.load(f)
    return mcp_env

def save_mcp_env(mcp_env):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_env_path = os.path.join(script_dir, "mcp.env")
    
    with open(mcp_env_path, "w", encoding="utf-8") as f:
        json.dump(mcp_env, f)

# api key to get weather information in agent
if aws_access_key and aws_secret_key:
    secretsmanager = boto3.client(
        service_name='secretsmanager',
        region_name=bedrock_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_session_token=aws_session_token,
    )
else:
    secretsmanager = boto3.client(
        service_name='secretsmanager',
        region_name=bedrock_region
    )

# api key for slack
slack_bot_token = ""
slack_team_id = ""
try:
    get_slack_secret = secretsmanager.get_secret_value(
        SecretId=f"slackapikey-{projectName}"
    )
    secret = json.loads(get_slack_secret['SecretString'])
    slack_bot_token = secret.get('slack_bot_token', '')
    slack_team_id = secret.get('slack_team_id', '')
    if slack_bot_token:
        os.environ["SLACK_BOT_TOKEN"] = slack_bot_token
    if slack_team_id:
        os.environ["SLACK_TEAM_ID"] = slack_team_id
except Exception as e:
    logger.info(f"Slack credential is required: {e}")
    pass

# api key to use Tavily Search (Secrets Manager or env TAVILY_API_KEY)
tavily_key = tavily_api_wrapper = ""
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    secret = json.loads(get_tavily_api_secret["SecretString"])

    if "tavily_api_key" in secret:
        tavily_key = secret["tavily_api_key"]

        if tavily_key:
            tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
            os.environ["TAVILY_API_KEY"] = tavily_key
        else:
            logger.info("tavily_api_key in secret is empty.")
except Exception as e:
    logger.warning(
        "Tavily AWS secret unavailable (%s). Set TAVILY_API_KEY env or create secret tavilyapikey-%s.",
        e,
        projectName,
    )

if not tavily_key and os.environ.get("TAVILY_API_KEY"):
    tavily_key = os.environ["TAVILY_API_KEY"].strip()
    if tavily_key:
        tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)

# api key to use notion
notion_key = ""
async def generate_pdf_report(body: str, request_id: str) -> str:
    """Generate a PDF report from markdown content using reportlab.

    Args:
        body: Markdown content string.
        request_id: Unique identifier used for the output filename.

    Returns:
        Output PDF file path on success, empty string on failure.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, HRFlowable,
        )
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        pdf_filename = f"artifacts/{request_id}.pdf"
        os.makedirs("artifacts", exist_ok=True)

        font_name = "Helvetica"
        cjk_font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for fp in cjk_font_paths:
            if os.path.isfile(fp):
                try:
                    pdfmetrics.registerFont(TTFont("CJKFont", fp))
                    font_name = "CJKFont"
                    break
                except Exception:
                    continue

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            "KTitle", parent=styles["Title"], fontName=font_name, fontSize=18,
            leading=24, spaceAfter=12,
        ))
        styles.add(ParagraphStyle(
            "KH2", parent=styles["Heading2"], fontName=font_name, fontSize=14,
            leading=20, spaceBefore=14, spaceAfter=8,
        ))
        styles.add(ParagraphStyle(
            "KH3", parent=styles["Heading3"], fontName=font_name, fontSize=12,
            leading=16, spaceBefore=10, spaceAfter=6,
        ))
        styles.add(ParagraphStyle(
            "KBody", parent=styles["BodyText"], fontName=font_name, fontSize=10,
            leading=15, spaceAfter=6,
        ))
        styles.add(ParagraphStyle(
            "KBullet", parent=styles["BodyText"], fontName=font_name, fontSize=10,
            leading=15, leftIndent=16, spaceAfter=3, bulletIndent=6,
        ))

        def _md_inline(text: str) -> str:
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
            text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" color="blue">\1</a>', text)
            return text

        story = []
        if not body:
            body = "## 결과\n\n내용이 없습니다."

        lines = body.split("\n")

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("# ") and not stripped.startswith("## "):
                story.append(Paragraph(_md_inline(stripped[2:]), styles["KTitle"]))
            elif stripped.startswith("### "):
                story.append(Paragraph(_md_inline(stripped[4:]), styles["KH3"]))
            elif stripped.startswith("## "):
                story.append(Spacer(1, 4 * mm))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
                story.append(Paragraph(_md_inline(stripped[3:]), styles["KH2"]))
            elif stripped.startswith("- ") or stripped.startswith("* "):
                content = stripped[2:]
                story.append(Paragraph(f"• {_md_inline(content)}", styles["KBullet"]))
            elif re.match(r'^\d+\.\s', stripped):
                story.append(Paragraph(_md_inline(stripped), styles["KBullet"]))
            elif stripped.startswith("|") and "---" not in stripped:
                pass  # skip raw table rows for simplicity
            elif stripped == "":
                story.append(Spacer(1, 3 * mm))
            else:
                story.append(Paragraph(_md_inline(stripped), styles["KBody"]))

        def _run():
            doc = SimpleDocTemplate(
                pdf_filename, pagesize=A4,
                topMargin=20 * mm, bottomMargin=20 * mm,
                leftMargin=18 * mm, rightMargin=18 * mm,
            )
            doc.build(story)

        await asyncio.to_thread(_run)
        logger.info(f"PDF report generated: {pdf_filename}")
        return pdf_filename

    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        return ""


def get_notion_key():
    global notion_key

    if not notion_key:
        try:
            get_notion_api_secret = secretsmanager.get_secret_value(
                SecretId=f"notionapikey-{projectName}"
            )
            #logger.info('get_perplexity_api_secret: ', get_perplexity_api_secret)
            secret = json.loads(get_notion_api_secret['SecretString'])
            #logger.info('secret: ', secret)

            if "notion_api_key" in secret:
                notion_key = secret['notion_api_key']
                # logger.info('updated notion_key: ', notion_key)

        except Exception as e: 
            logger.info(f"nova act credential is required: {e}")
            # raise e
            pass
    return notion_key
