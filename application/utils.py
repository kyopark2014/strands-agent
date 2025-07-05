import logging
import sys
import json
import traceback
import boto3
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-basic")

def load_config():
    config = None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
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
weather_api_key = ""

# AWS 자격 증명 확인 및 설정
def get_aws_client(service_name, region_name=None):
    """AWS 클라이언트를 생성하고 자격 증명을 확인합니다."""
    try:
        # 환경 변수에서 자격 증명 확인
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_region = region_name or os.environ.get('AWS_DEFAULT_REGION', bedrock_region)
        
        if aws_access_key and aws_secret_key:
            # 환경 변수로 자격 증명 설정
            client = boto3.client(
                service_name=service_name,
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                aws_session_token=os.environ.get('AWS_SESSION_TOKEN')
            )
        else:
            # 기본 자격 증명 파일 사용
            client = boto3.client(
                service_name=service_name,
                region_name=aws_region
            )
        
        return client
    except Exception as e:
        logger.error(f"AWS 클라이언트 생성 실패: {e}")
        return None

secretsmanager = get_aws_client('secretsmanager', bedrock_region)

# api key for weather
try:
    if secretsmanager:
        get_weather_api_secret = secretsmanager.get_secret_value(
            SecretId=f"openweathermap-{projectName}"
        )
        secret = json.loads(get_weather_api_secret['SecretString'])
        weather_api_key = secret['weather_api_key']
    else:
        logger.warning("AWS Secrets Manager 클라이언트를 사용할 수 없습니다. 환경 변수에서 직접 설정하세요.")
        weather_api_key = os.environ.get('WEATHER_API_KEY', '')

except Exception as e:
    logger.warning(f"Weather API 키를 가져올 수 없습니다: {e}")
    weather_api_key = os.environ.get('WEATHER_API_KEY', '')

# api key to use Tavily Search
tavily_key = ""
tavily_api_wrapper = None
try:
    if secretsmanager:
        get_tavily_api_secret = secretsmanager.get_secret_value(
            SecretId=f"tavilyapikey-{projectName}"
        )
        secret = json.loads(get_tavily_api_secret['SecretString'])

        if "tavily_api_key" in secret:
            tavily_key = secret['tavily_api_key']

            if tavily_key:
                tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
            else:
                logger.info(f"tavily_key is required.")
    else:
        logger.warning("AWS Secrets Manager 클라이언트를 사용할 수 없습니다. 환경 변수에서 직접 설정하세요.")
        tavily_key = os.environ.get('TAVILY_API_KEY', '')
        if tavily_key:
            tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
            
except Exception as e: 
    logger.warning(f"Tavily credential is required: {e}")
    tavily_key = os.environ.get('TAVILY_API_KEY', '')
    if tavily_key:
        tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)

# api key to use firecrawl Search
firecrawl_key = ""
try:
    if secretsmanager:
        get_firecrawl_secret = secretsmanager.get_secret_value(
            SecretId=f"firecrawlapikey-{projectName}"
        )
        secret = json.loads(get_firecrawl_secret['SecretString'])

        if "firecrawl_api_key" in secret:
            firecrawl_key = secret['firecrawl_api_key']
    else:
        firecrawl_key = os.environ.get('FIRECRAWL_API_KEY', '')
        
except Exception as e: 
    logger.warning(f"Firecrawl credential is required: {e}")
    firecrawl_key = os.environ.get('FIRECRAWL_API_KEY', '')

# api key to use perplexity Search
perplexity_key = ""
try:
    if secretsmanager:
        get_perplexity_api_secret = secretsmanager.get_secret_value(
            SecretId=f"perplexityapikey-{projectName}"
        )
        secret = json.loads(get_perplexity_api_secret['SecretString'])

        if "perplexity_api_key" in secret:
            perplexity_key = secret['perplexity_api_key']
    else:
        perplexity_key = os.environ.get('PERPLEXITY_API_KEY', '')

except Exception as e: 
    logger.warning(f"perplexity credential is required: {e}")
    perplexity_key = os.environ.get('PERPLEXITY_API_KEY', '')

async def generate_pdf_report(report_content: str, filename: str) -> str:
    """
    Generates a PDF report from the research findings.
    Args:
        report_content: The content to be converted into PDF format
        filename: Base name for the generated PDF file
    Returns:
        A message indicating the result of PDF generation
    """
    logger.info(f'###### generate_pdf_report ######')
    try:
        # Ensure directory exists
        os.makedirs("artifacts", exist_ok=True)
        # Set up the PDF file
        filepath = f"artifacts/{filename}.pdf"
        logger.info(f"filepath: {filepath}")
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        # Register TTF font directly (specify path to NanumGothic font file)
        font_path = "assets/NanumGothic-Regular.ttf"  # Change to actual TTF file path
        pdfmetrics.registerFont(TTFont('NanumGothic', font_path))
        # Create styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Normal_KO',
                                fontName='NanumGothic',
                                fontSize=10,
                                spaceAfter=12))  # 문단 간격 증가
        styles.add(ParagraphStyle(name='Heading1_KO',
                                fontName='NanumGothic',
                                fontSize=16,
                                spaceAfter=20,  # 제목 후 여백 증가
                                textColor=colors.HexColor('#0000FF')))  # 파란색
        styles.add(ParagraphStyle(name='Heading2_KO',
                                fontName='NanumGothic',
                                fontSize=14,
                                spaceAfter=16,  # 제목 후 여백 증가
                                textColor=colors.HexColor('#0000FF')))  # 파란색
        styles.add(ParagraphStyle(name='Heading3_KO',
                                fontName='NanumGothic',
                                fontSize=12,
                                spaceAfter=14,  # 제목 후 여백 증가
                                textColor=colors.HexColor('#0000FF')))  # 파란색
        # Process content
        elements = []
        lines = report_content.split('\n')
        for line in lines:
            if line.startswith('# '):
                elements.append(Paragraph(line[2:], styles['Heading1_KO']))
            elif line.startswith('## '):
                elements.append(Paragraph(line[3:], styles['Heading2_KO']))
            elif line.startswith('### '):
                elements.append(Paragraph(line[4:], styles['Heading3_KO']))
            elif line.strip():  # Skip empty lines
                elements.append(Paragraph(line, styles['Normal_KO']))
        # Build PDF
        doc.build(elements)
        return f"PDF report generated successfully: {filepath}"
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        # Fallback to text file
        try:
            text_filepath = f"artifacts/{filename}.txt"
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            return f"PDF generation failed. Saved as text file instead: {text_filepath}"
        except Exception as text_error:
            return f"Error generating report: {str(e)}. Text fallback also failed: {str(text_error)}"