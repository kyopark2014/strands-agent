import logging
import sys
import json
import traceback
import boto3
import os
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
