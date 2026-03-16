import chat
import os
import contextlib
import mcp_config
import logging
import sys
import utils
import boto3
import traceback
import yaml
import io

from contextlib import contextmanager
from typing import Dict, List, Optional
from strands.models import BedrockModel
from strands_tools import calculator, current_time
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from botocore.config import Config
from dataclasses import dataclass

from strands import Agent, tool

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("strands-agent")

initiated = False
strands_tools = []
mcp_servers = []

tool_list = []

memory_id = actor_id = session_id = namespace = None

s3_prefix = "docs"
capture_prefix = "captures"

selected_strands_tools = []
selected_mcp_servers = []

aws_region = utils.bedrock_region

config = utils.load_config()
s3_bucket = config.get("s3_bucket")
sharing_url = config.get("sharing_url")

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(WORKING_DIR, "skills")
ARTIFACTS_DIR = os.path.join(WORKING_DIR, "artifacts")

# ═══════════════════════════════════════════════════════════════════
#  1. Skill System  – Anthropic Agent Skills spec 구현
#     (https://agentskills.io/specification)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    path: str

class SkillManager:
    """Discovers, loads and selects Agent Skills following the Anthropic spec."""

    def __init__(self, skills_dir: str = SKILLS_DIR):
        self.skills_dir = skills_dir
        self.registry: dict[str, Skill] = {}
        self._discover()

    # ---- discovery & metadata loading ----

    def _discover(self):
        """Scan skills directory and load metadata (frontmatter only)."""
        if not os.path.isdir(self.skills_dir):
            os.makedirs(self.skills_dir, exist_ok=True)
            logger.info(f"Created skills directory: {self.skills_dir}")
            return

        for entry in os.listdir(self.skills_dir):
            skill_md = os.path.join(self.skills_dir, entry, "SKILL.md")
            if os.path.isfile(skill_md):
                try:
                    meta, instructions = self._parse_skill_md(skill_md)
                    skill = Skill(
                        name=meta.get("name", entry),
                        description=meta.get("description", ""),
                        instructions=instructions,
                        path=os.path.join(self.skills_dir, entry),
                    )
                    self.registry[skill.name] = skill
                    logger.info(f"Skill discovered: {skill.name}")
                except Exception as e:
                    logger.warning(f"Failed to load skill '{entry}': {e}")

    @staticmethod
    def _parse_skill_md(filepath: str) -> tuple[dict, str]:
        """Parse YAML frontmatter + markdown body from a SKILL.md file."""
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        if not raw.startswith("---"):
            return {}, raw

        parts = raw.split("---", 2)
        if len(parts) < 3:
            return {}, raw

        frontmatter = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()
        return frontmatter, body

    # ---- prompt generation (progressive disclosure) ----
    def available_skills_xml(self) -> str:
        """Generate <available_skills> XML for the system prompt (metadata only)."""
        if not self.registry:
            return ""
        lines = ["<available_skills>"]
        for s in self.registry.values():
            lines.append("  <skill>")
            lines.append(f"    <name>{s.name}</name>")
            lines.append(f"    <description>{s.description}</description>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def get_skill_instructions(self, name: str) -> Optional[str]:
        """Return full instructions for a skill (loaded on demand)."""
        skill = self.registry.get(name)
        return skill.instructions if skill else None

    def select_skills(self, query: str) -> list[Skill]:
        """Keyword-based matching to select relevant skills for a query."""
        query_lower = query.lower()
        selected = []
        for skill in self.registry.values():
            keywords = skill.description.lower().split()
            if any(kw in query_lower for kw in keywords if len(kw) > 3):
                selected.append(skill)
        return selected

    def build_active_skill_prompt(self, skills: list[Skill]) -> str:
        """Build the full instructions block for activated skills."""
        if not skills:
            return ""
        parts = ["<active_skills>"]
        for s in skills:
            parts.append(f'<skill name="{s.name}">')
            parts.append(s.instructions)
            parts.append("</skill>")
        parts.append("</active_skills>")
        return "\n".join(parts)

# global singleton
skill_manager = SkillManager()

SKILL_USAGE_GUIDE = (
    "\n## Skill 사용 가이드\n"
    "위의 <available_skills>에 나열된 skill이 사용자의 요청과 관련될 때:\n"
    "1. 먼저 get_skill_instructions 도구로 해당 skill의 상세 지침을 로드하세요.\n"
    "2. 지침에 포함된 코드 패턴을 execute_code 도구로 실행하세요.\n"
    "3. skill 지침이 없는 일반 질문은 직접 답변하세요.\n"
)

BASE_SYSTEM_PROMPT = (
    "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다.\n"
    "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다.\n"
    "모르는 질문을 받으면 솔직히 모른다고 말합니다.\n"
    "한국어로 답변하세요.\n\n"
    "## Agent Workflow\n"
    "1. 사용자 입력을 받는다\n"
    "2. 요청에 맞는 skill이 있으면 get_skill_instructions 도구로 상세 지침을 로드한다\n"
    "3. skill 지침에 따라 execute_code, write_file 등의 도구를 사용하여 작업을 수행한다\n"
    "4. 결과 파일이 있으면 upload_file_to_s3로 업로드하여 URL을 제공한다\n"
    "5. 최종 결과를 사용자에게 전달한다\n"
)

def build_system_prompt(custom_prompt: Optional[str] = None) -> str:
    """Assemble the full system prompt with available skills metadata."""
    if custom_prompt:
        base = custom_prompt
    else:
        base = BASE_SYSTEM_PROMPT

    skills_xml = skill_manager.available_skills_xml()
    logger.info(f"skills_xml: {skills_xml}")

    if skills_xml:
        return f"{base}\n\n{skills_xml}\n{SKILL_USAGE_GUIDE}"
    return base

# ═══════════════════════════════════════════════════════════════════
#  2. Built-in Tools – code execution, file I/O, S3 upload
# ═══════════════════════════════════════════════════════════════════
@tool
def execute_code(code: str) -> str:
    """Execute Python code and return stdout/stderr output.

    Use this tool to run Python code for tasks such as generating PDFs,
    processing data, or performing computations. The execution environment
    has access to common libraries: reportlab, pypdf, pdfplumber, pandas,
    json, csv, os, etc.

    Generated files should be saved to the 'artifacts/' directory.

    Args:
        code: Python code to execute.

    Returns:
        Captured stdout output, or error traceback if execution failed.
    """
    logger.info(f"###### execute_code ######")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    old_cwd = os.getcwd()
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        os.chdir(WORKING_DIR)

        skill_paths = []
        if os.path.isdir(SKILLS_DIR):
            for entry in os.listdir(SKILLS_DIR):
                sp = os.path.join(SKILLS_DIR, entry)
                if os.path.isdir(sp) and sp not in sys.path:
                    sys.path.insert(0, sp)
                    skill_paths.append(sp)

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture

        import subprocess, json, pathlib, shutil, tempfile, glob, datetime, math, re as _re
        exec_globals = {
            "__builtins__": __builtins__,
            "subprocess": subprocess,
            "json": json,
            "os": os,
            "sys": sys,
            "io": io,
            "pathlib": pathlib,
            "shutil": shutil,
            "tempfile": tempfile,
            "glob": glob,
            "datetime": datetime,
            "math": math,
            "re": _re,
        }
        exec(code, exec_globals)

        sys.stdout, sys.stderr = old_stdout, old_stderr
        os.chdir(old_cwd)
        for sp in skill_paths:
            if sp in sys.path:
                sys.path.remove(sp)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        result = ""
        if output:
            result += output
        if errors:
            result += f"\n[stderr]\n{errors}"
        if not result.strip():
            result = "Code executed successfully (no output)."

        return result

    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        os.chdir(old_cwd)
        for sp in skill_paths:
            if sp in sys.path:
                sys.path.remove(sp)
        tb = traceback.format_exc()
        logger.error(f"Code execution error: {tb}")
        return f"Error executing code:\n{tb}"

@tool
def write_file(filepath: str, content: str) -> str:
    """Write text content to a file.

    Args:
        filepath: Path relative to the working directory (e.g. 'artifacts/report.md').
        content: The text content to write.

    Returns:
        A success or failure message.
    """
    logger.info(f"###### write_file: {filepath} ######")
    try:
        full_path = os.path.join(WORKING_DIR, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        result_msg = f"파일이 저장되었습니다: {filepath}"

        s3_bucket = config.get("s3_bucket")
        if s3_bucket and sharing_url:
            try:
                import boto3
                from urllib import parse as url_parse
                s3 = boto3.client("s3", region_name=config.get("region", "us-west-2"))
                content_type = utils.get_contents_type(filepath)
                s3.put_object(Bucket=s3_bucket, Key=filepath, Body=content, ContentType=content_type)
                url = f"{sharing_url}/{url_parse.quote(filepath)}"
                result_msg += f"\nURL: {url}"
            except Exception as ue:
                logger.warning(f"S3 upload failed: {ue}")

        return result_msg
    except Exception as e:
        return f"파일 저장 실패: {str(e)}"


@tool
def read_file(filepath: str) -> str:
    """Read the contents of a local file.

    Args:
        filepath: Path relative to the working directory.

    Returns:
        The file contents as text, or an error message.
    """
    logger.info(f"###### read_file: {filepath} ######")
    try:
        full_path = os.path.join(WORKING_DIR, filepath)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"파일 읽기 실패: {str(e)}"


@tool
def upload_file_to_s3(filepath: str) -> str:
    """Upload a local file to S3 and return the download URL.

    Args:
        filepath: Path relative to the working directory (e.g. 'artifacts/report.pdf').

    Returns:
        The download URL, or an error message.
    """
    logger.info(f"###### upload_file_to_s3: {filepath} ######")
    try:
        import boto3
        from urllib import parse as url_parse

        s3_bucket = config.get("s3_bucket")
        if not s3_bucket:
            return "S3 버킷이 설정되어 있지 않습니다."

        full_path = os.path.join(WORKING_DIR, filepath)
        if not os.path.exists(full_path):
            return f"파일을 찾을 수 없습니다: {filepath}"

        content_type = utils.get_contents_type(filepath)
        s3 = boto3.client("s3", region_name=config.get("region", "us-west-2"))

        with open(full_path, "rb") as f:
            s3.put_object(Bucket=s3_bucket, Key=filepath, Body=f.read(), ContentType=content_type)

        if sharing_url:
            url = f"{sharing_url}/{url_parse.quote(filepath)}"
            return f"업로드 완료: {url}"
        return f"업로드 완료: s3://{s3_bucket}/{filepath}"

    except Exception as e:
        return f"업로드 실패: {str(e)}"


@tool
def get_skill_instructions(skill_name: str) -> str:
    """시스템 프롬프트의 <available_skills>에 나열된 skill의 상세 실행 지침을 로드합니다.

    사용자가 검색, 날씨, PDF, 도서 등 skill로 처리할 수 있는 작업을 요청하면
    반드시 이 도구를 먼저 호출하여 해당 skill의 실행 방법을 확인하세요.
    예: 인터넷 검색 → get_skill_instructions("tavily-search")
        날씨 조회 → get_skill_instructions("search-weather")

    Args:
        skill_name: skill 이름 (e.g. 'tavily-search', 'pdf', 'search-weather', 'book-search').

    Returns:
        해당 skill의 상세 실행 지침, 또는 skill을 찾을 수 없을 때 오류 메시지.
    """
    logger.info(f"###### get_skill_instructions: {skill_name} ######")
    instructions = skill_manager.get_skill_instructions(skill_name)
    if instructions:
        return instructions
    available = ", ".join(skill_manager.registry.keys())
    return f"Skill '{skill_name}'을 찾을 수 없습니다. 사용 가능한 skill: {available}"


def get_builtin_tools():
    """Return the list of built-in tools for the skill-aware agent."""
    return [execute_code, write_file, read_file, upload_file_to_s3, get_skill_instructions]


#########################################################
# Strands Agent 
#########################################################
def get_model():
    if chat.model_type == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif chat.model_type == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
    elif chat.model_type == 'openai':
        STOP_SEQUENCE = "" 

    if chat.model_type == 'claude':
        maxOutputTokens = chat.get_max_output_tokens()
    else:
        maxOutputTokens = 5120

    maxReasoningOutputTokens=64000
    thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens-1000)

    # AWS 자격 증명 설정
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_session_token = os.environ.get('AWS_SESSION_TOKEN')

    # Bedrock 클라이언트 설정
    bedrock_config = Config(
        read_timeout=900,
        connect_timeout=900,
        retries=dict(max_attempts=3, mode="adaptive"),
    )

    if aws_access_key and aws_secret_key:
        boto_session = boto3.Session(
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
        )
    else:
        boto_session = boto3.Session(region_name=aws_region)

    if chat.reasoning_mode=='Enable' and chat.model_type != 'openai':
        model = BedrockModel(
            boto_session=boto_session,
            boto_client_config=bedrock_config,
            model_id=chat.model_id,
            max_tokens=64000,
            stop_sequences = [STOP_SEQUENCE],
            temperature = 1,
            additional_request_fields={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
            },
        )
    elif chat.reasoning_mode=='Disable' and chat.model_type != 'openai':
        model = BedrockModel(
            boto_session=boto_session,
            boto_client_config=bedrock_config,
            model_id=chat.model_id,
            max_tokens=maxOutputTokens,
            stop_sequences = [STOP_SEQUENCE],
            temperature = 0.1,
            additional_request_fields={
                "thinking": {
                    "type": "disabled"
                }
            }
        )
    elif chat.model_type == 'openai':
        model = BedrockModel(
            model=chat.model_id,
            region=aws_region,
            streaming=True
        )
    return model

conversation_manager = SlidingWindowConversationManager(
    window_size=10,  
)

class MCPClientManager:
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        self.client_configs: Dict[str, dict] = {}  # Store client configurations
        self._persistent_stack: Optional[contextlib.ExitStack] = None
        self._persistent_client_names: List[str] = []
            
    def add_stdio_client(self, name: str, command: str, args: List[str], env: dict[str, str] = {}) -> None:
        """Add a new MCP client configuration (lazy initialization)"""
        self.client_configs[name] = {
            "transport": "stdio",
            "command": command,
            "args": args,
            "env": env
        }
    
    def add_streamable_client(self, name: str, url: str, headers: dict[str, str] = {}) -> None:
        """Add a new MCP client configuration (lazy initialization)"""
        self.client_configs[name] = {
            "transport": "streamable_http",
            "url": url,
            "headers": headers
        }
    
    def get_client(self, name: str) -> Optional[MCPClient]:
        """Get or create MCP client (lazy initialization)"""
        if name not in self.client_configs:
            logger.warning(f"No configuration found for MCP client: {name}")
            return None
            
        if name not in self.clients:
            # Create client on first use
            config = self.client_configs[name]
            logger.info(f"Creating {name} MCP client with config: {config}")
            try:
                if "transport" in config and config["transport"] == "streamable_http":
                    try:
                        self.clients[name] = MCPClient(lambda: streamablehttp_client(
                            url=config["url"], 
                            headers=config["headers"]
                        ))
                    except Exception as http_error:
                        logger.error(f"Failed to create streamable HTTP client for {name}: {http_error}")
                        if "403" in str(http_error) or "Forbidden" in str(http_error) or "MCPClientInitializationError" in str(http_error) or "client initialization failed" in str(http_error):
                            logger.error(f"Authentication failed for {name}. Attempting to refresh bearer token...")
                            
                        else:
                            raise http_error
                else:
                    self.clients[name] = MCPClient(lambda: stdio_client(
                        StdioServerParameters(
                            command=config["command"], 
                            args=config["args"], 
                            env=config["env"]
                        )
                    ))
                
                logger.info(f"Successfully created MCP client: {name}")
            except Exception as e:
                logger.error(f"Failed to create MCP client {name}: {e}")
                logger.error(f"Exception type: {type(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None
        else:
            # Check if client is already running and stop it if necessary
            try:
                client = self.clients[name]
                if hasattr(client, '_session') and client._session is not None:
                    logger.info(f"Stopping existing session for client: {name}")
                    try:
                        client.stop()
                    except Exception as stop_error:
                        # Ignore 404 errors during session termination (common with AWS Bedrock AgentCore)
                        if "404" in str(stop_error) or "Not Found" in str(stop_error):
                            logger.info(f"Session already terminated for {name} (404 expected)")
                        else:
                            logger.warning(f"Error stopping existing client session for {name}: {stop_error}")
            except Exception as e:
                logger.warning(f"Error checking client session for {name}: {e}")
        return self.clients[name]
    
    def remove_client(self, name: str) -> None:
        """Remove an MCP client"""
        if name in self.clients:
            del self.clients[name]
        if name in self.client_configs:
            del self.client_configs[name]
    
    def start_agent_clients(self, client_names: List[str]) -> bool:
        """Start MCP clients persistently. Only restarts if client list changed."""
        if self._persistent_stack and set(self._persistent_client_names) == set(client_names):
            logger.info(f"Persistent MCP clients already running: {client_names}")
            return False
        
        self.stop_agent_clients()
        
        logger.info(f"Starting persistent MCP clients: {client_names}")
        self._persistent_stack = contextlib.ExitStack()
        
        for name in client_names:
            client = self.get_client(name)
            if client:
                try:
                    if hasattr(client, '_session') and client._session is not None:
                        try:
                            client.stop()
                        except Exception:
                            pass
                    self._persistent_stack.enter_context(client)
                    logger.info(f"client started: {name}")
                except Exception as e:
                    logger.error(f"Error starting client {name}: {e}")
        
        self._persistent_client_names = list(client_names)
        return True
    
    def stop_agent_clients(self):
        """Stop all persistent MCP clients."""
        if self._persistent_stack:
            logger.info(f"Stopping persistent MCP clients: {self._persistent_client_names}")
            try:
                self._persistent_stack.close()
            except Exception as e:
                logger.warning(f"Error stopping persistent clients: {e}")
            self._persistent_stack = None
            self._persistent_client_names = []
    
    @contextmanager
    def get_active_clients(self, active_clients: List[str]):
        """Manage active clients context"""
        
        # Reuse persistent clients if the same set is already running
        if self._persistent_stack and set(self._persistent_client_names) == set(active_clients):
            logger.info(f"Reusing MCP clients")
            yield
            return
        
        active_contexts = []
        try:
            for client_name in active_clients:
                client = self.get_client(client_name)
                if client:
                    # Ensure client is not already running
                    try:
                        if hasattr(client, '_session') and client._session is not None:
                            logger.info(f"Stopping existing session for client: {client_name}")
                            try:
                                client.stop()
                            except Exception as stop_error:
                                # Ignore 404 errors during session termination (common with AWS Bedrock AgentCore)
                                if "404" in str(stop_error) or "Not Found" in str(stop_error):
                                    logger.info(f"Session already terminated for {client_name} (404 expected)")
                                else:
                                    logger.warning(f"Error stopping existing session for {client_name}: {stop_error}")
                    except Exception as e:
                        logger.warning(f"Error checking existing session for {client_name}: {e}")
                    
                    active_contexts.append(client)

            # logger.info(f"active_contexts: {active_contexts}")
            if active_contexts:
                with contextlib.ExitStack() as stack:
                    for client in active_contexts:
                        try:
                            stack.enter_context(client)
                        except Exception as e:
                            logger.error(f"Error entering context for client: {e}")
                            
                            # Check if this is a 403 error and try to refresh bearer token
                            logger.info(f"Error details: {type(e).__name__}: {str(e)}")
                            if "403" in str(e) or "Forbidden" in str(e) or "MCPClientInitializationError" in str(e) or "client initialization failed" in str(e):
                                logger.info("403 error detected, attempting to refresh bearer token...")
                                try:
                                    # Find the client name from the active_clients list
                                    client_name = None
                                    for name, client_obj in mcp_manager.clients.items():
                                        if client_obj == client:
                                            client_name = name
                                            break
                                                                        
                                except Exception as retry_error:
                                    logger.error(f"Error during bearer token refresh and retry: {retry_error}")
                            
                            # Try to stop the client if it's already running
                            try:
                                if hasattr(client, 'stop'):
                                    try:
                                        client.stop()
                                    except Exception as stop_error:
                                        # Ignore 404 errors during session termination
                                        if "404" in str(stop_error) or "Not Found" in str(stop_error):
                                            logger.info(f"Session already terminated (404 expected)")
                                        else:
                                            logger.warning(f"Error stopping client: {stop_error}")
                            except:
                                pass
                            raise
                    yield
            else:
                yield
        except Exception as e:
            logger.error(f"Error in MCP client context: {e}")
            raise

# Initialize MCP client manager
mcp_manager = MCPClientManager()

# Set up MCP clients
def init_mcp_clients(mcp_servers: list):
    for tool in mcp_servers:
        logger.info(f"Initializing MCP client for tool: {tool}")
        config = mcp_config.load_config(tool)
        # logger.info(f"config: {config}")

        # Skip if config is empty or doesn't have mcpServers
        if not config or "mcpServers" not in config:
            logger.warning(f"No configuration found for tool: {tool}")
            continue

        # Get the first key from mcpServers
        server_key = next(iter(config["mcpServers"]))
        server_config = config["mcpServers"][server_key]
        
        if "type" in server_config and server_config["type"] == "streamable_http":
            name = tool  # Use tool name as client name
            url = server_config["url"]
            headers = server_config.get("headers", {})                
            logger.info(f"Adding MCP client - name: {name}, url: {url}, headers: {headers}")
                
            try:                
                mcp_manager.add_streamable_client(name, url, headers)
                logger.info(f"Successfully added streamable MCP client for {name}")
            except Exception as e:
                logger.error(f"Failed to add streamable MCP client for {name}: {e}")
                
        else:
            name = tool  # Use tool name as client name
            command = server_config["command"]
            args = server_config["args"]
            env = server_config.get("env", {})  # Use empty dict if env is not present            
            logger.info(f"name: {name}, command: {command}, args: {args}, env: {env}")        

            try:
                mcp_manager.add_stdio_client(name, command, args, env)
                logger.info(f"Successfully added {name} MCP client")
            except Exception as e:
                logger.error(f"Failed to add stdio MCP client for {name}: {e}")
                continue
                            
def update_tools(strands_tools: list, mcp_servers: list):
    tools = []
    tool_map = {
        "calculator": calculator,
        "current_time": current_time
    }

    for tool_item in strands_tools:
        if isinstance(tool_item, list):
            tools.extend(tool_item)
        elif isinstance(tool_item, str) and tool_item in tool_map:
            tools.append(tool_map[tool_item])

    # MCP tools
    mcp_servers_loaded = 0
    for mcp_tool in mcp_servers:
        logger.info(f"Processing MCP tool: {mcp_tool}")        
        try:
            with mcp_manager.get_active_clients([mcp_tool]) as _:
                client = mcp_manager.get_client(mcp_tool)
                if client:
                    logger.info(f"Got client for {mcp_tool}, attempting to list tools...")
                    try:
                        mcp_servers_list = client.list_tools_sync()
                        # logger.info(f"{mcp_tool}_tools: {mcp_servers_list}")
                        if mcp_servers_list:
                            tools.extend(mcp_servers_list)
                            mcp_servers_loaded += 1
                            logger.info(f"Successfully added {len(mcp_servers_list)} tools from {mcp_tool} server")
                        else:
                            logger.warning(f"No tools returned from {mcp_tool}")
                    except Exception as tool_error:
                        logger.error(f"Error listing tools for {mcp_tool}: {tool_error}")
                        continue
                else:
                    logger.error(f"Failed to get client for {mcp_tool}")
        except Exception as e:
            logger.error(f"Error getting tools for {mcp_tool}: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    return tools

def create_agent(system_prompt, tools):
    if system_prompt==None:
        system_prompt = (
            "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )

    if not system_prompt or not system_prompt.strip():
        system_prompt = "You are a helpful AI assistant."
    
    # add skills metadata to system prompt
    system_prompt = build_system_prompt(system_prompt)

    model = get_model()
    
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        conversation_manager=conversation_manager,
        #max_parallel_tools=2
    )

    return agent

def get_tool_list(tools):
    tool_list = []
    for tool in tools:
        if hasattr(tool, 'tool_name'):  # MCP tool
            tool_list.append(tool.tool_name)
                
        if str(tool).startswith("<module 'strands_tools."):   # strands_tools 
            module_name = str(tool).split("'")[1].split('.')[-1]
            tool_list.append(module_name)
    return tool_list

async def initiate_agent(system_prompt, strands_tools, mcp_servers):
    global agent, initiated
    global selected_strands_tools, selected_mcp_servers, tool_list

    update_required = False
    if selected_strands_tools != strands_tools:
        selected_strands_tools = strands_tools
        update_required = True
        logger.info(f"strands_tools: {strands_tools}")

    if selected_mcp_servers != mcp_servers:
        selected_mcp_servers = mcp_servers
        update_required = True
        logger.info(f"mcp_servers: {mcp_servers}")

    if not initiated or update_required:
        mcp_manager.stop_agent_clients()
        
        init_mcp_clients(mcp_servers)
        tools = update_tools(strands_tools, mcp_servers)
        logger.info(f"tools: {tools}")

        if chat.skill_mode == 'Enable':
            builtin_tools = get_builtin_tools()
            logger.info(f"builtin_tools: {builtin_tools}")
            
            tool_names = {tool.tool_name for tool in tools}
            for bt in builtin_tools:
                if bt.tool_name not in tool_names:
                    tools.append(bt)
                else:
                    logger.info(f"builtin_tool {bt.tool_name} already in tools")

        agent = create_agent(system_prompt, tools)
        tool_list = get_tool_list(tools)

        if not initiated:
            logger.info("create agent!")
            initiated = True
        else:
            logger.info("update agent!")
            update_required = False
    
    # Start or reuse persistent MCP clients
    mcp_manager.start_agent_clients(mcp_servers)

