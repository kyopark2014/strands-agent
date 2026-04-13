import chat
import os
import contextlib
import mcp_config
import logging
import sys
import utils
import boto3
import yaml
import skill
import chat
import subprocess

from contextlib import contextmanager
from typing import Dict, List, Optional
from strands.models import BedrockModel
from strands_tools import current_time, file_read, file_write
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

strands_tools = []
mcp_servers = []

tool_list = []

memory_id = actor_id = session_id = namespace = None

s3_prefix = "docs"
capture_prefix = "captures"

aws_region = utils.bedrock_region

config = utils.load_config()
s3_bucket = config.get("s3_bucket")
sharing_url = config.get("sharing_url")

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(WORKING_DIR, "skills")
ARTIFACTS_DIR = os.path.join(WORKING_DIR, "artifacts")


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

def build_system_prompt(plugin_name: Optional[str] = None, command: Optional[str] = None) -> str:
    """Assemble the full system prompt with available skills metadata."""
    if command:
        base = skill.build_command_prompt(plugin_name, command)
    elif plugin_name:
        base = skill.build_skill_prompt(plugin_name)
    else:
        base = BASE_SYSTEM_PROMPT

    return base

def s3_uri_to_console_url(uri: str, region: str) -> str:
    """Open the object in the AWS S3 console (when sharing_url is not configured)."""
    if not uri or not uri.startswith("s3://"):
        return ""
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    enc_key = parse.quote(key, safe="")
    return f"https://{region}.console.aws.amazon.com/s3/object/{bucket}?prefix={enc_key}"

import io, os, sys, json, traceback
import subprocess as _subprocess, pathlib as _pathlib, shutil as _shutil
import tempfile as _tempfile, glob as _glob, datetime as _datetime
import math as _math, re as _re, requests as _requests
from pathlib import Path

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(WORKING_DIR, "artifacts")

_ARTIFACT_EXT = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"})

_mpl_runtime_ready = False

def _artifact_files_mtime_snapshot() -> dict:
    """Relative path from WORKING_DIR -> mtime. Only scans under artifacts/."""
    snap = {}
    if not os.path.isdir(ARTIFACTS_DIR):
        return snap
    for dirpath, _, filenames in os.walk(ARTIFACTS_DIR):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                rel = os.path.relpath(full, WORKING_DIR)
                snap[rel] = os.path.getmtime(full)
            except OSError:
                pass
    return snap


def _touched_artifact_paths(before: dict, after: dict) -> list:
    """Only files created or modified between pre/post execution snapshots."""
    touched = []
    for rel, mt in after.items():
        if rel not in before or before[rel] != mt:
            touched.append(rel)
    return sorted(touched)


def _paths_for_ui(relative_paths: list) -> list:
    """absolute path for Streamlit st.image."""
    out = []
    for rel in relative_paths:
            out.append(os.path.abspath(os.path.join(WORKING_DIR, rel)))
    return out


def _ensure_matplotlib_runtime():
    """Use non-interactive Agg backend, prefer CJK-capable fonts, silence headless/show noise."""
    global _mpl_runtime_ready
    if _mpl_runtime_ready:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")

        import warnings

        warnings.filterwarnings(
            "ignore",
            message=r"Glyph .* missing from font",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"FigureCanvasAgg is non-interactive.*",
            category=UserWarning,
        )

        import matplotlib.font_manager as fm
        import matplotlib as mpl

        mpl.rcParams["axes.unicode_minus"] = False
        cjk_candidates = (
            "AppleGothic",
            "Apple SD Gothic Neo",
            "Malgun Gothic",
            "NanumGothic",
            "NanumBarunGothic",
            "Noto Sans CJK KR",
            "Noto Sans KR",
        )
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = list(cjk_candidates) + ["DejaVu Sans", "sans-serif"]

        _mpl_runtime_ready = True
    except Exception as e:
        logger.info(f"matplotlib runtime setup skipped: {e}")
        _mpl_runtime_ready = True

_exec_globals = {
    "__builtins__": __builtins__,
    "subprocess": _subprocess,
    "json": json,
    "os": os,
    "sys": sys,
    "io": io,
    "pathlib": _pathlib,
    "shutil": _shutil,
    "tempfile": _tempfile,
    "glob": _glob,
    "datetime": _datetime,
    "math": _math,
    "re": _re,
    "requests": _requests,
    "WORKING_DIR": WORKING_DIR,
    "ARTIFACTS_DIR": ARTIFACTS_DIR,
}

@tool
def execute_code(code: str) -> str:
    """Execute Python code and return stdout/stderr output.

    Use this tool to run Python code for tasks such as processing data,
    processing data, or performing computations. The execution environment
    has access to common libraries: pandas, numpy, matplotlib, seaborn, etc.
    json, csv, os, requests, etc.

    Variables and imports from previous calls persist across invocations.
    Generated files should be saved to the 'artifacts/' directory.

    Path variables (pre-defined, do NOT redefine):
    - WORKING_DIR: absolute path to application directory
    - ARTIFACTS_DIR: absolute path to artifacts directory (WORKING_DIR/artifacts)

    Args:
        code: Python code to execute.

    Returns:
        Captured stdout output, or error traceback if execution failed.
        If there is a result file, return the path of the file.            
    """
    logger.info(f"###### execute_code ######")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    before_files = _artifact_files_mtime_snapshot()

    old_cwd = os.getcwd()
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        os.chdir(WORKING_DIR)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture

        _ensure_matplotlib_runtime()
        exec(code, _exec_globals)

        sys.stdout, sys.stderr = old_stdout, old_stderr
        os.chdir(old_cwd)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        result = ""
        if output:
            result += output
        if errors:
            result += f"\n[stderr]\n{errors}"
        if not result.strip():
            result = "Code executed successfully (no output)."

        after_files = _artifact_files_mtime_snapshot()
        touched = _touched_artifact_paths(before_files, after_files)
        artifact_rels = [
            r
            for r in touched
            if os.path.splitext(r)[1].lower() in _ARTIFACT_EXT
        ]
        other_rels = [r for r in touched if r not in artifact_rels]
        if other_rels:
            lines = "\n".join(
                os.path.abspath(os.path.join(WORKING_DIR, r)) for r in other_rels
            )
            result += f"\n[artifacts]\n{lines}"

        if artifact_rels:
            payload = {"output": result.strip()}
            payload["path"] = _paths_for_ui(artifact_rels)
            return json.dumps(payload, ensure_ascii=False)

        return result

    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        os.chdir(old_cwd)
        tb = traceback.format_exc()
        logger.error(f"Code execution error: {tb}")
        return f"Error executing code:\n{tb}"


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
            return "S3 bucket is not configured."

        full_path = os.path.join(WORKING_DIR, filepath)
        if not os.path.exists(full_path):
            return f"File not found: {filepath}"

        content_type = utils.get_contents_type(filepath)
        s3 = boto3.client("s3", region_name=config.get("region", "us-west-2"))

        with open(full_path, "rb") as f:
            s3.put_object(Bucket=s3_bucket, Key=filepath, Body=f.read(), ContentType=content_type)

        if sharing_url:
            url = f"{sharing_url}/{url_parse.quote(filepath)}"
            return f"Upload complete: {url}"
        return f"Upload complete: {s3_uri_to_console_url(f"s3://{s3_bucket}/{filepath}", config.get("region", "us-west-2"))}"

    except Exception as e:
        return f"Upload failed: {str(e)}"

@tool
def memory_search(query: str, max_results: int = 5, min_score: float = 0.0) -> str:
    """Search across memory files (MEMORY.md and memory/*.md) for relevant information.

    Performs keyword-based search over all memory files and returns matching snippets
    ranked by relevance score.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default: 5).
        min_score: Minimum relevance score threshold 0.0-1.0 (default: 0.0).

    Returns:
        JSON array of matching snippets with text, path, from (line), lines, and score.
    """
    import re as _re
    logger.info(f"###### memory_search: {query} ######")

    memory_root = Path(WORKING_DIR)
    memory_dir = memory_root / "memory"

    target_files = []
    memory_md = memory_root / "MEMORY.md"
    if memory_md.exists():
        target_files.append(memory_md)
    if memory_dir.exists():
        target_files.extend(sorted(memory_dir.glob("*.md"), reverse=True))

    if not target_files:
        return json.dumps([], ensure_ascii=False)

    query_lower = query.lower()
    query_tokens = [t for t in _re.split(r'\s+', query_lower) if len(t) >= 2]

    results = []
    for fpath in target_files:
        try:
            content = fpath.read_text(encoding="utf-8")
        except Exception:
            continue

        lines = content.split("\n")
        content_lower = content.lower()

        if not any(tok in content_lower for tok in query_tokens):
            continue

        window_size = 5
        for i in range(0, len(lines), window_size):
            chunk_lines = lines[i:i + window_size]
            chunk_text = "\n".join(chunk_lines)
            chunk_lower = chunk_text.lower()

            matched_tokens = sum(1 for tok in query_tokens if tok in chunk_lower)
            if matched_tokens == 0:
                continue

            score = matched_tokens / len(query_tokens) if query_tokens else 0.0

            if score >= min_score:
                rel_path = str(fpath.relative_to(memory_root))
                results.append({
                    "text": chunk_text.strip(),
                    "path": rel_path,
                    "from": i + 1,
                    "lines": len(chunk_lines),
                    "score": round(score, 3),
                })

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:max_results]

    return json.dumps(results, indent=2, ensure_ascii=False)


@tool
def memory_get(path: str, from_line: int = 0, lines: int = 0) -> str:
    """Read a specific memory file (MEMORY.md or memory/*.md).

    Use after memory_search to get full context, or when you know the exact file path.

    Args:
        path: Workspace-relative path (e.g. "MEMORY.md", "memory/2026-03-02.md").
        from_line: Starting line number, 1-indexed (0 = read from beginning).
        lines: Number of lines to read (0 = read entire file).

    Returns:
        JSON with 'text' (file content) and 'path'. Returns empty text if file doesn't exist.
    """
    logger.info(f"###### memory_get: {path} ######")

    full_path = Path(WORKING_DIR) / path

    if not full_path.exists():
        return json.dumps({"text": "", "path": path}, ensure_ascii=False)

    try:
        content = full_path.read_text(encoding="utf-8")

        if from_line > 0 or lines > 0:
            all_lines = content.split("\n")
            start = max(0, from_line - 1)
            if lines > 0:
                end = start + lines
                content = "\n".join(all_lines[start:end])
            else:
                content = "\n".join(all_lines[start:])

        return json.dumps({"text": content, "path": path}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"text": f"Error reading file: {e}", "path": path}, ensure_ascii=False)


@tool
def get_skill_instructions(plugin_name: str, skill_name: str) -> str:
    """Load the full instructions for a specific skill by name.

    Use this when you need detailed instructions for a task that matches
    one of the available skills listed in the system prompt.

    Args:
        plugin_name: The plugin name (e.g. 'base', 'frontend-design').
        skill_name: The name of the skill to load (e.g. 'pdf').

    Returns:
        The full skill instructions, or an error message if not found.
    """
    logger.info(f"###### get_skill_instructions: {skill_name} (plugin={plugin_name}) ######")
    skill_mgr = skill.skill_managers.get(plugin_name)
    if skill_mgr is None:
        if plugin_name == "base":
            skills_dir = skill.SKILLS_DIR
        else:
            skills_dir = os.path.join(WORKING_DIR, "plugins", plugin_name, "skills")
        skill_mgr = skill.SkillManager(skills_dir)
        skill.skill_managers[plugin_name] = skill_mgr

    instructions = skill_mgr.get_skill_instructions(skill_name)
    if instructions:
        return instructions

    base_mgr = skill.skill_managers.get("base")
    if base_mgr is None:
        base_mgr = skill.SkillManager(skill.SKILLS_DIR)
        skill.skill_managers["base"] = base_mgr
    instructions = base_mgr.get_skill_instructions(skill_name)
    if instructions:
        return instructions

    available = ", ".join(skill_mgr.registry.keys())
    return f"Skill '{skill_name}' not found. Available skills: {available}"

def _ensure_cli_scripts_on_path() -> None:
    """Prepend pip user script dir so CLIs (e.g. browser-use) resolve in subprocess."""
    import site
    import sysconfig

    extra: list[str] = []
    user_base = getattr(site, "USER_BASE", None)
    if user_base:
        user_bin = os.path.join(user_base, "bin")
        if os.path.isdir(user_bin):
            extra.append(user_bin)
    try:
        scripts = sysconfig.get_path("scripts")
        if scripts and os.path.isdir(scripts):
            extra.append(scripts)
    except Exception:
        pass
    path = os.environ.get("PATH", "")
    parts = [p for p in path.split(os.pathsep) if p]
    for d in reversed(extra):
        if d and d not in parts:
            parts.insert(0, d)
    os.environ["PATH"] = os.pathsep.join(parts)


@tool
def bash(command: str) -> str:
    """Execute a bash command and return the result"""
    logger.info(f"###### bash: {command} ######")
    _ensure_cli_scripts_on_path()
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        cwd=WORKING_DIR, timeout=300,
        env=os.environ,
    )
    parts = []
    if result.stdout:
        parts.append(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")
    if result.returncode != 0:
        parts.append(f"Return code: {result.returncode}")
    return "\n".join(parts) if parts else "(no output)"

def get_builtin_tools() -> list:
    """Return the list of built-in tools for the skill-aware agent."""
    return [execute_code, bash, upload_file_to_s3]

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

            # Skip if command is a file path and the executable doesn't exist
            cmd_path = os.path.expanduser(command) if isinstance(command, str) else str(command)
            if "/" in cmd_path or (isinstance(command, str) and command.startswith("~")):
                if not os.path.isfile(cmd_path):
                    logger.warning(f"Skipping {name}: executable not found at {cmd_path}")
                    continue

            try:
                mcp_manager.add_stdio_client(name, command, args, env)
                logger.info(f"Successfully added {name} MCP client")
            except Exception as e:
                logger.error(f"Failed to add stdio MCP client for {name}: {e}")
                continue
                            
def update_tools(strands_tools: list, mcp_servers: list):
    # builtin tools
    tools = get_builtin_tools()
        
    tool_map = {
        "current_time": current_time,
        "file_read": file_read,
        "file_write": file_write
    }

    for tool_item in strands_tools:
        if isinstance(tool_item, str):
            if tool_item in tool_map:
                tools.append(tool_map[tool_item])
            else:
                logger.warning(f"Unknown string tool: {tool_item}")
            continue

        if isinstance(tool_item, list):
            tools.extend(tool_item)
            continue

        if hasattr(tool_item, 'tool_name') and tool_item.tool_name in [t.tool_name if hasattr(t, 'tool_name') else str(t) for t in tools]:
            logger.info(f"builtin tool {tool_item.tool_name} already in tools")
            continue

        tools.append(tool_item)

    # MCP tools
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

                        for mcp_server_item in mcp_servers_list:
                            if mcp_server_item.tool_name in tools:
                                logger.info(f"{mcp_server_item.tool_name} already in tools")
                                continue

                            tools.append(mcp_server_item)
                            logger.info(f"Successfully added {mcp_server_item.tool_name} from {mcp_tool} server")
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

BASE_SYSTEM_PROMPT = (
    "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
    "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
    "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    "결과 파일이 있으면 upload_file_to_s3로 업로드하여 URL을 제공합니다."
)

def create_agent(tools: list, plugin_name: Optional[str], command: Optional[str] = None):
    # add skills metadata to system prompt
    system_prompt = build_system_prompt(plugin_name, command)

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


selected_strands_tools = []
selected_mcp_servers = []
active_plugin = None

async def run_strands_agent(query: str, strands_tools: list[str], mcp_servers: list[str], plugin_name: Optional[str], containers: dict):
    """Run the strands agent with streaming and tool notifications. Uses chat module for UI callbacks."""
    chat.index = 0
    chat.tool_info_list.clear()
    chat.tool_name_list.clear()

    image_url = []
    references = []
    index = 0

    global agent, selected_strands_tools, selected_mcp_servers, active_plugin

    if selected_strands_tools != strands_tools or selected_mcp_servers != mcp_servers or active_plugin != plugin_name:        
        selected_strands_tools = strands_tools
        selected_mcp_servers = mcp_servers
        active_plugin = plugin_name
        
        mcp_manager.stop_agent_clients()
        
        init_mcp_clients(mcp_servers)

        tools = update_tools(strands_tools, mcp_servers)

        if chat.skill_mode == 'Enable':
            tools.append(get_skill_instructions)

        agent = create_agent(tools, plugin_name)
        tool_list = get_tool_list(tools)
    
        # Start or reuse persistent MCP clients
        mcp_manager.start_agent_clients(mcp_servers)

    # run agent
    final_result = current = ""
    with mcp_manager.get_active_clients(mcp_servers) as _:
        agent_stream = agent.stream_async(query)

        async for event in agent_stream:
            text = ""
            if "data" in event:
                text = event["data"]
                logger.info(f"[data] {text}")
                current += text
                chat.index = index
                chat.update_streaming_result(containers, current)

            elif "result" in event:
                final = event["result"]
                message = final.message
                if message:
                    content = message.get("content", [])
                    result = content[0].get("text", "")
                    logger.info(f"[result] {result}")
                    final_result = result

            elif "current_tool_use" in event:
                current_tool_use = event["current_tool_use"]
                logger.info(f"current_tool_use: {current_tool_use}")
                name = current_tool_use.get("name", "")
                input_val = current_tool_use.get("input", "")
                toolUseId = current_tool_use.get("toolUseId", "")

                text = f"name: {name}, input: {input_val}"
                logger.info(f"[current_tool_use] {text}")

                if toolUseId not in chat.tool_info_list:
                    index += 1
                    current = ""
                    logger.info(f"new tool info: {toolUseId} -> {index}")
                    chat.tool_info_list[toolUseId] = index
                    chat.tool_name_list[toolUseId] = name
                    chat.index = index
                    chat.add_notification(containers, f"Tool: {name}, Input: {input_val}")
                    index = chat.index
                else:
                    logger.info(f"overwrite tool info: {toolUseId} -> {chat.tool_info_list[toolUseId]}")
                    containers['notification'][chat.tool_info_list[toolUseId]].info(f"Tool: {name}, Input: {input_val}")

            elif "message" in event:
                message = event["message"]
                logger.info(f"[message] {message}")

                if "content" in message:
                    content = message["content"]
                    logger.info(f"tool content: {content}")
                    if "toolResult" in content[0]:
                        toolResult = content[0]["toolResult"]
                        toolUseId = toolResult["toolUseId"]
                        toolContent = toolResult["content"]
                        toolResult = toolContent[0].get("text", "")
                        tool_name = chat.tool_name_list[toolUseId]
                        logger.info(f"[toolResult] {toolResult}, [toolUseId] {toolUseId}")
                        chat.add_notification(containers, f"Tool Result: {str(toolResult)}")

                        content, urls, refs = chat.get_tool_info(tool_name, toolResult)
                        if refs:
                            for r in refs:
                                references.append(r)
                            logger.info(f"refs: {refs}")
                        if urls:
                            for url in urls:
                                image_url.append(url)
                            logger.info(f"urls: {urls}")

                        if content:
                            logger.info(f"content: {content}")

            elif "contentBlockDelta" or "contentBlockStop" or "messageStop" or "metadata" in event:
                pass

            else:
                logger.info(f"event: {event}")

        if references:
            ref = "\n\n### Reference\n"
            for i, reference in enumerate(references):
                content = reference['content'][:100].replace("\n", "")
                ref += f"{i+1}. [{reference['title']}]({reference['url']}), {content}...\n"
            final_result += ref

        if containers is not None:
            containers['notification'][chat.index].markdown(final_result)

    return final_result, image_url

