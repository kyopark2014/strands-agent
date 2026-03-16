import os
import json
import yaml
import logging
import sys
import traceback
import io
import utils
import io
import json
import yaml

from pathlib import Path
from dataclasses import dataclass
from pathlib import Path
from dataclasses import dataclass, field
from langchain_core.tools import tool
from typing import Literal, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("skill")

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(WORKING_DIR, "skills")
ARTIFACTS_DIR = os.path.join(WORKING_DIR, "artifacts")

config = utils.load_config()
sharing_url = config.get("sharing_url")

# ═══════════════════════════════════════════════════════════════════
#  Skill Manager – implementation of Anthropic Agent Skills spec
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
        self._discover(skills_dir)

    # ---- discovery & metadata loading ----
    def _discover(self, skills_dir: str):
        """Scan a skills directory and load metadata (frontmatter only) into registry."""
        if not os.path.isdir(skills_dir):
            logger.info(f"skills directory is not found: {skills_dir}")
            return

        for entry in os.listdir(skills_dir):
            skill_md = os.path.join(skills_dir, entry, "SKILL.md")
            if os.path.isfile(skill_md):
                try:
                    meta, instructions = self._parse_skill_md(skill_md)
                    skill = Skill(
                        name=meta.get("name", entry),
                        description=meta.get("description", ""),
                        instructions=instructions,
                        path=os.path.join(skills_dir, entry),
                    )
                    self.registry[skill.name] = skill
                    logger.info(f"Skill discovered: {skill.name}")
                except Exception as e:
                    logger.warning(f"Failed to load skill '{entry}': {e}")

    def discover_plugin_skills(self, skills_dir: str):
        """Scan a plugin's skills directory and add to registry (merge, do not replace)."""
        if not os.path.isdir(skills_dir):
            return
        for entry in os.listdir(skills_dir):
            skill_md = os.path.join(skills_dir, entry, "SKILL.md")
            if os.path.isfile(skill_md):
                try:
                    meta, instructions = self._parse_skill_md(skill_md)
                    skill = Skill(
                        name=meta.get("name", entry),
                        description=meta.get("description", ""),
                        instructions=instructions,
                        path=os.path.join(skills_dir, entry),
                    )
                    self.registry[skill.name] = skill
                    logger.info(f"Plugin skill discovered: {skill.name}")
                except Exception as e:
                    logger.warning(f"Failed to load plugin skill '{entry}': {e}")

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

    def get_skill_instructions(self, name: str) -> Optional[str]:
        """Return full instructions for a skill (loaded on demand)."""
        skill = self.registry.get(name)
        return skill.instructions if skill else None

# define global skill_managers
skill_managers: dict[str, SkillManager] = {}

def get_skills_xml(skill_info: list) -> str:
    lines = ["<available_skills>"]
    for s in skill_info:
        lines.append("  <skill>")
        lines.append(f"    <name>{s['name']}</name>")
        lines.append(f"    <description>{s['description']}</description>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)

def register_plugin_skills(plugin_name: str):
    """Register skills from a plugin's skills directory into SkillManager's registry."""    
    if plugin_name == "base": # base skills
        skills_dir = SKILLS_DIR
    else:   # plugin skills
        skills_dir = os.path.join(WORKING_DIR, "plugins", plugin_name, "skills")
    
    skill_manager = skill_managers.get(plugin_name)
    if skill_manager is None:
        skill_manager = SkillManager(skills_dir)
        skill_managers[plugin_name] = skill_manager

    skill_manager.discover_plugin_skills(skills_dir)


def available_skill_info(plugin_name: str) -> list:
    skill_manager = skill_managers.get(plugin_name)
    if skill_manager is None:
        if plugin_name == "base": # base skills
            skills_dir = SKILLS_DIR
        else:   # plugin skills
            skills_dir = os.path.join(WORKING_DIR, "plugins", plugin_name, "skills")
        skill_manager = SkillManager(skills_dir)
        skill_managers[plugin_name] = skill_manager

    registry = skill_manager.registry
    
    if not registry:
        return []
    
    skill_info = []
    for s in registry.values():
        skill_info.append({"name": s.name, "description": s.description})
        
    return skill_info


def selected_skill_info(plugin_name: str) -> list:
    config = utils.load_config()
    if plugin_name == "base":
        skill_list = config.get("default_skills") or []
    else:   # plugin skills
        skill_list = config.get("plugin_skills", {}).get(plugin_name) or []
    logger.info(f"plugin_name: {plugin_name}, skill_list: {skill_list}")

    skill_info = available_skill_info(plugin_name)

    selected_skill_info = []
    for s in skill_info:
        if s["name"] in skill_list:
            selected_skill_info.append(s)
    return selected_skill_info


SKILL_SYSTEM_PROMPT = (
    "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다.\n"
    "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다.\n"
    "모르는 질문을 받으면 솔직히 모른다고 말합니다.\n"
    "한국어로 답변하세요.\n\n"
    "## Agent Workflow\n"
    "1. 사용자 입력을 받는다\n"
    "2. 요청에 맞는 skill이 있으면 get_skill_instructions 도구로 상세 지침을 로드한다\n"
    "3. skill 지침에 따라 execute_code, write_file 등의 도구를 사용하여 작업을 수행한다\n"
    "4. 결과 파일이 있으면 upload_file_to_s3로 업로드하여 URL을 제공한다\n"
    "5. 최종 결과를 사용자에게 전달한다\n\n"
)

SKILL_USAGE_GUIDE = (
    "\n## Skill 사용 가이드\n"
    "위의 <available_skills>에 나열된 skill이 사용자의 요청과 관련될 때:\n"
    "1. 먼저 get_skill_instructions 도구로 해당 skill의 상세 지침을 로드하세요.\n"
    "2. 지침에 포함된 코드 패턴을 execute_code 도구로 실행하세요.\n"
    "3. skill 지침이 없는 일반 질문은 직접 답변하세요.\n"
)

def build_skill_prompt(plugin_name: str) -> str:
    """Build skill-related prompt: path info, available skills XML, and usage guide."""
    skill_info = selected_skill_info(plugin_name)
    logger.info(f"plugin_name: {plugin_name}, skill_info: {skill_info}")

    if plugin_name != "base":
        default_skill_info = selected_skill_info("base")
        if default_skill_info:
            skill_info.extend(default_skill_info)
            logger.info(f"default_skill_info: {default_skill_info}")

    path_info = (
        f"## Paths (use absolute paths for write_file, read_file)\n"
        f"- WORKING_DIR: {WORKING_DIR}\n"
        f"- ARTIFACTS_DIR: {ARTIFACTS_DIR}\n"
        f"Example: write_file(filepath='{os.path.join(ARTIFACTS_DIR, 'report.drawio')}', content='...')\n\n"
    )

    skills_xml = get_skills_xml(skill_info)
    if skills_xml:
        return f"{SKILL_SYSTEM_PROMPT}\n{path_info}\n{skills_xml}\n{SKILL_USAGE_GUIDE}"
    return f"{SKILL_SYSTEM_PROMPT}\n{path_info}"

def get_command_instructions(plugin_name: str, command_name: str) -> str:
    """Load the full instructions for a specific command by name.

    Use this when you need detailed instructions for a command.
    """
    logger.info(f"###### get_command_instructions: {command_name} ######")

    commands_dir = os.path.join(WORKING_DIR, "plugins", plugin_name, "commands")
    if not os.path.isdir(commands_dir):
        return f"Plugin '{plugin_name}' has no commands directory."

    command_name_normalized = command_name.lower().strip()
    filepath = os.path.join(commands_dir, f"{command_name_normalized}.md")

    if not os.path.isfile(filepath):
        available = [
            p[:-3] for p in os.listdir(commands_dir)
            if p.endswith(".md")
        ]
        return f"Command '{command_name}' not found. Available commands: {', '.join(available)}"

    frontmatter, body = SkillManager._parse_skill_md(filepath)
    # Return body (instructions); optionally prefix with frontmatter summary
    if frontmatter:
        desc = frontmatter.get("description", "")
        hint = frontmatter.get("argument-hint", "")
        header = f"**{desc}**\n"
        if hint:
            header += f"Argument hint: {hint}\n\n"
        return header + body
    return body

COMMAND_USAGE_GUIDE = (
    "\n## Command 사용 가이드\n"
    "위의 <command_instructions>에 따라 사용자 요청을 처리하세요.\n"
    "필요한 경우 get_skill_instructions로 skill 지침을 추가 로드하거나, execute_code, write_file 등 도구를 사용하세요.\n"
)


def build_command_prompt(plugin_name: str, command: str) -> str:
    """Build prompt for command mode: path info, command instructions, and available skills."""
    skill_info = selected_skill_info(plugin_name)
    logger.info(f"plugin_name: {plugin_name}, command: {command}, skill_info: {skill_info}")

    if plugin_name != "base":
        default_skill_info = selected_skill_info("base")
        if default_skill_info:
            skill_info.extend(default_skill_info)
            logger.info(f"default_skill_info: {default_skill_info}")

    path_info = (
        f"## Paths (use absolute paths for write_file, read_file)\n"
        f"- WORKING_DIR: {WORKING_DIR}\n"
        f"- ARTIFACTS_DIR: {ARTIFACTS_DIR}\n"
        f"Example: write_file(filepath='{os.path.join(ARTIFACTS_DIR, 'report.drawio')}', content='...')\n\n"
    )

    command_instructions = get_command_instructions(plugin_name, command)
    command_section = f"## Command Instructions\n<command_instructions>\n{command_instructions}\n</command_instructions>\n\n"

    skills_xml = get_skills_xml(skill_info)
    skills_section = f"{skills_xml}\n" if skills_xml else ""

    return f"{SKILL_SYSTEM_PROMPT}\n{path_info}\n{command_section}\n{skills_section}\n{COMMAND_USAGE_GUIDE}"


# ═══════════════════════════════════════════════════════════════════
#  2. Built-in Tools – code execution, file I/O, S3 upload
# ═══════════════════════════════════════════════════════════════════

import subprocess as _subprocess, pathlib as _pathlib, shutil as _shutil
import tempfile as _tempfile, glob as _glob, datetime as _datetime
import math as _math, re as _re, requests as _requests

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
    "SKILLS_DIR": SKILLS_DIR,
    "ARTIFACTS_DIR": ARTIFACTS_DIR,
}

@tool
def execute_code(code: str) -> str:
    """Execute Python code and return stdout/stderr output.

    Use this tool to run Python code for tasks such as generating PDFs,
    processing data, or performing computations. The execution environment
    has access to common libraries: reportlab, pypdf, pdfplumber, pandas,
    json, csv, os, requests, etc.

    Variables and imports from previous calls persist across invocations.
    Generated files should be saved to the 'artifacts/' directory.

    Path variables (pre-defined, do NOT redefine):
    - WORKING_DIR: absolute path to application directory
    - SKILLS_DIR: absolute path to skills directory (WORKING_DIR/skills)
    Use directly: script = os.path.join(SKILLS_DIR, 'drawio/scripts/find_aws_icon.py')
    Never use os.environ.get('SKILLS_DIR', ...) — it overwrites the correct path.

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
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture

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

        return result

    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        os.chdir(old_cwd)
        tb = traceback.format_exc()
        logger.error(f"Code execution error: {tb}")
        return f"Error executing code:\n{tb}"


@tool
def write_file(filepath: str, content: str = "") -> str:
    """Write text content to a file.

    CRITICAL: content must always be passed. Calling without content will fail.
    Never call without content. Both filepath and content are required in a single call.

    Args:
        filepath: Absolute path or path relative to WORKING_DIR.
        content: The text content to write. REQUIRED - must not be omitted. Must include full file content.

    Returns:
        A success or failure message.
    """
    if not content:
        return (
            "Error: content parameter is required. "
            "Pass the full content to save in the form write_file(filepath='path', content='content_to_save')."
        )
    logger.info(f"###### write_file: {filepath} ######")
    try:
        full_path = filepath if os.path.isabs(filepath) else os.path.join(WORKING_DIR, filepath)
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        result_msg = f"File saved: {filepath}"
        return result_msg
    except Exception as e:
        return f"Failed to save file: {str(e)}"


@tool
def read_file(filepath: str) -> str:
    """Read the contents of a local file.

    Args:
        filepath: Absolute path or path relative to WORKING_DIR.

    Returns:
        The file contents as text, or an error message.
    """
    logger.info(f"###### read_file: {filepath} ######")
    try:
        full_path = filepath if os.path.isabs(filepath) else os.path.join(WORKING_DIR, filepath)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Failed to read file: {str(e)}"


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
        return f"Upload complete: s3://{s3_bucket}/{filepath}"

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
        skill_name: The name of the skill to load (e.g. 'pdf').

    Returns:
        The full skill instructions, or an error message if not found.
    """    
    logger.info(f"###### get_skill_instructions: {skill_name} ######")
    skill_manager = skill_managers.get(plugin_name)
    if skill_manager is None:
        if plugin_name == "base": # base skills
            skills_dir = SKILLS_DIR
        else:   # plugin skills
            skills_dir = os.path.join(WORKING_DIR, "plugins", plugin_name, "skills")
        skill_manager = SkillManager(skills_dir)
        skill_managers[plugin_name] = skill_manager

    instructions = skill_manager.get_skill_instructions(skill_name)
    if instructions:
        return instructions

    # fallback to base skills
    skill_manager = skill_managers.get("base")
    if skill_manager is None:
        skills_dir = SKILLS_DIR
        skill_manager = SkillManager(skills_dir)
        skill_managers["base"] = skill_manager
    instructions = skill_manager.get_skill_instructions(skill_name)
    if instructions:
        return instructions

    available = ", ".join(skill_manager.registry.keys())
    return f"Skill '{skill_name}' not found. Available skills: {available}"


def get_builtin_tools():
    """Return the list of built-in tools for the skill-aware agent."""
    return [execute_code, write_file, read_file, upload_file_to_s3, memory_search, memory_get, get_skill_instructions]

