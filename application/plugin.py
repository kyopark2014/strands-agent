import os
import json
import yaml
import logging
import sys
import skill

from langchain_core.tools import tool
from dataclasses import dataclass, field
from typing import Literal, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger("plugin-agent")

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(WORKING_DIR, "artifacts")
PLUGINS_DIR = os.path.join(WORKING_DIR, "plugins")

# ═══════════════════════════════════════════════════════════════════
#  PluginManager – load skills of plugin
# ═══════════════════════════════════════════════════════════════════
@dataclass
class PluginSkill:
    """Plugin skill metadata"""
    name: str
    description: str
    instructions: str
    path: str


class PluginManager:
    """Manager that discovers and loads skills from the plugin directory."""

    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir
        self.registry: dict[str, PluginSkill] = {}
        self._discover(skills_dir)

    def _discover(self, skills_dir: str):
        """Scan the skills directory and load SKILL.md metadata into the registry."""
        if not os.path.isdir(skills_dir):
            return

        for entry in os.listdir(skills_dir):
            skill_md = os.path.join(skills_dir, entry, "SKILL.md")
            if os.path.isfile(skill_md):
                try:
                    meta, instructions = self._parse_skill_md(skill_md)
                    skill = PluginSkill(
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
        """Parse YAML frontmatter and markdown body from SKILL.md."""
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
        """Return full instructions for the skill by name."""
        skill = self.registry.get(name)
        return skill.instructions if skill else None

    def available_skills_xml(self, skill_names: list[str]) -> str:
        """Generate <available_skills> XML for system prompt (metadata only)."""
        if not self.registry:
            return ""
        lines = ["<available_skills>"]
        for s in self.registry.values():
            if s.name in skill_names:
                lines.append("  <skill>")
                lines.append(f"    <name>{s.name}</name>")
                lines.append(f"    <description>{s.description}</description>")
                lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)


# Cache of PluginManager per plugin (plugin_name -> PluginManager)
plugin_managers: dict[str, PluginManager] = {}

def available_plugins_list():
    plugin_dir = PLUGINS_DIR
    if not os.path.isdir(plugin_dir):
        return []
    
    plugin_list = []
    for plugin in os.listdir(plugin_dir):
        plugin_list.append({"name": plugin})
        
    return plugin_list


def available_plugin_skills(plugin_name: str) -> list:
    """Return skill names that belong to the given plugin (from plugins/<name>/skills/)."""
    if plugin_name not in plugin_managers:
        skills_dir = os.path.join(PLUGINS_DIR, plugin_name, "skills")
        plugin_managers[plugin_name] = PluginManager(skills_dir)

    registry = plugin_managers[plugin_name].registry

    if not registry:
        return []

    return [{"name": s.name, "description": s.description} for s in registry.values()]

def is_command(query: str, plugin_name: str) -> bool:
    """Check if the query is a command."""

    logger.info(f"query: {query} - plugin_name: {plugin_name}")

    if plugin_name == "base":
        return False

    if not query.startswith("/"):
        return False

    command = query.split(" ")[0]
    command_name = command.lstrip("/").lower()  # "/search" -> "search"
    logger.info(f"command: {command} - not confiremed")

    commands_dir = os.path.join(PLUGINS_DIR, plugin_name, "commands")
    if not os.path.isdir(commands_dir):
        logger.warning(f"Commands directory not found: {commands_dir}")
        return False

    commands = os.listdir(commands_dir)
    if command_name + ".md" not in commands:
        logger.warning(f"Command not found: {command}")
        return False
    else:
        logger.info(f"command: {command} - confirmed")
        return True
    
def create_plugin_and_get_skill_instructions(plugin_name: str):
    """Create get_skill_instructions tool that uses the plugin's PluginManager.

    The builtin get_skill_instructions from langgraph_agent uses the global SkillManager
    (application/skills/), which does not include plugin skills (plugins/<name>/skills/).
    This tool uses the plugin's PluginManager so frontend-design and other plugin skills
    are found correctly.
    """
    plugin_manager = plugin_managers.get(plugin_name)
    if plugin_name not in plugin_managers:
        skills_dir = os.path.join(PLUGINS_DIR, plugin_name, "skills")
        plugin_managers[plugin_name] = PluginManager(skills_dir)
        plugin_manager = plugin_managers[plugin_name]

    @tool
    def get_skill_instructions(skill_name: str) -> str:
        """Load the full instructions for a specific skill by name.

        Use this when you need detailed instructions for a task that matches
        one of the available skills listed in the system prompt.

        Args:
            skill_name: The name of the skill to load (e.g. 'frontend-design').

        Returns:
            The full skill instructions, or an error message if not found.
        """
        logger.info(f"###### get_skill_instructions (plugin={plugin_name}): {skill_name} ######")
        instructions = plugin_manager.get_skill_instructions(skill_name)
        if instructions:
            return instructions

        # fallback to base skills
        if plugin_name != "base":
            base_manager = plugin_managers.get("base")
            if base_manager is None:
                skills_dir = os.path.join(WORKING_DIR, "skills")
                base_manager = PluginManager(skills_dir)
                plugin_managers["base"] = base_manager
            instructions = base_manager.get_skill_instructions(skill_name)
            if instructions:
                return instructions

        available = ", ".join(plugin_manager.registry.keys())
        return f"Skill '{skill_name}' not found. Available skills: {available}"

    return get_skill_instructions


def load_plugin_mcp_config_from_json(plugin_path: str) -> dict:
    """Load MCP config from plugin's .mcp.json file.
    
    Args:
        plugin_path: Absolute path to plugin directory (e.g. application/plugins/enterprise-search)
    
    Returns:
        MCP config dict with mcpServers key, or empty dict if file not found.
    """
    import json
    mcp_path = os.path.join(plugin_path, ".mcp.json")
    if not os.path.isfile(mcp_path):
        logger.warning(f"Plugin MCP config not found: {mcp_path}")
        return {"mcpServers": {}}
    try:
        with open(mcp_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config if isinstance(config.get("mcpServers"), dict) else {"mcpServers": {}}
    except Exception as e:
        logger.error(f"Failed to load plugin MCP config from {mcp_path}: {e}")
        return {"mcpServers": {}}


def load_plugin_mcp_servers_from_list(plugin_path: str) -> list:
    """Load MCP server names from plugin's mcp_servers.list file.

    Args:
        plugin_path: Absolute path to plugin directory (e.g. application/plugins/enterprise-search)

    Returns:
        List of MCP server names, or empty list if file not found.
    """
    list_path = os.path.join(plugin_path, "mcp_servers.list")
    if not os.path.isfile(list_path):
        logger.info(f"mcp_servers.list not found: {list_path}")
        return []
    try:
        with open(list_path, "r", encoding="utf-8") as f:
            servers = json.load(f)
        return servers if isinstance(servers, list) else []
    except Exception as e:
        logger.error(f"Failed to load mcp_servers.list from {list_path}: {e}")
        return []


def get_builtin_tools():
    """Return the list of built-in tools for the skill-aware agent."""
    return [skill.execute_code, skill.write_file, skill.read_file, skill.upload_file_to_s3, skill.get_skill_instructions]
