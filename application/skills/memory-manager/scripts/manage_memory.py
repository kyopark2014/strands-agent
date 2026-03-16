#!/usr/bin/env python3
"""
Memory file management utilities for agent

This script helps manage memory Markdown files:
- Create daily logs
- Append entries to daily logs or MEMORY.md
- List recent memory files
- Archive old daily logs
"""
import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path


def get_memory_dir(workspace=None):
    """Get memory directory path"""
    if workspace:
        return Path(workspace) / "memory"

    # Default: ~/.agent/workspace/memory
    home = Path.home()
    return home / ".agent" / "workspace" / "memory"


def get_memory_root(workspace=None):
    """Get workspace root (for MEMORY.md)"""
    if workspace:
        return Path(workspace)

    home = Path.home()
    return home / ".agent" / "workspace"


def ensure_memory_dir(workspace=None):
    """Ensure memory directory exists"""
    memory_dir = get_memory_dir(workspace)
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir


def get_daily_log_path(date=None, workspace=None):
    """Get path to daily log file"""
    if date is None:
        date = datetime.now()
    elif isinstance(date, str):
        date = datetime.fromisoformat(date)

    memory_dir = ensure_memory_dir(workspace)
    filename = date.strftime("%Y-%m-%d.md")
    return memory_dir / filename


def create_daily_log(date=None, workspace=None):
    """Create daily log file if it doesn't exist"""
    path = get_daily_log_path(date, workspace)

    if path.exists():
        return str(path)

    if date is None:
        date = datetime.now()
    elif isinstance(date, str):
        date = datetime.fromisoformat(date)

    day_name = date.strftime("%A")
    date_str = date.strftime("%Y-%m-%d")

    header = f"# {date_str} ({day_name})\n\n"
    path.write_text(header)

    return str(path)


def append_to_file(path, content, section=None):
    """Append content to a memory file"""
    path = Path(path)

    if not path.exists():
        if path.name == "MEMORY.md":
            path.write_text("# MEMORY.md\n\n")
        else:
            # Create daily log with header
            date_str = path.stem
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                day_name = date.strftime("%A")
                path.write_text(f"# {date_str} ({day_name})\n\n")
            except ValueError:
                path.write_text("")

    current = path.read_text()

    if section:
        # Add section header if specified
        if not current.strip().endswith("\n"):
            current += "\n"
        current += f"\n## {section}\n\n"

    if not current.endswith("\n"):
        current += "\n"

    if not content.endswith("\n"):
        content += "\n"

    path.write_text(current + content)
    return str(path)


def list_recent_logs(days=7, workspace=None):
    """List recent daily log files"""
    memory_dir = get_memory_dir(workspace)

    if not memory_dir.exists():
        return []

    logs = []
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        path = get_daily_log_path(date, workspace)
        if path.exists():
            stat = path.stat()
            logs.append({
                "date": date.strftime("%Y-%m-%d"),
                "path": str(path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

    return logs


def archive_old_logs(days=90, workspace=None):
    """Move old daily logs to archive subdirectory"""
    memory_dir = get_memory_dir(workspace)

    if not memory_dir.exists():
        return []

    archive_dir = memory_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    cutoff = datetime.now() - timedelta(days=days)
    archived = []

    for file in memory_dir.glob("????-??-??.md"):
        try:
            date = datetime.strptime(file.stem, "%Y-%m-%d")
            if date < cutoff:
                dest = archive_dir / file.name
                file.rename(dest)
                archived.append(str(dest))
        except ValueError:
            continue

    return archived


def main():
    parser = argparse.ArgumentParser(description="Memory file management utilities")
    parser.add_argument("--workspace", help="Workspace directory (default: ~/.agent/workspace)")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # create-daily command
    create_parser = subparsers.add_parser("create-daily", help="Create daily log file")
    create_parser.add_argument("--date", help="Date (YYYY-MM-DD, default: today)")

    # append command
    append_parser = subparsers.add_parser("append", help="Append content to memory file")
    append_parser.add_argument("file", help="File path (relative to workspace)")
    append_parser.add_argument("content", help="Content to append")
    append_parser.add_argument("--section", help="Section header")

    # list command
    list_parser = subparsers.add_parser("list", help="List recent daily logs")
    list_parser.add_argument("--days", type=int, default=7, help="Number of days (default: 7)")
    list_parser.add_argument("--json", action="store_true", help="Output JSON")

    # archive command
    archive_parser = subparsers.add_parser("archive", help="Archive old daily logs")
    archive_parser.add_argument("--days", type=int, default=90, help="Archive logs older than N days")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "create-daily":
        path = create_daily_log(args.date, args.workspace)
        print(f"Created: {path}")

    elif args.command == "append":
        workspace_root = get_memory_root(args.workspace)
        file_path = workspace_root / args.file
        result = append_to_file(file_path, args.content, args.section)
        print(f"Appended to: {result}")

    elif args.command == "list":
        logs = list_recent_logs(args.days, args.workspace)
        if args.json:
            print(json.dumps(logs, indent=2))
        else:
            for log in logs:
                print(f"{log['date']}: {log['path']} ({log['size']} bytes)")

    elif args.command == "archive":
        archived = archive_old_logs(args.days, args.workspace)
        print(f"Archived {len(archived)} files")
        for path in archived:
            print(f"  {path}")


if __name__ == "__main__":
    main()