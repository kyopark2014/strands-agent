# Connectors

## How tool references work

Plugin files use `~~category` as a placeholder for whatever tool the user connects in that category. For example, `~~chat` might mean Slack, Microsoft Teams, or any other chat tool with an MCP server.

Plugins are **tool-agnostic** — they describe workflows in terms of categories (chat, email, cloud storage, etc.) rather than specific products. The mcp_servers.list avaalable specific MCP servers, but any MCP server in that category works.

This plugin uses `~~category` references extensively as source labels in search output (e.g. `~~chat:`, `~~email:`). These are intentional — they represent dynamic category markers that resolve to whatever tool is connected.

## Connectors for this plugin

| Category | Placeholder | Included servers | Other options |
|----------|-------------|-----------------|---------------|
| Chat | `~~chat` | Slack | - |
| Email | `~~email` | Gog | — |
| Cloud storage | `~~cloud storage` | - | - |
| Knowledge base | `~~knowledge base` | Notion | Confluence, Slite |
| Project tracker | `~~project tracker` | Atlassian (Jira/Confluence) | - |
| CRM | `~~CRM` | *(not pre-configured)* | Salesforce |
| Office suite | `~~office suite` | Google Workspace | - |
