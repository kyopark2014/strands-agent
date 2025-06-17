import logging
import sys
import mcp_log as log

from typing import Dict, Optional, Any
from mcp.server.fastmcp import FastMCP 

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("aws-log")

try:
    mcp = FastMCP(
        name = "AWSLogs",
        instructions=(
            "You are a helpful assistant. "
            "You can load logs of AWS CloudWatch and retrieve insights."
        ),
    )
    logger.info("MCP server initialized successfully")
except Exception as e:
        err_msg = f"Error: {str(e)}"
        logger.info(f"{err_msg}")

######################################
# AWS Logs
######################################
@mcp.tool()
async def list_groups(
    prefix: Optional[str] = None,
    region: Optional[str] = 'us-west-2'
) -> str:
    """List available CloudWatch log groups."""
    logger.info(f"list_groups --> prefix: {prefix}, region: {region}")

    return await log.list_groups(prefix=prefix, region=region)

@mcp.tool()
async def get_logs(
    logGroupName: str,
    logStreamName: Optional[str] = None,
    startTime: Optional[str] = None,
    endTime: Optional[str] = None,
    filterPattern: Optional[str] = None,
    region: Optional[str] = 'us-west-2'
) -> str:
    """Get CloudWatch logs from a specific log group and stream."""
    logger.info(f"get_logs --> logGroupName: {logGroupName}, logStreamName: {logStreamName}, startTime: {startTime}, endTime: {endTime}, filterPattern: {filterPattern}, region: {region}")

    return await log.get_logs(
        logGroupName=logGroupName,
        logStreamName=logStreamName,
        startTime=startTime,
        endTime=endTime,
        filterPattern=filterPattern,
        region=region
    )
    
######################################
# AWS Logs
######################################

if __name__ =="__main__":
    print(f"###### main ######")
    mcp.run(transport="stdio")


