import logging
import sys
import mcp_basic

from mcp.server.fastmcp import FastMCP 

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("aws-server-basic")

try:
    mcp = FastMCP(
        name="basic",
        instructions=(
            "You are a helpful assistant. "
            "You can use tools for the user's question and provide the answer."
        ),
    )
    logger.info("MCP server initialized successfully")
except Exception as e:
        err_msg = f"Error: {str(e)}"
        logger.info(f"{err_msg}")

######################################
# Time
######################################
@mcp.tool()
def get_current_time(format: str=f"%Y-%m-%d %H:%M:%S")->str:
    """Returns the current date and time in the specified format"""
    logger.info(f"get_current_time --> format: {format}")

    return mcp_basic.get_current_time(format)

######################################
# Book
######################################    
@mcp.tool()  
def get_book_list(keyword: str) -> str:
    """
    Search book list by keyword and then return book list
    keyword: search keyword
    return: book list
    """
    logger.info(f"get_book_list --> keyword: {keyword}")

    return mcp_basic.get_book_list(keyword)

######################################
# Weather
######################################
@mcp.tool()    
def get_weather_info(city: str) -> str:
    """
    retrieve weather information by city name and then return weather statement.
    city: the English name of city to retrieve
    return: weather statement
    """   
    logger.info(f"get_weather_info --> city: {city}")
    
    return mcp_basic.get_weather_info(city)

if __name__ =="__main__":
    mcp.run(transport="stdio")


