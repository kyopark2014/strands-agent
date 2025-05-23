import os
import logging
import pandas as pd
from mcp.server.fastmcp import FastMCP
from vanna.remote import Vanna  # Or vanna.base if building a custom Vanna class
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.openai.openai_chat import OpenAI_Chat

# Setup Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Vanna Instance
vn = None

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        if config is None:
            config = {}
        ChromaDB_VectorStore.__init__(self, config=config.get('chromadb'))
        OpenAI_Chat.__init__(self, config=config.get('openai'))

def setup_vanna():
    global vn
    api_key = os.environ.get('VANNA_OPENAI_API_KEY')
    chroma_path = os.environ.get('VANNA_CHROMA_PATH', './vanna_chroma_db')

    if not api_key:
        logger.warning("VANNA_OPENAI_API_KEY environment variable not set. Vanna OpenAI features will be unavailable.")
        # Decide if you want to proceed without OpenAI or raise an error
        # For now, we'll allow it to proceed, but OpenAI_Chat might fail later if used.

    config = {
        'openai': {'api_key': api_key},
        'chromadb': {'path': chroma_path}
    }

    try:
        vn = MyVanna(config=config)
        logger.info("Vanna instance setup complete.")
        # Attempt to connect to ChromaDB to ensure the path is valid
        # This is a basic check; actual data interaction methods will be separate
        vn.client # Accessing client to trigger connection/init
        logger.info(f"ChromaDB initialized at path: {chroma_path}")
    except Exception as e:
        logger.error(f"Failed to initialize Vanna or ChromaDB: {e}")
        vn = None # Ensure vn is None if setup fails

# MCP Instance
mcp = FastMCP("VannaAgent", dependencies=["vanna", "chromadb", "openai"])

# Initial Call to setup_vanna()
setup_vanna()

# Tools will be defined here
@mcp.tool()
def vanna_connect_db(db_type: str, db_name: str, host: str = None, user: str = None, password: str = None, port: int = None, ddl: str = None) -> str:
    """
    Connects to the specified database and optionally trains on DDL.
    db_type: Type of the database ('sqlite', 'postgres', 'mysql').
    db_name: Database name (for SQLite, this is the file path).
    host: Hostname for Postgres/MySQL.
    user: Username for Postgres/MySQL.
    password: Password for Postgres/MySQL.
    port: Port number for Postgres/MySQL.
    ddl: Optional DDL string to train Vanna on immediately after connection.
    """
    global vn
    if vn is None:
        logger.error("Vanna instance (vn) is not initialized.")
        return "Error: Vanna instance not initialized. Please check server logs."

    try:
        if db_type.lower() == 'sqlite':
            vn.connect_to_sqlite(db_name)
            logger.info(f"Connected to SQLite database: {db_name}")
            message = f"Successfully connected to SQLite database: {db_name}"
        elif db_type.lower() == 'postgres':
            if not all([host, user, password, port]):
                return "Error: For PostgreSQL, host, user, password, and port are required."
            vn.connect_to_postgres(host=host, dbname=db_name, user=user, password=password, port=port)
            logger.info(f"Connected to PostgreSQL database: {db_name} on {host}")
            message = f"Successfully connected to PostgreSQL database: {db_name} on {host}"
        elif db_type.lower() == 'mysql':
            if not all([host, user, password, port]):
                return "Error: For MySQL, host, user, password, and port are required."
            vn.connect_to_mysql(host=host, db=db_name, user=user, password=password, port=port)
            logger.info(f"Connected to MySQL database: {db_name} on {host}")
            message = f"Successfully connected to MySQL database: {db_name} on {host}"
        else:
            return f"Error: Unsupported database type '{db_type}'. Supported types are 'sqlite', 'postgres', 'mysql'."

        if ddl:
            vn.train(ddl=ddl)
            logger.info("Trained Vanna on provided DDL.")
            message += " and trained on provided DDL."
        return message
    except Exception as e:
        logger.error(f"Error connecting to database or training DDL for {db_type}: {e}")
        return f"Error connecting to {db_type} or training DDL: {str(e)}"

@mcp.tool()
def vanna_train_ddl(ddl_statements: str) -> str:
    """Trains Vanna using DDL statements."""
    global vn
    if vn is None:
        return "Error: Vanna not initialized."
    try:
        vn.train(ddl=ddl_statements)
        logger.info("Vanna training with DDL completed.")
        return "Successfully trained Vanna with DDL statements."
    except Exception as e:
        logger.error(f"Error training Vanna with DDL: {e}")
        return f"Error training Vanna with DDL: {str(e)}"

@mcp.tool()
def vanna_train_documentation(documentation: str) -> str:
    """Trains Vanna using documentation."""
    global vn
    if vn is None:
        return "Error: Vanna not initialized."
    try:
        vn.train(documentation=documentation)
        logger.info("Vanna training with documentation completed.")
        return "Successfully trained Vanna with documentation."
    except Exception as e:
        logger.error(f"Error training Vanna with documentation: {e}")
        return f"Error training Vanna with documentation: {str(e)}"

@mcp.tool()
def vanna_train_sql(sql_query: str) -> str:
    """Trains Vanna using SQL queries."""
    global vn
    if vn is None:
        return "Error: Vanna not initialized."
    try:
        vn.train(sql=sql_query)
        logger.info("Vanna training with SQL query completed.")
        return "Successfully trained Vanna with SQL query."
    except Exception as e:
        logger.error(f"Error training Vanna with SQL: {e}")
        return f"Error training Vanna with SQL: {str(e)}"

@mcp.tool()
def vanna_ask_sql(natural_language_question: str) -> str:
    """Generates SQL from a natural language question."""
    global vn
    if vn is None:
        return "Error: Vanna not initialized."
    try:
        sql = vn.generate_sql(question=natural_language_question)
        logger.info(f"Generated SQL for question '{natural_language_question}': {sql}")
        return sql if sql else "No SQL generated."
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        return f"Error generating SQL: {str(e)}"

@mcp.tool()
def vanna_run_sql(sql_query: str) -> str:
    """Runs SQL query and returns results as JSON."""
    global vn
    if vn is None:
        return "Error: Vanna not initialized."
    try:
        df = vn.run_sql(sql=sql_query)
        if df is None:
            logger.info(f"SQL query '{sql_query}' executed but returned no data (None DataFrame).")
            return "[]" # Return empty JSON array for None DataFrame
        logger.info(f"SQL query '{sql_query}' executed, returning {len(df)} rows.")
        return df.to_json(orient='records')
    except Exception as e:
        logger.error(f"Error running SQL: {e}")
        return f"Error running SQL: {str(e)}"

@mcp.tool()
def vanna_generate_plotly_code(natural_language_question: str, sql_query: str, df_json_string: str) -> str:
    """Generates Plotly code for a question, SQL, and DataFrame JSON."""
    global vn
    if vn is None:
        return "Error: Vanna not initialized."
    try:
        df = pd.read_json(df_json_string, orient='records')
        plotly_code = vn.generate_plotly_code(question=natural_language_question, sql=sql_query, df=df)
        logger.info(f"Generated Plotly code for question '{natural_language_question}'.")
        return plotly_code if plotly_code else "No Plotly code generated."
    except Exception as e:
        logger.error(f"Error generating Plotly code: {e}")
        return f"Error generating Plotly code: {str(e)}"

@mcp.tool()
def vanna_get_training_data() -> list:
    """Retrieves Vanna's training data."""
    global vn
    if vn is None:
        logger.error("Vanna instance (vn) is not initialized.")
        return ["Error: Vanna instance not initialized. Please check server logs."]
    try:
        training_data = vn.get_training_data()
        logger.info(f"Retrieved {len(training_data)} training data items.")
        # The training data items might not be directly JSON serializable if they are custom objects.
        # For now, assuming they are, or that FastMCP handles it.
        # If they are complex objects, they might need conversion to dict/JSON serializable format.
        return training_data
    except Exception as e:
        logger.error(f"Error retrieving training data: {e}")
        return [f"Error retrieving training data: {str(e)}"]

@mcp.tool()
def vanna_remove_training_data(data_id: str) -> bool:
    """Removes a specific piece of training data by ID."""
    global vn
    if vn is None:
        logger.error("Vanna instance (vn) is not initialized.")
        return False
    try:
        success = vn.remove_training_data(id=data_id)
        if success:
            logger.info(f"Successfully removed training data with ID: {data_id}")
        else:
            logger.warning(f"Failed to remove training data with ID: {data_id} (or data_id not found).")
        return success
    except Exception as e:
        logger.error(f"Error removing training data with ID {data_id}: {e}")
        return False

if __name__ == "__main__":
    if vn is None:
        logger.error("Vanna instance (vn) is not initialized. Cannot start MCP server.")
        logger.error("Please check your VANNA_OPENAI_API_KEY and VANNA_CHROMA_PATH environment variables.")
    else:
        logger.info("Starting MCP server for VannaAgent...")
        mcp.run()
