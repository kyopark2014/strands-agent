import re
import base64
import logging
import traceback
import chat 
import sys
import uuid

from urllib import parse
from langchain_experimental.tools import PythonAstREPLTool
from io import BytesIO

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-cost")

repl = PythonAstREPLTool()

def repl_coder(code):
    """
    Use this to execute python code and do math. 
    If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.
    code: the Python code was written in English
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    if result is None:
        result = "It didn't return anything."

    return result

def generate_short_uuid(length=8):
    full_uuid = uuid.uuid4().hex
    return full_uuid[:length]

def repl_drawer(code):
    """
    Execute a Python script for draw a graph.
    Since Python runtime cannot use external APIs, necessary data must be included in the code.
    The graph should use English exclusively for all textual elements.
    Do not save pictures locally bacause the runtime does not have filesystem.
    When a comparison is made, all arrays must be of the same length.
    code: the Python code was written in English
    return: the url of graph
    """ 
        
    code = re.sub(r"seaborn", "classic", code)
    code = re.sub(r"plt.savefig", "#plt.savefig", code)
    code = re.sub(r"plt.show", "#plt.show", code)

    post = """\n
import io
import base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.getvalue()).decode()

print(image_base64)
"""
    code = code + post    
    logger.info(f"code: {code}")
    
    image_url = ""
    try:     
        resp = repl.run(code)

        base64Img = resp
        
        if base64Img:
            byteImage = BytesIO(base64.b64decode(base64Img))

            image_name = generate_short_uuid()+'.png'
            url = chat.upload_to_s3(byteImage, image_name)
            logger.info(f"url: {url}")

            file_name = url[url.rfind('/')+1:]
            logger.info(f"file_name: {file_name}")

            image_url = chat.path+'/'+chat.s3_image_prefix+'/'+parse.quote(file_name)
            logger.info(f"image_url: {image_url}")

            # im = Image.open(BytesIO(base64.b64decode(base64Img)))  # for debuuing
            # im.save(image_name, 'PNG')

    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")

    return {
        "path": image_url
    }
