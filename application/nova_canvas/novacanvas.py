# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
# with the License. A copy of the License is located at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# or in the 'license' file accompanying this file. This file is distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions
# and limitations under the License.
"""Amazon Nova Canvas API interaction module.

This module provides functions for generating images using Amazon Nova Canvas
through the AWS Bedrock service. It handles the API requests, response processing,
and image saving functionality.
"""

import base64
import json
import os
import random
import boto3 
import logging
import sys
import chat
from botocore.config import Config

from .models import (
    ColorGuidedGenerationParams,
    ColorGuidedRequest,
    ImageGenerationConfig,
    ImageGenerationResponse,
    Quality,
    TextImageRequest,
    TextToImageParams,
)
from nova_canvas.consts import (
    DEFAULT_CFG_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NUMBER_OF_IMAGES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUALITY,
    DEFAULT_WIDTH,
    NOVA_CANVAS_MODEL_ID,
)

from typing import TYPE_CHECKING, Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-log")

def load_image_generator_config():
    config = None
    try:
        with open("image_generator_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            logger.info(f"image_generator_config: {config}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")    
    return config

import requests
from PIL import Image
import io
import base64
import traceback

def resize_image(image_data, min_size=320, max_size=4096):
    img = Image.open(io.BytesIO(image_data))    
    width, height = img.size
    
    # Check if resizing is necessary
    if width < min_size or height < min_size or width > max_size or height > max_size:
        if width < min_size or height < min_size:            
            scale = min_size / min(width, height) # Need to upscale
        else:            
            scale = max_size / max(width, height) # Need to downscale
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
        
        buffer = io.BytesIO()
        img.save(buffer, format=img.format if img.format else 'PNG')
        return buffer.getvalue()
    
    return image_data

def get_seed_image():
    config = load_image_generator_config()
    url_seed_image = config["seed_image"]
    logger.info(f"url_seed_image: {url_seed_image}")

    # for debugging
    # url = "https://d2ktwrbyxtrufc.cloudfront.net/images/seed_image.png"
    # url_seed_image = url
    # logger.info(f"url_seed_image: {url_seed_image}")

    if url_seed_image:        
        response = requests.get(url_seed_image)
        logger.info(f"response: {response}")
        
        if response.status_code == 200:
            logger.info(f"success to load: {url_seed_image}")
            # Perform image resizing
            resized_image = resize_image(response.content, min_size=320, max_size=2048)
            return base64.b64encode(resized_image).decode('utf8')
        else:
            logger.error(f"Failed to load image: {url_seed_image}")
            raise Exception(f"Image download failed: {response.status_code}")
    else:
        logger.info(f"no seed image")
        return None

# aws_region = os.environ.get('AWS_REGION', 'us-west-2')
aws_region = "us-east-1"
import boto3
try:
    if aws_profile := os.environ.get('AWS_PROFILE'):
        bedrock_runtime_client = boto3.Session(
            profile_name=aws_profile, region_name=aws_region
        ).client('bedrock-runtime')
    else:
        bedrock_runtime_client = boto3.Session(region_name=aws_region).client('bedrock-runtime')
except Exception as e:
    logger.error(f'Error creating bedrock runtime client: {str(e)}')
    raise

def save_generated_images(
    base64_images: List[str],
    filename: Optional[str] = None,
    number_of_images: int = DEFAULT_NUMBER_OF_IMAGES,
    prefix: str = 'nova_canvas'
) -> List[str]:
    """Save base64-encoded images to files.

    Args:
        base64_images: List of base64-encoded image data.
        filename: Base filename to use (without extension). If None, a random name is generated.
        number_of_images: Number of images being saved.
        prefix: Prefix to use for randomly generated filenames.

    Returns:
        Dictionary with lists of paths to the saved image files and PIL Image objects.
    """
    logger.debug(f'Saving {len(base64_images)} images')
    # Determine the output directory

    # Save the generated images
    saved_paths: List[str] = []
    for i, base64_image_data in enumerate(base64_images):
        # Generate filename if not provided
        if filename:
            image_filename = (
                f'{filename}_{i + 1}.png' if number_of_images > 1 else f'{filename}.png'
            )
        else:
            # Generate a random filename
            random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
            image_filename = f'{prefix}_{random_id}_{i + 1}.png'

        # Decode the base64 image data
        image_data = base64.b64decode(base64_image_data)
    
        url = chat.upload_to_s3(image_data, image_filename)
        logger.info(f"Uploaded image to S3: {url}")

        saved_paths.append(url)

    return saved_paths

async def invoke_nova_canvas(
    request_model_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Invoke the Nova Canvas API with the given request.

    Args:
        request_model_dict: Dictionary representation of the request model.

    Returns:
        Dictionary containing the API response.

    Raises:
        Exception: If the API call fails.
    """
    logger.debug('Invoking Nova Canvas API')

    # Convert the request payload to JSON
    request = json.dumps(request_model_dict)
    # logger.info(f"request_model_dict: {request_model_dict}")

    try:
        # Invoke the model
        logger.info(f'Sending request to Nova Canvas model: {NOVA_CANVAS_MODEL_ID}')
        response = bedrock_runtime_client.invoke_model(modelId=NOVA_CANVAS_MODEL_ID, body=request)

        # Decode the response body
        result = json.loads(response['body'].read().decode('utf-8'))
        logger.info('Nova Canvas API call successful')
        return result
    except Exception as e:
        logger.error(f'Nova Canvas API call failed: {str(e)}')
        raise

async def generate_image_with_text(
    prompt: str,
    negative_prompt: Optional[str] = None,
    filename: Optional[str] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    quality: str = DEFAULT_QUALITY,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    seed: Optional[int] = None,
    number_of_images: int = DEFAULT_NUMBER_OF_IMAGES
) -> ImageGenerationResponse:
    """Generate an image using Amazon Nova Canvas with text prompt.

    This function uses Amazon Nova Canvas to generate images based on a text prompt.
    The generated image will be saved to a file and the path will be returned.

    Args:
        prompt: The text description of the image to generate (1-1024 characters).
        negative_prompt: Text to define what not to include in the image (1-1024 characters).
        filename: The name of the file to save the image to (without extension).
            If not provided, a random name will be generated.
        width: The width of the generated image (320-4096, divisible by 16).
        height: The height of the generated image (320-4096, divisible by 16).
        quality: The quality of the generated image ("standard" or "premium").
        cfg_scale: How strongly the image adheres to the prompt (1.1-10.0).
        seed: Seed for generation (0-858,993,459). Random if not provided.
        number_of_images: The number of images to generate (1-5).        
    Returns:
        ImageGenerationResponse: An object containing the paths to the generated images,
        PIL Image objects, and status information.
    """
    logger.debug(f"Generating text-to-image with prompt: '{prompt[:30]}...' ({width}x{height})")

    try:
        # Validate input parameters using Pydantic
        try:
            logger.debug('Validating parameters and creating request model')

            # Create image generation config
            config = ImageGenerationConfig(
                width=width,
                height=height,
                quality=Quality.STANDARD if quality == DEFAULT_QUALITY else Quality.PREMIUM,
                cfgScale=cfg_scale,
                seed=seed if seed is not None else random.randint(0, 858993459),
                numberOfImages=number_of_images,
            )

            logger.info("loading seed image")
            seed_image = get_seed_image()

            # Create text-to-image params
            if seed_image is not None:
                if negative_prompt is not None:
                    text_params = TextToImageParams(text=prompt, negativeText=negative_prompt, conditionImage=seed_image)
                else:
                    text_params = TextToImageParams(text=prompt, conditionImage=seed_image)
            else:
                if negative_prompt is not None:
                    text_params = TextToImageParams(text=prompt, negativeText=negative_prompt)
                else:
                    text_params = TextToImageParams(text=prompt)
            # logger.info(f"text_params: {text_params}")

            # Create the full request
            request_model = TextImageRequest(
                textToImageParams=text_params, imageGenerationConfig=config
            )

            # Convert model to dictionary
            request_model_dict = request_model.to_api_dict()                        
            # logger.info(f"request_model_dict of generate_image_with_text: {request_model_dict}")

        except Exception as e:
            logger.error(f'Parameter validation failed: {str(e)}')
            return ImageGenerationResponse(
                status='error',
                message=f'Validation error: {str(e)}',
                paths=[],
                prompt=prompt,
                negative_prompt=negative_prompt,
            )

        try:
            # Invoke the Nova Canvas API
            logger.debug('Sending request to Nova Canvas API')
            model_response = await invoke_nova_canvas(request_model_dict)

            # Extract the image data
            base64_images = model_response['images']
            logger.info(f'Received {len(base64_images)} images from Nova Canvas API')

            # Save the generated images
            paths = save_generated_images(
                base64_images,
                filename,
                number_of_images,
                prefix='nova_canvas'
            )
            logger.info(f"paths: {paths}")

            if len(paths) > 0:
                url_message = f'Generated image url: '
                url_message += ', '.join(paths)
            else:
                url_message = f'No image generated'

            return ImageGenerationResponse(
                status='success',
                message=url_message,
                paths=paths,
                prompt=prompt,
                negative_prompt=negative_prompt,
            )
        except Exception as e:
            logger.error(f'Image generation failed: {str(e)}')
            return ImageGenerationResponse(
                status='error',
                message=str(e),
                paths=[],
                prompt=prompt,
                negative_prompt=negative_prompt,
            )

    except Exception as e:
        logger.error(f'Unexpected error in generate_image_with_text: {str(e)}')
        return ImageGenerationResponse(
            status='error',
            message=str(e),
            paths=[],
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

async def generate_image_with_colors(
    prompt: str,
    colors: List[str],
    negative_prompt: Optional[str] = None,
    filename: Optional[str] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    quality: str = DEFAULT_QUALITY,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    seed: Optional[int] = None,
    number_of_images: int = DEFAULT_NUMBER_OF_IMAGES    
) -> ImageGenerationResponse:
    """Generate an image using Amazon Nova Canvas with color guidance.

    This function uses Amazon Nova Canvas to generate images based on a text prompt and color palette.
    The generated image will be saved to a file and the path will be returned.

    Args:
        prompt: The text description of the image to generate (1-1024 characters).
        colors: List of up to 10 hexadecimal color values (e.g., "#FF9800").
        negative_prompt: Text to define what not to include in the image (1-1024 characters).
        filename: The name of the file to save the image to (without extension).
            If not provided, a random name will be generated.
        width: The width of the generated image (320-4096, divisible by 16).
        height: The height of the generated image (320-4096, divisible by 16).
        quality: The quality of the generated image ("standard" or "premium").
        cfg_scale: How strongly the image adheres to the prompt (1.1-10.0).
        seed: Seed for generation (0-858,993,459). Random if not provided.
        number_of_images: The number of images to generate (1-5).
    Returns:
        ImageGenerationResponse: An object containing the paths to the generated images,
        PIL Image objects, and status information.
    """
    logger.debug(
        f"Generating color-guided image with prompt: '{prompt[:30]}...' and {len(colors)} colors"
    )

    try:
        # Validate input parameters using Pydantic
        try:
            logger.debug('Validating parameters and creating color-guided request model')

            # Create image generation config
            config = ImageGenerationConfig(
                width=width,
                height=height,
                quality=Quality.STANDARD if quality == DEFAULT_QUALITY else Quality.PREMIUM,
                cfgScale=cfg_scale,
                seed=seed if seed is not None else random.randint(0, 858993459),
                numberOfImages=number_of_images,
            )

            logger.info("loading seed image")
            seed_image = get_seed_image()

            # Create color-guided params
            # The Nova Canvas API doesn't accept null for negativeText
            if seed_image is not None:
                if negative_prompt is not None:
                    color_params = ColorGuidedGenerationParams(
                        colors=colors,
                        text=prompt,
                        negativeText=negative_prompt,
                        referenceImage=seed_image
                    )
                else:
                    color_params = ColorGuidedGenerationParams(
                        colors=colors,
                        text=prompt,
                        referenceImage=seed_image
                    )
            else:
                if negative_prompt is not None:
                    color_params = ColorGuidedGenerationParams(
                        colors=colors,
                        text=prompt,
                        negativeText=negative_prompt,
                    )
                else:
                    color_params = ColorGuidedGenerationParams(
                        colors=colors,
                        text=prompt,
                    )

            # Create the full request
            request_model = ColorGuidedRequest(
                colorGuidedGenerationParams=color_params, imageGenerationConfig=config
            )

            # Convert model to dictionary
            request_model_dict = request_model.to_api_dict()
            #logger.info(f"request_model_dict of generate_image_with_colors: {request_model_dict}")

        except Exception as e:
            logger.error(f'Color-guided parameter validation failed: {str(e)}')
            return ImageGenerationResponse(
                status='error',
                message=f'Validation error: {str(e)}',
                paths=[],
                prompt=prompt,
                negative_prompt=negative_prompt,
                colors=colors,
            )

        try:
            # Invoke the Nova Canvas API
            logger.debug('Sending color-guided request to Nova Canvas API')
            model_response = await invoke_nova_canvas(request_model_dict)

            # Extract the image data
            base64_images = model_response['images']
            logger.info(f'Received {len(base64_images)} images from Nova Canvas API')

            # Save the generated images
            paths = save_generated_images(
                base64_images,
                filename,
                number_of_images,
                prefix='nova_canvas_color'
            )
            logger.info(f"paths: {paths}")

            if len(paths) > 0:
                url_message = f'Generated image url: '
                url_message += ', '.join(paths)
            else:
                url_message = f'No image generated'

            return ImageGenerationResponse(
                status='success',
                message=url_message,
                paths=paths,
                prompt=prompt,
                negative_prompt=negative_prompt,
                colors=colors,
            )
        except Exception as e:
            logger.error(f'Color-guided image generation failed: {str(e)}')
            return ImageGenerationResponse(
                status='error',
                message=str(e),
                paths=[],
                prompt=prompt,
                negative_prompt=negative_prompt,
                colors=colors,
            )

    except Exception as e:
        logger.error(f'Unexpected error in generate_image_with_colors: {str(e)}')
        return ImageGenerationResponse(
            status='error',
            message=str(e),
            paths=[],
            prompt=prompt,
            negative_prompt=negative_prompt,
            colors=colors,
        )