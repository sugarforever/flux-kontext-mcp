import os
import sys
import argparse
import logging
import time
from typing import Any, Dict, List, Optional
import replicate
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flux_kontext_mcp.log')
    ]
)
logger = logging.getLogger(__name__)

def get_api_token() -> str:
    """Get the Replicate API token from environment variables"""
    logger.info("Retrieving Replicate API token from environment")
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        logger.error("REPLICATE_API_TOKEN environment variable not found")
        raise ValueError("REPLICATE_API_TOKEN environment variable is required")
    logger.info("Replicate API token successfully retrieved")
    return api_token

# Set the API token for replicate
try:
    os.environ["REPLICATE_API_TOKEN"] = get_api_token()
    logger.info("Replicate API token set successfully")
except Exception as e:
    logger.error(f"Failed to set Replicate API token: {str(e)}")
    raise

mcp = FastMCP("flux-kontext-pro")
logger.info("FastMCP server initialized with name: flux-kontext-pro")

def rewrite_kontext_prompt(prompt: str) -> str:
    """
    Rewrite the prompt to be more specific and detailed using OpenAI API.
    Falls back to original prompt if API is unavailable or fails.
    """
    logger.info(f"Starting prompt rewrite for prompt: {prompt[:100]}...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found, using original prompt")
        return prompt
    
    try:
        # Get OpenAI configuration from environment variables
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        
        logger.info(f"Using OpenAI API with base_url: {base_url}, model: {model_name}")
        
        # Initialize OpenAI client
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        logger.debug("OpenAI client initialized successfully")
        
        # System prompt to instruct the model
        system_prompt = """You are a helpful assistant that translates and polishes prompts for image editing, to English. Your task is to:

1. If the input prompt is not in English, translate it to English
2. Keep the original intention and meaning intact
3. Do not add creative elements that weren't in the original prompt
4. Return only the improved prompt, no explanations or additional text

Be conservative and faithful to the original intent."""
        
        logger.info("Making OpenAI API call for prompt rewriting")
        start_time = time.time()
        
        # Make the API call with temperature 0 for conservative output
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        api_duration = time.time() - start_time
        logger.info(f"OpenAI API call completed in {api_duration:.2f} seconds")
        logger.info(f"OpenAI API response: {response}")
        # Extract the improved prompt
        improved_prompt = response.choices[0].message.content.strip()
        
        if improved_prompt:
            logger.info(f"Prompt successfully rewritten: {improved_prompt[:100]}...")
            return improved_prompt
        else:
            logger.warning("OpenAI API returned empty response, using original prompt")
            return prompt
        
    except Exception as e:
        # Return original prompt if any error occurs
        logger.error(f"OpenAI API call failed: {str(e)}, falling back to original prompt")
        return prompt

@mcp.tool()
def flux_kontext_generate(
    prompt: str,
    input_image: str,
    aspect_ratio: str = "match_input_image",
    output_format: str = "png",
    safety_tolerance: int = 2
) -> Dict[str, Any]:
    """
    Generate images using Flux Kontext Pro model from Replicate.
    
    Args:
        prompt (str): Text prompt describing the desired image or modifications
        input_image (str): URL of the input image to modify. If not provided, generates from scratch
        aspect_ratio (str): Aspect ratio for the output image. Options: "match_input_image", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"
        output_format (str): Output image format. Options: "png", "jpg", "webp"
        safety_tolerance (int): Safety tolerance level (0-5). Higher values are more permissive
        
    Returns:
        Dict[str, Any]: Generated image URL and metadata or error information
    """
    logger.info(f"Starting flux_kontext_generate with prompt: {prompt[:100]}...")
    logger.info(f"Parameters - aspect_ratio: {aspect_ratio}, output_format: {output_format}, safety_tolerance: {safety_tolerance}")
    logger.info(f"Input image provided: {bool(input_image)}")
    
    try:
        # Rewrite the prompt for better results
        rewritten_prompt = rewrite_kontext_prompt(prompt)
        
        # Prepare the input parameters
        input_params = {
            "prompt": rewritten_prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "safety_tolerance": safety_tolerance
        }
        
        # Add input image if provided
        if input_image:
            input_params["input_image"] = input_image
            logger.info(f"Added input image to parameters: {input_image[:100]}...")
        
        logger.info("Starting Replicate model execution")
        start_time = time.time()
        
        # Run the model
        output = replicate.run(
            "black-forest-labs/flux-kontext-pro",
            input=input_params
        )
        
        execution_duration = time.time() - start_time
        logger.info(f"Replicate model execution completed in {execution_duration:.2f} seconds")
        logger.info(f"Generated image URL: {output}")
        
        result = {
            "status": "success",
            "image_url": output,
            "model": "black-forest-labs/flux-kontext-pro",
            "input": input_params,
            "execution_time": execution_duration
        }
        
        logger.info("flux_kontext_generate completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"flux_kontext_generate failed: {str(e)}")
        return {"error": f"Image generation failed: {str(e)}"}

@mcp.tool()
def flux_kontext_async_generate(
    prompt: str,
    input_image: str,
    aspect_ratio: str = "match_input_image",
    output_format: str = "png",
    safety_tolerance: int = 2
) -> Dict[str, Any]:
    """
    Start an asynchronous image generation job using Flux Kontext Pro model.
    Use this for non-blocking requests when you don't need to wait for completion.
    
    Args:
        prompt (str): Text prompt describing the desired image or modifications
        input_image (Optional[str]): URL of the input image to modify
        aspect_ratio (str): Aspect ratio for the output image
        output_format (str): Output image format
        safety_tolerance (int): Safety tolerance level (0-5)
        
    Returns:
        Dict[str, Any]: Prediction object with ID and initial status
    """
    logger.info(f"Starting flux_kontext_async_generate with prompt: {prompt[:100]}...")
    logger.info(f"Parameters - aspect_ratio: {aspect_ratio}, output_format: {output_format}, safety_tolerance: {safety_tolerance}")
    logger.info(f"Input image provided: {bool(input_image)}")
    
    try:
        # Rewrite the prompt for better results
        rewritten_prompt = rewrite_kontext_prompt(prompt)
        
        # Prepare the input parameters
        input_params = {
            "prompt": rewritten_prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "safety_tolerance": safety_tolerance
        }
        
        # Add input image if provided
        if input_image:
            input_params["input_image"] = input_image
            logger.info(f"Added input image to parameters: {input_image[:100]}...")
        
        logger.info("Creating async prediction on Replicate")
        
        # Create a prediction without waiting
        prediction = replicate.predictions.create(
            model="black-forest-labs/flux-kontext-pro",
            input=input_params
        )
        
        logger.info(f"Async prediction created with ID: {prediction.id}")
        logger.info(f"Initial status: {prediction.status}")
        
        result = {
            "prediction_id": prediction.id,
            "status": prediction.status,
            "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
            "input": input_params,
            "urls": {
                "get": prediction.urls.get if hasattr(prediction, 'urls') else None,
                "cancel": prediction.urls.cancel if hasattr(prediction, 'urls') else None
            }
        }
        
        logger.info("flux_kontext_async_generate completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"flux_kontext_async_generate failed: {str(e)}")
        return {"error": f"Async generation failed: {str(e)}"}

@mcp.tool()
def flux_kontext_get_prediction(prediction_id: str) -> Dict[str, Any]:
    """
    Get the status and result of a specific prediction by ID.
    
    Args:
        prediction_id (str): The ID of the prediction to check
        
    Returns:
        Dict[str, Any]: Prediction status, result, and metadata
    """
    logger.info(f"Getting prediction status for ID: {prediction_id}")
    
    try:
        prediction = replicate.predictions.get(prediction_id)
        
        logger.info(f"Prediction {prediction_id} status: {prediction.status}")
        
        if prediction.status == "succeeded":
            logger.info(f"Prediction {prediction_id} completed successfully")
        elif prediction.status == "failed":
            logger.error(f"Prediction {prediction_id} failed: {prediction.error}")
        elif prediction.status == "canceled":
            logger.info(f"Prediction {prediction_id} was canceled")
        
        result = {
            "id": prediction.id,
            "status": prediction.status,
            "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
            "started_at": prediction.started_at.isoformat() if prediction.started_at else None,
            "completed_at": prediction.completed_at.isoformat() if prediction.completed_at else None,
            "output": prediction.output,
            "error": prediction.error,
            "input": prediction.input,
            "model": prediction.model,
            "logs": prediction.logs
        }
        
        logger.info("flux_kontext_get_prediction completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"flux_kontext_get_prediction failed: {str(e)}")
        return {"error": f"Failed to get prediction: {str(e)}"}

@mcp.tool()
def flux_kontext_list_predictions(limit: int = 20) -> Dict[str, Any]:
    """
    List recent predictions for your account.
    
    Args:
        limit (int): Maximum number of predictions to return (default: 20)
        
    Returns:
        Dict[str, Any]: List of recent predictions with their status and metadata
    """
    logger.info(f"Listing predictions with limit: {limit}")
    
    try:
        # Get predictions with limit
        logger.info("Fetching predictions from Replicate")
        predictions_list = list(replicate.predictions.list())[:limit]
        logger.info(f"Retrieved {len(predictions_list)} total predictions")
        
        # Filter for Flux Kontext Pro predictions
        flux_predictions = []
        for pred in predictions_list:
            if pred.model and "flux-kontext-pro" in pred.model:
                flux_predictions.append({
                    "id": pred.id,
                    "status": pred.status,
                    "created_at": pred.created_at.isoformat() if pred.created_at else None,
                    "completed_at": pred.completed_at.isoformat() if pred.completed_at else None,
                    "model": pred.model,
                    "input_prompt": pred.input.get("prompt") if pred.input else None,
                    "has_output": pred.output is not None
                })
        
        logger.info(f"Found {len(flux_predictions)} Flux Kontext Pro predictions")
        
        result = {
            "predictions": flux_predictions,
            "total_returned": len(flux_predictions)
        }
        
        logger.info("flux_kontext_list_predictions completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"flux_kontext_list_predictions failed: {str(e)}")
        return {"error": f"Failed to list predictions: {str(e)}"}

@mcp.tool()
def flux_kontext_cancel_prediction(prediction_id: str) -> Dict[str, Any]:
    """
    Cancel a running prediction.
    
    Args:
        prediction_id (str): The ID of the prediction to cancel
        
    Returns:
        Dict[str, Any]: Cancellation status and updated prediction info
    """
    logger.info(f"Canceling prediction with ID: {prediction_id}")
    
    try:
        prediction = replicate.predictions.cancel(prediction_id)
        
        logger.info(f"Prediction {prediction_id} cancellation requested")
        logger.info(f"New status: {prediction.status}")
        
        result = {
            "id": prediction.id,
            "status": prediction.status,
            "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
            "completed_at": prediction.completed_at.isoformat() if prediction.completed_at else None,
            "error": prediction.error,
            "cancelled": prediction.status == "canceled"
        }
        
        if result["cancelled"]:
            logger.info(f"Prediction {prediction_id} successfully canceled")
        else:
            logger.warning(f"Prediction {prediction_id} cancellation may not have succeeded, status: {prediction.status}")
        
        logger.info("flux_kontext_cancel_prediction completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"flux_kontext_cancel_prediction failed: {str(e)}")
        return {"error": f"Failed to cancel prediction: {str(e)}"}

@mcp.tool()
def flux_kontext_wait_for_prediction(prediction_id: str, timeout: int = 300) -> Dict[str, Any]:
    """
    Wait for a prediction to complete with optional timeout.
    
    Args:
        prediction_id (str): The ID of the prediction to wait for
        timeout (int): Maximum time to wait in seconds (default: 300)
        
    Returns:
        Dict[str, Any]: Final prediction result or timeout error
    """
    logger.info(f"Waiting for prediction {prediction_id} with timeout: {timeout} seconds")
    
    try:
        start_time = time.time()
        check_count = 0
        
        while time.time() - start_time < timeout:
            check_count += 1
            prediction = replicate.predictions.get(prediction_id)
            
            elapsed_time = int(time.time() - start_time)
            logger.info(f"Check #{check_count} - Prediction {prediction_id} status: {prediction.status} (elapsed: {elapsed_time}s)")
            
            if prediction.status in ["succeeded", "failed", "canceled"]:
                logger.info(f"Prediction {prediction_id} reached final status: {prediction.status}")
                
                result = {
                    "id": prediction.id,
                    "status": prediction.status,
                    "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
                    "completed_at": prediction.completed_at.isoformat() if prediction.completed_at else None,
                    "output": prediction.output,
                    "error": prediction.error,
                    "waited_seconds": elapsed_time,
                    "checks_performed": check_count
                }
                
                if prediction.status == "succeeded":
                    logger.info(f"Prediction {prediction_id} completed successfully after {elapsed_time} seconds")
                elif prediction.status == "failed":
                    logger.error(f"Prediction {prediction_id} failed after {elapsed_time} seconds: {prediction.error}")
                else:
                    logger.info(f"Prediction {prediction_id} was canceled after {elapsed_time} seconds")
                
                return result
            
            time.sleep(2)  # Check every 2 seconds
        
        logger.warning(f"Prediction {prediction_id} timed out after {timeout} seconds")
        return {"error": f"Prediction timed out after {timeout} seconds", "status": "timeout"}
        
    except Exception as e:
        logger.error(f"flux_kontext_wait_for_prediction failed: {str(e)}")
        return {"error": f"Failed to wait for prediction: {str(e)}"}

@mcp.tool()
def flux_kontext_get_model_info(random_string: str = "dummy") -> Dict[str, Any]:
    """
    Get information about the Flux Kontext Pro model.
    
    Returns:
        Dict[str, Any]: Model information including description, input schema, etc.
    """
    logger.info("Getting Flux Kontext Pro model information")
    
    try:
        model = replicate.models.get("black-forest-labs/flux-kontext-pro")
        
        logger.info("Successfully retrieved model information")
        
        result = {
            "name": model.name,
            "description": model.description,
            "visibility": model.visibility,
            "github_url": model.github_url,
            "paper_url": model.paper_url,
            "license_url": model.license_url,
            "cover_image_url": model.cover_image_url,
            "latest_version": {
                "id": model.latest_version.id if model.latest_version else None,
                "created_at": model.latest_version.created_at.isoformat() if model.latest_version and model.latest_version.created_at else None
            }
        }
        
        logger.info("flux_kontext_get_model_info completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"flux_kontext_get_model_info failed: {str(e)}")
        return {"error": f"Failed to get model info: {str(e)}"}

if __name__ == "__main__":
    logger.info("Starting Flux Kontext Pro MCP Server")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flux Kontext Pro MCP Server")
    parser.add_argument('transport', nargs='?', default='stdio', choices=['stdio', 'sse', 'streamable-http'],
                        help='Transport type (stdio, sse, or streamable-http)')
    args = parser.parse_args()
    
    logger.info(f"Server configuration - Transport: {args.transport}")
    
    # Log environment variables status (without exposing sensitive data)
    logger.info(f"Environment check - REPLICATE_API_TOKEN: {'SET' if os.getenv('REPLICATE_API_TOKEN') else 'NOT SET'}")
    logger.info(f"Environment check - OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    logger.info(f"Environment check - OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'DEFAULT')}")
    logger.info(f"Environment check - OPENAI_MODEL_NAME: {os.getenv('OPENAI_MODEL_NAME', 'DEFAULT')}")
    
    try:
        # Run the MCP server with the specified transport
        logger.info("Starting MCP server...")
        mcp.run(transport=args.transport)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        sys.exit(1)
