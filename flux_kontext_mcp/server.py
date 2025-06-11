import os
import sys
import argparse
from typing import Any, Dict, List, Optional
import replicate
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

def get_api_token() -> str:
    """Get the Replicate API token from environment variables"""
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        raise ValueError("REPLICATE_API_TOKEN environment variable is required")
    return api_token

# Set the API token for replicate
os.environ["REPLICATE_API_TOKEN"] = get_api_token()

mcp = FastMCP("flux-kontext-pro")

def rewrite_kontext_prompt(prompt: str) -> str:
    """
    Rewrite the prompt to be more specific and detailed using OpenAI API.
    Falls back to original prompt if API is unavailable or fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return prompt
    
    try:
        # Get OpenAI configuration from environment variables
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        
        # Initialize OpenAI client
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # System prompt to instruct the model
        system_prompt = """You are a helpful assistant that translates and polishes image generation prompts to English. Your task is to:

1. If the input prompt is not in English, translate it to English
2. Polish and improve the prompt for better image generation results
3. Keep the original intention and meaning intact
4. Do not add creative elements that weren't in the original prompt
5. Return only the improved prompt, no explanations or additional text

Be conservative and faithful to the original intent."""
        
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
        
        # Extract the improved prompt
        improved_prompt = response.choices[0].message.content.strip()
        return improved_prompt if improved_prompt else prompt
        
    except Exception as e:
        # Return original prompt if any error occurs
        print(f"Warning: OpenAI API call failed: {str(e)}", file=sys.stderr)
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
        
        # Run the model
        output = replicate.run(
            "black-forest-labs/flux-kontext-pro",
            input=input_params
        )
        
        return {
            "status": "success",
            "image_url": output,
            "model": "black-forest-labs/flux-kontext-pro",
            "input": input_params
        }
        
    except Exception as e:
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
        
        # Create a prediction without waiting
        prediction = replicate.predictions.create(
            model="black-forest-labs/flux-kontext-pro",
            input=input_params
        )
        
        return {
            "prediction_id": prediction.id,
            "status": prediction.status,
            "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
            "input": input_params,
            "urls": {
                "get": prediction.urls.get if hasattr(prediction, 'urls') else None,
                "cancel": prediction.urls.cancel if hasattr(prediction, 'urls') else None
            }
        }
        
    except Exception as e:
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
    try:
        prediction = replicate.predictions.get(prediction_id)
        
        return {
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
        
    except Exception as e:
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
    try:
        # Get predictions with limit
        predictions_list = list(replicate.predictions.list())[:limit]
        
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
        
        return {
            "predictions": flux_predictions,
            "total_returned": len(flux_predictions)
        }
        
    except Exception as e:
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
    try:
        prediction = replicate.predictions.cancel(prediction_id)
        
        return {
            "id": prediction.id,
            "status": prediction.status,
            "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
            "completed_at": prediction.completed_at.isoformat() if prediction.completed_at else None,
            "error": prediction.error,
            "cancelled": prediction.status == "canceled"
        }
        
    except Exception as e:
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
    try:
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            prediction = replicate.predictions.get(prediction_id)
            
            if prediction.status in ["succeeded", "failed", "canceled"]:
                return {
                    "id": prediction.id,
                    "status": prediction.status,
                    "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
                    "completed_at": prediction.completed_at.isoformat() if prediction.completed_at else None,
                    "output": prediction.output,
                    "error": prediction.error,
                    "waited_seconds": int(time.time() - start_time)
                }
            
            time.sleep(2)  # Check every 2 seconds
        
        return {"error": f"Prediction timed out after {timeout} seconds", "status": "timeout"}
        
    except Exception as e:
        return {"error": f"Failed to wait for prediction: {str(e)}"}

@mcp.tool()
def flux_kontext_get_model_info() -> Dict[str, Any]:
    """
    Get information about the Flux Kontext Pro model.
    
    Returns:
        Dict[str, Any]: Model information including description, input schema, etc.
    """
    try:
        model = replicate.models.get("black-forest-labs/flux-kontext-pro")
        
        return {
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
        
    except Exception as e:
        return {"error": f"Failed to get model info: {str(e)}"}

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flux Kontext Pro MCP Server")
    parser.add_argument('transport', nargs='?', default='stdio', choices=['stdio', 'sse', 'streamable-http'],
                        help='Transport type (stdio, sse, or streamable-http)')
    args = parser.parse_args()
    
    # Run the MCP server with the specified transport
    mcp.run(transport=args.transport)
