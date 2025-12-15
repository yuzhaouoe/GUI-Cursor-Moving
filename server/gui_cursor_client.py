"""
Client script for GUI Cursor Grounding Server

This script sends requests to the GUI cursor grounding server to predict
cursor coordinates for a given image and query.

Usage:
    python gui_cursor_client.py --image path/to/image.png --query "Click on the search button"
    python gui_cursor_client.py --image path/to/image.png --query "Click submit" --server http://[serverip]:54302
"""

import argparse
import base64
import json
from tabnanny import check
import requests
from pathlib import Path
from PIL import Image
import io


def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        return base64_string


def encode_pil_image_to_base64(image):
    """
    Encode a PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string of the image
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_string


def send_grounding_request(server_url, image_path, query, env_kwargs=None):
    """
    Send a grounding request to the server.
    
    Args:
        server_url: URL of the server (e.g., "http://localhost:54302")
        image_path: Path to the image file or PIL Image object
        query: Text query describing what to click
        env_kwargs: Optional dictionary of environment parameters
        
    Returns:
        Server response as dictionary
    """
    # Encode image
    if isinstance(image_path, (str, Path)):
        image_base64 = encode_image_to_base64(image_path)
    else:
        # Assume it's a PIL Image
        image_base64 = encode_pil_image_to_base64(image_path)
    
    # Prepare request data
    request_data = {
        "image": image_base64,
        "query": query
    }
    
    # Add optional environment kwargs if provided
    if env_kwargs:
        request_data["env_kwargs"] = env_kwargs
    
    # Send POST request
    endpoint = f"{server_url}/gui_cursor_grounding"
    
    try:
        response = requests.post(
            endpoint,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60  # 60 second timeout
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        return response.json()
    
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return {"error": "Failed to connect to server"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e}", "response": response.text}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def check_server_health(server_url):
    """
    Check if the server is healthy and running.
    
    Args:
        server_url: URL of the server
        
    Returns:
        True if server is healthy, False otherwise
    """
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("status") == "healthy"
    except:
        return False


def main():
    # Test localhost first (app runs on port 5000)
    # external IP port
    external_port = 54302
    internal_port = 54302

    print(f"Testing localhost:{internal_port}...")
    local_health = check_server_health(f"http://localhost:{internal_port}")
    print(f"  Local health: {local_health}")
    
    # Test external IP (tunnel forwards 54302 -> local 5000)
    print(f"\nTesting 34.76.82.176:{external_port}...")
    server_url = f"http://34.76.82.176:{external_port}"
    check_health = check_server_health(server_url)
    print(f"  External health: {check_health}")
    
    if local_health and not check_health:
        print("\n⚠ Server is running locally but not accessible externally!")
        print(f"  This means port {external_port} is not properly forwarded from the gateway server.")
        print(f"  Check that SSH tunnel is running: ssh -p 41280 -R {external_port}:localhost:{internal_port} root@34.76.82.176")

if __name__ == "__main__":
    main()
