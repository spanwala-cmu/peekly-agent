#!/usr/bin/env python3
"""
REST API Service for Multi-Agent Analytics System 
that converts natural language queries to analytics insights.
"""

from flask import Flask, request, jsonify
import json
import os
import time
from flask_cors import CORS

# Import the analytics workflow
from analytics_core import run_analytics_workflow

# Configure environment variables with fallbacks
def load_environment():
    """Load environment variables with fallbacks for development."""
    # Get the directory where the script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to load from .env file with absolute path
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(base_dir, '.env')
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            print(f"Loaded environment variables from {env_path}")
        else:
            print("No .env file found, using system environment variables or defaults")
    except ImportError:
        print("python-dotenv not installed, using system environment variables or defaults")

    # Set development defaults if needed
    dev_defaults = {
        "PORT": "8000",
        "FLASK_ENV": "development"
    }
    
    # Apply defaults
    for key, value in dev_defaults.items():
        if key not in os.environ:
            os.environ[key] = value

    # Set development defaults if needed
    dev_defaults = {
        "PORT": "8000",
        "FLASK_ENV": "development"
    }
    
    # Apply defaults
    for key, value in dev_defaults.items():
        if key not in os.environ:
            os.environ[key] = value

# Load environment variables
load_environment()

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time()
    })

@app.route('/api/v1/analyze', methods=['POST'])
def analyze():
    """Main endpoint to analyze a natural language analytics query."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Extract query from request
        query = data.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        # Optional parameters
        # You can extend this to support additional parameters like timeframe, etc.
        params = data.get('parameters', {})
        
        # Process the query using the analytics workflow
        result = run_analytics_workflow(query)
        
        # Return the results
        return jsonify({
            "query": query,
            "result": result,
            "timestamp": time.time()
        })
        
    except Exception as e:
        # Log the error
        print(f"Error processing request: {str(e)}")
        
        # Return error response
        return jsonify({
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with basic documentation."""
    return jsonify({
        "name": "Multi-Agent Analytics API",
        "version": "1.0.0",
        "description": "REST API Service for natural language analytics queries",
        "endpoints": {
            "/": "This documentation",
            "/health": "Health check endpoint",
            "/api/v1/analyze": "Main endpoint for analytics queries (POST)"
        },
        "example": {
            "request": {
                "method": "POST",
                "url": "/api/v1/analyze",
                "body": {
                    "query": "How many users from paid search purchased our premium product last month?"
                }
            }
        },
        "timestamp": time.time()
    })

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8000))
    
    # Show configuration info
    print(f"Starting Multi-Agent Analytics API on port {port}")
    print(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV') == 'development')