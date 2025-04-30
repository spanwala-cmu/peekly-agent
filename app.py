#!/usr/bin/env python3
"""
REST API Service for Multi-Agent Analytics System 
that integrates Router, GA Query Builder, Shopify Query Builder, 
and Synthesizer agents to answer complex analytics queries.
"""

from flask import Flask, request, jsonify
import json
import os
import time
from flask_cors import CORS
import pandas as pd
from typing import Dict, List, Any, TypedDict, Optional, Union, Literal
from langgraph.graph import StateGraph, END
import asyncio
import requests

# Import the agent classes
from agents.router import RouterAgent
from agents.ga_query_builder import GAQueryBuilderAgent
from agents.shopify_query_builder import ShopifyQueryBuilderAgent
from agents.synthesizer import SynthesizerAgent

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
    
    # Required environment variables (these should be set in production)
    required_vars = [
        "SERVICE_ACCOUNT_KEY",
        "PROPERTY_ID",
        "SHOPIFY_API_VERSION",
        "SHOPIFY_STORE_DOMAIN",
        "SHOPIFY_ACCESS_TOKEN",
        "ROUTER_ENDPOINT",
        "ROUTER_KEY",
        "GA_QUERY_BUILDER_ENDPOINT",
        "GA_QUERY_BUILDER_KEY",
        "SHOPIFY_QUERY_BUILDER_ENDPOINT",
        "SHOPIFY_QUERY_BUILDER_KEY",
        "SYNTHESIZER_ENDPOINT",
        "SYNTHESIZER_KEY"
    ]
    
    # Check for required variables
    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars and os.environ.get("FLASK_ENV") == "production":
        print(f"WARNING: Missing required environment variables: {', '.join(missing_vars)}")
    
    # Apply defaults only if not in environment
    for key, value in dev_defaults.items():
        if key not in os.environ:
            os.environ[key] = value

# Load environment variables
load_environment()

# Get configuration from environment
SERVICE_ACCOUNT_KEY = os.environ.get("SERVICE_ACCOUNT_KEY")
PROPERTY_ID = os.environ.get("PROPERTY_ID")
SHOPIFY_API_VERSION = os.environ.get("SHOPIFY_API_VERSION")
SHOPIFY_STORE_DOMAIN = os.environ.get("SHOPIFY_STORE_DOMAIN")
SHOPIFY_ACCESS_TOKEN = os.environ.get("SHOPIFY_ACCESS_TOKEN")
ROUTER_ENDPOINT = os.environ.get("ROUTER_ENDPOINT")
ROUTER_KEY = os.environ.get("ROUTER_KEY")
GA_QUERY_BUILDER_ENDPOINT = os.environ.get("GA_QUERY_BUILDER_ENDPOINT")
GA_QUERY_BUILDER_KEY = os.environ.get("GA_QUERY_BUILDER_KEY")
SHOPIFY_QUERY_BUILDER_ENDPOINT = os.environ.get("SHOPIFY_QUERY_BUILDER_ENDPOINT")
SHOPIFY_QUERY_BUILDER_KEY = os.environ.get("SHOPIFY_QUERY_BUILDER_KEY")
SYNTHESIZER_ENDPOINT = os.environ.get("SYNTHESIZER_ENDPOINT")
SYNTHESIZER_KEY = os.environ.get("SYNTHESIZER_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Define the state for our LangGraph
class AnalyticsState(TypedDict):
    """State for the analytics workflow."""
    query: str
    required_data_sources: List[str]
    ga_needed: bool
    shopify_needed: bool
    ga_query: Optional[Dict[str, Any]]
    ga_data: Optional[Any]
    ga_error: Optional[str]
    shopify_query: Optional[Dict[str, Any]]
    shopify_data: Optional[Dict[str, Any]]
    shopify_error: Optional[str]
    collected_data: Dict[str, Any]
    final_answer: str
    error: Optional[str]
    current_step: str

# Initialize all agents
router = RouterAgent(
    available_data_sources=[
        "Google Analytics", 
        "Shopify"
    ],
    agent_endpoint=ROUTER_ENDPOINT,
    agent_key=ROUTER_KEY
)

ga_builder = GAQueryBuilderAgent(
    agent_endpoint=GA_QUERY_BUILDER_ENDPOINT,
    agent_key=GA_QUERY_BUILDER_KEY
)

shopify_builder = ShopifyQueryBuilderAgent(
    agent_endpoint=SHOPIFY_QUERY_BUILDER_ENDPOINT,
    agent_key=SHOPIFY_QUERY_BUILDER_KEY
)

synthesizer = SynthesizerAgent(
    agent_endpoint=SYNTHESIZER_ENDPOINT,
    agent_key=SYNTHESIZER_KEY
)

# Utility function to fetch GA4 data
def fetch_ga4_data(service_account_key_path, property_id, query_json):
    """Fetch Google Analytics 4 data based on a JSON query."""
    try:
        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.data_v1beta.types import (
            DateRange, Dimension, Metric, RunReportRequest,
            Filter, FilterExpression, OrderBy
        )
        from google.oauth2 import service_account
        
        # Authenticate using the service account
        info = json.loads(os.environ["SERVICE_ACCOUNT_KEY"])
        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/analytics.readonly"],
        )

        # Create the Analytics Data API client
        client = BetaAnalyticsDataClient(credentials=credentials)

        # Parse date range
        date_ranges = [DateRange(
            start_date=query_json.get("start_date", "30daysAgo"), 
            end_date=query_json.get("end_date", "yesterday")
        )]

        # Parse dimensions and metrics
        dimensions = [Dimension(name=d["name"]) for d in query_json["dimensions"]]
        metrics = [Metric(name=m["name"]) for m in query_json["metrics"]]

        # Parse filters (if any)
        dimension_filter = None
        if "filters" in query_json and "fieldName" in query_json["filters"]:
            dimension_filter = FilterExpression(
                filter=Filter(
                    field_name=query_json["filters"]["fieldName"],
                    string_filter=Filter.StringFilter(
                        value=query_json["filters"]["stringFilter"]["value"]
                    )
                )
            )

        # Construct the API request
        request = RunReportRequest(
            property=property_id,
            date_ranges=date_ranges,
            dimensions=dimensions,
            metrics=metrics,
            limit=query_json.get("limit", 50)
        )

        # Add filters if present
        if dimension_filter:
            request.dimension_filter = dimension_filter

        # Execute the API request
        response = client.run_report(request)

        # Process the response into a DataFrame
        rows = []
        for row in response.rows:
            dimension_values = [dim.value for dim in row.dimension_values]
            metric_values = [float(metric.value) if '.' in metric.value else int(metric.value) 
                            for metric in row.metric_values]
            rows.append(dimension_values + metric_values)

        # Define column names
        column_names = [d["name"] for d in query_json["dimensions"]] + [m["name"] for m in query_json["metrics"]]

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=column_names)
        return df
    except Exception as e:
        print(f"Error fetching GA4 data: {str(e)}")
        raise e

# Function to fetch data from Shopify
def fetch_shopify_data(api_request):
    """Fetch data from Shopify based on the API request."""
    try:
        print(f"Executing Shopify request: {api_request['url']}")
        
        response = requests.request(
            method=api_request["method"],
            url=api_request["url"],
            headers=api_request["headers"],
            params=api_request["params"]
        )
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Convert response to a usable format
        data = response.json()
        
        # Try to convert to DataFrame if possible
        try:
            # Determine the main data key in the response
            # Shopify often returns data with the primary resource as the key
            if len(data.keys()) == 1:
                main_key = list(data.keys())[0]
                items = data[main_key]
                
                # If it's a list, convert to DataFrame
                if isinstance(items, list):
                    df = pd.DataFrame(items)
                    return df
            
            # If we can't easily convert to DataFrame, return the raw JSON
            return data
        except Exception as e:
            print(f"Could not convert Shopify data to DataFrame: {str(e)}")
            return data
            
    except Exception as e:
        print(f"Error fetching Shopify data: {str(e)}")
        raise e

# LangGraph node functions
def route_query(state: AnalyticsState) -> AnalyticsState:
    """Node to route the query to identify required data sources."""
    try:
        query = state["query"]
        print(f"Routing query: {query}")
        
        route_result = router.route_query(query)
        
        print(f"Required data sources: {', '.join(route_result['data_sources'])}")
        
        # Set flags for needed data sources
        ga_needed = "Google Analytics" in route_result["data_sources"]
        shopify_needed = "Shopify" in route_result["data_sources"]
        
        return {
            **state,
            "required_data_sources": route_result["data_sources"],
            "ga_needed": ga_needed,
            "shopify_needed": shopify_needed,
            "current_step": "process_ga" if ga_needed else "process_shopify"
        }
    except Exception as e:
        error_msg = f"Error in routing: {str(e)}"
        print(error_msg)
        return {**state, "error": error_msg, "current_step": "synthesize"}

def process_ga(state: AnalyticsState) -> AnalyticsState:
    """Node to process Google Analytics if needed."""
    if not state["ga_needed"] or state.get("error"):
        return {**state, "current_step": "process_shopify"}
    
    try:
        query = state["query"]
        print("\nBuilding Google Analytics query...")
        
        ga_query = ga_builder.build_query(query)
        
        print("GA4 Query structure:")
        print(json.dumps(ga_query, indent=2))
        
        # Fix common GA4 metric errors
        if "metrics" in ga_query:
            for i, metric in enumerate(ga_query["metrics"]):
                if metric["name"] == "purchases":
                    print("Replacing invalid metric 'purchases' with 'transactions'")
                    ga_query["metrics"][i]["name"] = "transactions"
        
        print("\nFetching data from Google Analytics...")
        ga_data = fetch_ga4_data(SERVICE_ACCOUNT_KEY, PROPERTY_ID, ga_query)
        
        print(f"Retrieved {len(ga_data)} rows of GA data")
        print(ga_data)
        
        # Update collected_data
        collected_data = state.get("collected_data", {})
        collected_data["Google Analytics"] = ga_data
        
        return {
            **state,
            "ga_query": ga_query,
            "ga_data": ga_data,
            "collected_data": collected_data,
            "current_step": "process_shopify"
        }
    except Exception as e:
        error_msg = f"Error processing GA data: {str(e)}"
        print(error_msg)
        
        # Still update collected_data with error information
        collected_data = state.get("collected_data", {})
        collected_data["Google Analytics"] = {"error": error_msg}
        
        return {
            **state, 
            "ga_error": error_msg,
            "collected_data": collected_data,
            "current_step": "process_shopify"
        }

def process_shopify(state: AnalyticsState) -> AnalyticsState:
    """Node to process Shopify data if needed."""
    if not state["shopify_needed"] or state.get("error"):
        return {**state, "current_step": "synthesize"}
    
    try:
        query = state["query"]
        print("\nBuilding Shopify query...")
        
        # Build the Shopify query
        shopify_query = shopify_builder.build_query(query)
        
        print("Shopify Query structure:")
        print(json.dumps(shopify_query, indent=2))
        
        # Format the API request
        api_request = shopify_builder.format_api_request(
            shopify_query,
            api_version=SHOPIFY_API_VERSION,
            store_domain=SHOPIFY_STORE_DOMAIN,
            access_token=SHOPIFY_ACCESS_TOKEN
        )
        
        print("\nFetching data from Shopify...")
        shopify_data = fetch_shopify_data(api_request)
        
        # Print information about the data we got back
        if isinstance(shopify_data, pd.DataFrame):
            print(f"Retrieved {len(shopify_data)} rows of Shopify data")
            print(shopify_data.head())
        else:
            print("Retrieved Shopify data as JSON")
            print(json.dumps(shopify_data, indent=2)[:500] + "..." if len(json.dumps(shopify_data)) > 500 else json.dumps(shopify_data, indent=2))
        
        # Update collected_data
        collected_data = state.get("collected_data", {})
        collected_data["Shopify"] = shopify_data
        
        return {
            **state,
            "shopify_query": shopify_query,
            "shopify_data": shopify_data,
            "collected_data": collected_data,
            "current_step": "synthesize"
        }
    except Exception as e:
        error_msg = f"Error processing Shopify data: {str(e)}"
        print(error_msg)
        
        # Still update collected_data with error information
        collected_data = state.get("collected_data", {})
        collected_data["Shopify"] = {"error": error_msg}
        
        return {
            **state, 
            "shopify_error": error_msg,
            "collected_data": collected_data,
            "current_step": "synthesize"
        }

def synthesize_results(state: AnalyticsState) -> AnalyticsState:
    """Node to synthesize results from all collected data sources."""
    if state.get("error"):
        return state
        
    try:
        query = state["query"]
        collected_data = state.get("collected_data", {})
        
        if not collected_data:
            return {**state, "error": "No data was collected from any source", "current_step": "end"}
        
        print("\nSynthesizing insights from collected data...")
        
        # Include information about any failed data sources
        if state.get("ga_error"):
            print(f"Note: Google Analytics data collection failed: {state['ga_error']}")
        if state.get("shopify_error"):
            print(f"Note: Shopify data collection failed: {state['shopify_error']}")
            
        final_answer = synthesizer.synthesize_data(query, collected_data)
        
        return {
            **state,
            "final_answer": final_answer,
            "current_step": "end"
        }
    except Exception as e:
        error_msg = f"Error in synthesis: {str(e)}"
        print(error_msg)
        return {**state, "error": error_msg, "current_step": "end"}

def route_next_step(state: AnalyticsState) -> str:
    """Determine the next step based on the current state."""
    return state["current_step"]

def build_analytics_graph():
    """Build and return the LangGraph workflow."""
    # Initialize the state graph
    workflow = StateGraph(AnalyticsState)
    
    # Add nodes
    workflow.add_node("route_query", route_query)
    workflow.add_node("process_ga", process_ga)
    workflow.add_node("process_shopify", process_shopify)
    workflow.add_node("synthesize", synthesize_results)
    
    # Define edges with conditional routing
    workflow.add_conditional_edges(
        "route_query",
        route_next_step,
        {
            "process_ga": "process_ga",
            "process_shopify": "process_shopify",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_conditional_edges(
        "process_ga",
        route_next_step,
        {
            "process_shopify": "process_shopify",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_conditional_edges(
        "process_shopify",
        route_next_step,
        {
            "synthesize": "synthesize",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "synthesize",
        route_next_step,
        {
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("route_query")
    
    return workflow.compile()

def initialize_state(query: str) -> AnalyticsState:
    """Initialize the state with the user query."""
    return {
        "query": query,
        "required_data_sources": [],
        "ga_needed": False,
        "shopify_needed": False,
        "ga_query": None,
        "ga_data": None,
        "ga_error": None,
        "shopify_query": None,
        "shopify_data": None,
        "shopify_error": None,
        "collected_data": {},
        "final_answer": "",
        "error": None,
        "current_step": "route_query"
    }

def run_analytics_workflow(query: str) -> str:
    """Run the analytics workflow with the given query."""
    # Build the graph
    graph = build_analytics_graph()
    
    # Initialize the state
    initial_state = initialize_state(query)
    
    # Run the workflow
    print(f"Starting analytics workflow for query: {query}")
    final_state = graph.invoke(initial_state)
    
    # Check if an error occurred
    if final_state.get("error"):
        return f"Error: {final_state['error']}"
    
    # Return the final answer
    return final_state.get("final_answer", "No answer generated.")

# API Routes
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
        "description": "REST API Service for natural language analytics queries using LangGraph",
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