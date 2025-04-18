#!/usr/bin/env python3
"""
Core Analytics Module - Multi-Agent Analytics System using LangGraph
"""

import json
import pandas as pd
import os
from typing import Dict, List, Any, TypedDict, Optional, Union, Literal
from langgraph.graph import StateGraph, END

# Import the agent classes
from agents.router import RouterAgent
from agents.ga_query_builder import GAQueryBuilderAgent
from agents.stripe_query_builder import StripeMockAgent 
from agents.synthesizer import SynthesizerAgent

# Load configuration from environment variables or use defaults
# This allows for easier configuration in different environments (local/staging/production)
def get_env(key, default=None):
    return os.environ.get(key, default)

# Google Analytics settings
SERVICE_ACCOUNT_KEY = get_env('GA_SERVICE_ACCOUNT_KEY')
PROPERTY_ID = get_env('GA_PROPERTY_ID')

# Digital Ocean GenAI endpoints and keys
ROUTER_ENDPOINT = get_env('ROUTER_ENDPOINT')
ROUTER_KEY = get_env('ROUTER_KEY')

GA_QUERY_BUILDER_ENDPOINT = get_env('GA_QUERY_BUILDER_ENDPOINT')
GA_QUERY_BUILDER_KEY = get_env('GA_QUERY_BUILDER_KEY')

STRIPE_MOCK_ENDPOINT = get_env('STRIPE_MOCK_ENDPOINT')
STRIPE_MOCK_KEY = get_env('STRIPE_MOCK_KEY')

SYNTHESIZER_ENDPOINT = get_env('SYNTHESIZER_ENDPOINT')
SYNTHESIZER_KEY = get_env('SYNTHESIZER_KEY')

# Define the state for our LangGraph
class AnalyticsState(TypedDict):
    """State for the analytics workflow."""
    query: str
    required_data_sources: List[str]
    ga_needed: bool
    stripe_needed: bool
    ga_query: Optional[Dict[str, Any]]
    ga_data: Optional[Any]
    ga_error: Optional[str]
    stripe_data: Optional[Dict[str, Any]]
    stripe_error: Optional[str]
    collected_data: Dict[str, Any]
    final_answer: str
    error: Optional[str]
    current_step: str

# Initialize all agents (moved to a function to avoid immediate initialization)
def initialize_agents():
    """Initialize all agents with their endpoints and keys."""
    if not all([ROUTER_ENDPOINT, ROUTER_KEY, GA_QUERY_BUILDER_ENDPOINT, 
               GA_QUERY_BUILDER_KEY, STRIPE_MOCK_ENDPOINT, STRIPE_MOCK_KEY,
               SYNTHESIZER_ENDPOINT, SYNTHESIZER_KEY]):
        raise EnvironmentError("Missing required environment variables for agent initialization. "
                              "Please check your .env file or environment configuration.")
        
    return {
        "router": RouterAgent(
            available_data_sources=[
                "Google Analytics", 
                "Stripe", 
                "Ad Platforms", 
                "Social Media"
            ],
            agent_endpoint=ROUTER_ENDPOINT,
            agent_key=ROUTER_KEY
        ),
        "ga_builder": GAQueryBuilderAgent(
            agent_endpoint=GA_QUERY_BUILDER_ENDPOINT,
            agent_key=GA_QUERY_BUILDER_KEY
        ),
        "stripe_agent": StripeMockAgent(
            agent_endpoint=STRIPE_MOCK_ENDPOINT,
            agent_key=STRIPE_MOCK_KEY
        ),
        "synthesizer": SynthesizerAgent(
            agent_endpoint=SYNTHESIZER_ENDPOINT,
            agent_key=SYNTHESIZER_KEY
        )
    }

# Utility function to fetch GA4 data
def fetch_ga4_data(service_account_key_path, property_id, query_json):
    """Fetch Google Analytics 4 data based on a JSON query."""
    if not service_account_key_path or not property_id:
        raise ValueError("Missing GA service account key path or property ID. "
                        "Please check your environment variables.")
        
    try:
        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.data_v1beta.types import (
            DateRange, Dimension, Metric, RunReportRequest,
            Filter, FilterExpression, OrderBy
        )
        from google.oauth2 import service_account
        
        # Authenticate using the service account
        credentials = service_account.Credentials.from_service_account_file(
            service_account_key_path,
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
            limit=query_json.get("limit", 10)
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

# LangGraph node functions
def route_query(state: AnalyticsState, agents) -> AnalyticsState:
    """Node to route the query to identify required data sources."""
    try:
        query = state["query"]
        print(f"Routing query: {query}")
        
        route_result = agents["router"].route_query(query)
        
        print(f"Required data sources: {', '.join(route_result['data_sources'])}")
        
        # Set flags for needed data sources
        ga_needed = "Google Analytics" in route_result["data_sources"]
        stripe_needed = "Stripe" in route_result["data_sources"]
        
        return {
            **state,
            "required_data_sources": route_result["data_sources"],
            "ga_needed": ga_needed,
            "stripe_needed": stripe_needed,
            "current_step": "process_ga" if ga_needed else "process_stripe"
        }
    except Exception as e:
        error_msg = f"Error in routing: {str(e)}"
        print(error_msg)
        return {**state, "error": error_msg, "current_step": "synthesize"}

def process_ga(state: AnalyticsState, agents) -> AnalyticsState:
    """Node to process Google Analytics if needed."""
    if not state["ga_needed"] or state.get("error"):
        return {**state, "current_step": "process_stripe"}
    
    try:
        query = state["query"]
        print("\nBuilding Google Analytics query...")
        
        ga_query = agents["ga_builder"].build_query(query)
        
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
        
        # Update collected_data
        collected_data = state.get("collected_data", {})
        collected_data["Google Analytics"] = ga_data
        
        return {
            **state,
            "ga_query": ga_query,
            "ga_data": ga_data,
            "collected_data": collected_data,
            "current_step": "process_stripe"
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
            "current_step": "process_stripe"
        }

def process_stripe(state: AnalyticsState, agents) -> AnalyticsState:
    """Node to process Stripe data if needed."""
    if not state["stripe_needed"] or state.get("error"):
        return {**state, "current_step": "synthesize"}
    
    try:
        query = state["query"]
        print("\nGenerating Stripe data...")
        
        stripe_data = agents["stripe_agent"].generate_mock_data(query)
        
        print(f"Generated {stripe_data['count']} records of Stripe data")
        
        # Update collected_data
        collected_data = state.get("collected_data", {})
        collected_data["Stripe"] = stripe_data
        
        return {
            **state,
            "stripe_data": stripe_data,
            "collected_data": collected_data,
            "current_step": "synthesize"
        }
    except Exception as e:
        error_msg = f"Error generating Stripe data: {str(e)}"
        print(error_msg)
        
        # Still update collected_data with error information
        collected_data = state.get("collected_data", {})
        collected_data["Stripe"] = {"error": error_msg}
        
        return {
            **state, 
            "stripe_error": error_msg,
            "collected_data": collected_data,
            "current_step": "synthesize"
        }

def synthesize_results(state: AnalyticsState, agents) -> AnalyticsState:
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
        if state.get("stripe_error"):
            print(f"Note: Stripe data collection failed: {state['stripe_error']}")
            
        final_answer = agents["synthesizer"].synthesize_data(query, collected_data)
        
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

def build_analytics_graph(agents):
    """Build and return the LangGraph workflow with agent dependency injection."""
    # Initialize the state graph
    workflow = StateGraph(AnalyticsState)
    
    # Add nodes with agents injected
    workflow.add_node("route_query", lambda state: route_query(state, agents))
    workflow.add_node("process_ga", lambda state: process_ga(state, agents))
    workflow.add_node("process_stripe", lambda state: process_stripe(state, agents))
    workflow.add_node("synthesize", lambda state: synthesize_results(state, agents))
    
    # Define edges with conditional routing
    workflow.add_conditional_edges(
        "route_query",
        route_next_step,
        {
            "process_ga": "process_ga",
            "process_stripe": "process_stripe",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_conditional_edges(
        "process_ga",
        route_next_step,
        {
            "process_stripe": "process_stripe",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_conditional_edges(
        "process_stripe",
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
        "stripe_needed": False,
        "ga_query": None,
        "ga_data": None,
        "ga_error": None,
        "stripe_data": None,
        "stripe_error": None,
        "collected_data": {},
        "final_answer": "",
        "error": None,
        "current_step": "route_query"
    }

def run_analytics_workflow(query: str) -> str:
    """Run the analytics workflow with the given query."""
    # Initialize agents
    agents = initialize_agents()
    
    # Build the graph with agents
    graph = build_analytics_graph(agents)
    
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

# Entry point for testing directly
if __name__ == "__main__":
    print("Multi-Agent Analytics System (LangGraph Implementation)")
    print("=" * 80)
    print("This system uses multiple AI agents to answer complex analytics questions.")
    
    # Sample test query
    test_query = "How many users from paid search purchased our premium product last month?"
    
    print(f"\nTesting with query: {test_query}")
    answer = run_analytics_workflow(test_query)
    
    print("\n" + "=" * 80)
    print("ANSWER:")
    print(answer)
    print("=" * 80)