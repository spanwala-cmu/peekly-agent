import json
import re
from typing import Dict, Any, Optional
from openai import OpenAI
from datetime import datetime, timedelta

class ShopifyQueryBuilderAgent:
    """
    Shopify Query Builder Agent that converts natural language queries
    into valid Shopify Admin REST API query structures and endpoints.
    This improved version handles date ranges dynamically.
    """
    
    def __init__(
        self, 
        agent_endpoint: Optional[str] = None,
        agent_key: Optional[str] = None
    ):
        """
        Initialize the Shopify Query Builder agent with Digital Ocean GenAI credentials.
        
        Args:
            agent_endpoint: URL endpoint for the Digital Ocean GenAI agent
            agent_key: API key for Digital Ocean GenAI service
        """
        self.agent_endpoint = agent_endpoint
        self.agent_key = agent_key
        
        # System prompt template for the LLM
        self.system_prompt = """
        You are a Shopify REST Admin API query builder.
        Convert the user's natural language query into a valid Shopify Admin REST API query.

        Your task is to:
        1. Analyze the query to identify what Shopify resource the user wants information about
        2. Determine the appropriate REST Admin API endpoint to use
        3. Identify any query parameters needed for filtering
        4. Only generate read-only operations (GET requests)
        5. Format the response as a properly structured API call
        6. For time-based queries, use placeholder tokens that will be replaced with actual dates

        Common Shopify REST Admin API endpoints for read operations include:
        - GET products.json - Get product information
        - GET orders.json - Get order information
        - GET customers.json - Get customer information
        - GET inventory_levels.json - Get inventory information
        - GET collects.json - Get product collection associations
        - GET collections.json - Get collection information
        - GET shop.json - Get shop details

        For time-based queries, use the following placeholder tokens:
        - {{TODAY}} - Will be replaced with today's date
        - {{YESTERDAY}} - Will be replaced with yesterday's date
        - {{LAST_WEEK_START}} - Will be replaced with the start date of last week
        - {{LAST_WEEK_END}} - Will be replaced with the end date of last week
        - {{LAST_MONTH_START}} - Will be replaced with the start date of last month
        - {{LAST_MONTH_END}} - Will be replaced with the end date of last month
        - {{THIS_MONTH_START}} - Will be replaced with the start date of current month
        - {{THIS_MONTH_END}} - Will be replaced with the current date

        For example, if the query mentions "last week", use:
        "created_at_min": "{{LAST_WEEK_START}}",
        "created_at_max": "{{LAST_WEEK_END}}"

        Note that all paths are relative to /admin/api/{VERSION}/ and should NOT include this prefix.
        For pagination, always use cursor-based pagination with the limit parameter (NOT page-based pagination).

        Sample JSON response structure:
        {
            "method": "GET",
            "endpoint": "orders.json",
            "query_params": {
                "limit": 250,
                "status": "any",
                "created_at_min": "{{LAST_WEEK_START}}",
                "created_at_max": "{{LAST_WEEK_END}}"
            },
            "description": "Get orders from last week"
        }
            
        Return ONLY a valid JSON object with the Shopify API query structure. Do not include any explanations or text before or after the JSON.
        Remember to ONLY create read operations (GET requests).
        Do NOT include the /admin/api/{VERSION}/ prefix in the endpoint.
        """
    
    def get_openai_client(self):
        """Create and return an OpenAI client configured for Digital Ocean GenAI."""
        if not self.agent_endpoint or not self.agent_key:
            raise ValueError("Missing agent_endpoint or agent_key for Digital Ocean GenAI")
            
        return OpenAI(
            base_url=self.agent_endpoint,
            api_key=self.agent_key,
        )
    
    def extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract a JSON object from text that might contain explanatory content.
        
        Args:
            text: The text that might contain a JSON object
            
        Returns:
            Extracted JSON object as a dictionary
        """
        # Try to find a JSON object in the text using regex
        json_match = re.search(r'({[\s\S]*})', text)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If parsing fails, try to clean up the JSON string
                # Remove any formatting characters that might be causing issues
                clean_json_str = re.sub(r'[\n\r\t]', '', json_str)
                try:
                    return json.loads(clean_json_str)
                except json.JSONDecodeError:
                    # If it still fails, raise an error
                    raise ValueError(f"Failed to parse JSON from response: {text}")
        
        # If we couldn't find a JSON object, raise an error
        raise ValueError(f"No JSON object found in response: {text}")
    
    def build_query(self, query: str) -> Dict[str, Any]:
        """
        Build a Shopify Admin REST API query from a natural language query.
        
        Args:
            query: The natural language query about shop data
            
        Returns:
            Dictionary with the Shopify API query structure
        """
        # Create the OpenAI client for Digital Ocean
        client = self.get_openai_client()
        
        # Create the messages for the chat completion
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Call the GenAI API
        response = client.chat.completions.create(
            model="n/a",  # Model is determined by the Digital Ocean GenAI Platform
            messages=messages,
            temperature=0.0,  # Deterministic output
        )
        
        # Extract the content from the response
        content = response.choices[0].message.content
        
        # Extract JSON from the content
        query_data = self.extract_json_from_text(content)
        
        # Replace date placeholders with actual dates
        query_data = self._replace_date_placeholders(query_data)
        
        return query_data
    
    def _replace_date_placeholders(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace date placeholders with actual date values.
        
        Args:
            query_data: Query data with potential placeholder tokens
            
        Returns:
            Query data with actual date values
        """
        # Get current date and time
        now = datetime.now()
        today = now.date()
        yesterday = today - timedelta(days=1)
        
        # Calculate last week (previous 7 days)
        last_week_start = today - timedelta(days=7)
        last_week_end = today - timedelta(days=1)
        
        # Calculate last month (previous calendar month)
        if now.month == 1:
            last_month_start = datetime(now.year - 1, 12, 1).date()
            last_month_end = datetime(now.year, 1, 1).date() - timedelta(days=1)
        else:
            last_month_start = datetime(now.year, now.month - 1, 1).date()
            last_month_end = datetime(now.year, now.month, 1).date() - timedelta(days=1)
        
        # Calculate this month (current calendar month)
        this_month_start = datetime(now.year, now.month, 1).date()
        this_month_end = today
        
        # Define the placeholder mappings
        placeholders = {
            "{{TODAY}}": today.strftime("%Y-%m-%dT00:00:00-00:00"),
            "{{YESTERDAY}}": yesterday.strftime("%Y-%m-%dT00:00:00-00:00"),
            "{{LAST_WEEK_START}}": last_week_start.strftime("%Y-%m-%dT00:00:00-00:00"),
            "{{LAST_WEEK_END}}": last_week_end.strftime("%Y-%m-%dT23:59:59-00:00"),
            "{{LAST_MONTH_START}}": last_month_start.strftime("%Y-%m-%dT00:00:00-00:00"),
            "{{LAST_MONTH_END}}": last_month_end.strftime("%Y-%m-%dT23:59:59-00:00"),
            "{{THIS_MONTH_START}}": this_month_start.strftime("%Y-%m-%dT00:00:00-00:00"),
            "{{THIS_MONTH_END}}": this_month_end.strftime("%Y-%m-%dT23:59:59-00:00")
        }
        
        # If there are query parameters with date strings, replace placeholders
        if "query_params" in query_data:
            for key, value in query_data["query_params"].items():
                if isinstance(value, str):
                    for placeholder, date_str in placeholders.items():
                        if placeholder in value:
                            query_data["query_params"][key] = value.replace(placeholder, date_str)
        
        return query_data
    
    def format_api_request(self, query_data: Dict[str, Any], api_version: str = '2023-10', 
                          store_domain: str = None, access_token: str = None) -> Dict[str, Any]:
        """
        Format the query data into actual API request details
        
        Args:
            query_data: The query data returned by build_query
            api_version: Shopify API version to use
            store_domain: The Shopify store domain
            access_token: Shopify access token
            
        Returns:
            Dictionary with formatted API request details
        """
        if not store_domain or not access_token:
            raise ValueError("store_domain and access_token are required")
            
        # Construct the full URL with properly formatted endpoint
        # The endpoint from query_data doesn't include the /admin/api/{VERSION}/ prefix
        endpoint = query_data["endpoint"]
        url = f"https://{store_domain}/admin/api/{api_version}/{endpoint}"
        
        # Prepare headers with the access token
        headers = {
            "X-Shopify-Access-Token": access_token,
            "Content-Type": "application/json",
        }
        
        # Handle query parameters
        params = query_data.get("query_params", {})
        
        # Return the formatted request details
        return {
            "method": query_data["method"],
            "url": url,
            "headers": headers,
            "params": params,
            "description": query_data.get("description", "")
        }