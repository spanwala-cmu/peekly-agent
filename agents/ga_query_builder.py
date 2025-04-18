import json
import re
from typing import Dict, Any, Optional
from openai import OpenAI

class GAQueryBuilderAgent:
    """
    Google Analytics Query Builder Agent that converts natural language queries
    into valid Google Analytics 4 API query structures.
    """
    
    def __init__(
        self, 
        agent_endpoint: Optional[str] = None,
        agent_key: Optional[str] = None
    ):
        """
        Initialize the GA Query Builder agent with Digital Ocean GenAI credentials.
        
        Args:
            agent_endpoint: URL endpoint for the Digital Ocean GenAI agent
            agent_key: API key for Digital Ocean GenAI service
        """
        self.agent_endpoint = agent_endpoint
        self.agent_key = agent_key
        
        # System prompt template for the LLM
        self.system_prompt = """
        You are a Google Analytics query builder.
        Convert the user's natural language query into a valid Google Analytics 4 API query.

        Your task is to:
        1. Analyze the query to identify metrics, dimensions, and filters needed
        2. Format these into a valid Google Analytics 4 API query structure
        3. Include appropriate date ranges based on the query
        4. Make sure to properly format the query parameters according to GA4 API requirements
        5. For date ranges, start_date MUST be one of: YYYY-MM-DD format, NdaysAgo, yesterday, or today
        6. For date ranges, end_date MUST be one of: YYYY-MM-DD format, NdaysAgo, yesterday, or today
        7. ALWAYS include at least one dimension in your query (use "date" if no specific dimension is mentioned)
        8. Never use "lastMonth" or other date formats not explicitly listed above

        Sample json structure: 
        {
            "start_date": "30daysAgo",
            "end_date": "yesterday",
            "dimensions": [
                {"name": "date"}
            ],
            "metrics": [
                {"name": "activeUsers"}
            ],
            "filters": {
                "fieldName": "deviceCategory",
                "stringFilter": {
                    "value": "mobile"
                }
            },
            "orderBys": [
                {
                    "metric": {
                        "name": "activeUsers"
                    },
                    "desc": true
                }
            ],
            "limit": 10
        }
            
        Return ONLY a valid JSON object with the GA4 query structure. Do not include any explanations or text before or after the JSON.
        Remember that every GA4 query MUST include at least one dimension - if none is specified, use {"name": "date"} as the default dimension.
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
        Build a Google Analytics 4 query from a natural language query.
        
        Args:
            query: The natural language analytics query
            
        Returns:
            Dictionary with the GA4 query structure
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
        return self.extract_json_from_text(content)