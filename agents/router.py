import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI

class RouterAgent:
    """
    Router Agent that determines appropriate data sources for analytics queries
    using Digital Ocean's GenAI Platform, with improved JSON extraction.
    """
    
    def __init__(
        self, 
        available_data_sources: List[str], 
        agent_endpoint: Optional[str] = None,
        agent_key: Optional[str] = None
    ):
        """
        Initialize the router agent with available data sources and Digital Ocean GenAI credentials.
        
        Args:
            available_data_sources: List of data source names that can be queried
            agent_endpoint: URL endpoint for the Digital Ocean GenAI agent
            agent_key: API key for Digital Ocean GenAI service
        """
        self.available_data_sources = available_data_sources
        self.agent_endpoint = agent_endpoint
        self.agent_key = agent_key
        
        # System prompt template for the LLM
        self.system_prompt = """
        You are a data source router for analytics queries. 
        Analyze the user query and determine which data sources are needed to answer it completely.

        Available data sources: {available_data_sources}

        Each data source contains different types of information:
        - Google Analytics (GA): User behavior, traffic sources, conversion tracking, page views, session data
        - Stripe: Payment processing, subscription data, invoices, customer payment methods
        - Database: Product catalog, inventory, user accounts, order history
        - CRM: Customer information, leads, sales pipeline, customer support tickets
        - Ad Platforms: Ad spend, impressions, clicks, campaign performance
        - Social Media: Engagement metrics, follower counts, post performance

        Your task is to examine the user's query and determine which data sources contain the necessary information to provide a complete answer.

        Return ONLY a valid JSON object with this exact structure:
        {{
            "data_sources": ["Source1", "Source2"], 
            "reasoning": "Brief explanation of why these sources were selected"
        }}
        
        Include ONLY data sources from the provided list of available data sources.
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
                    # If it still fails, fall back to the default response
                    pass
        
        # If we couldn't extract a valid JSON, create a default response
        default_response = {
            "data_sources": self.extract_data_sources_from_text(text),
            "reasoning": "Extracted from agent response"
        }
        return default_response
    
    def extract_data_sources_from_text(self, text: str) -> List[str]:
        """
        Extract mentioned data sources from text when JSON parsing fails.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of data sources mentioned in the text
        """
        mentioned_sources = []
        
        for source in self.available_data_sources:
            if source in text:
                mentioned_sources.append(source)
        
        # If no sources were found, return all available sources
        if not mentioned_sources:
            return self.available_data_sources
            
        return mentioned_sources
    
    def route_query_with_llm(self, query: str) -> Dict[str, Any]:
        """
        Route query using Digital Ocean GenAI Platform with improved JSON extraction.
        
        Args:
            query: The natural language analytics query
            
        Returns:
            Dictionary with data_sources and reasoning
        """
        # Create the OpenAI client for Digital Ocean
        client = self.get_openai_client()
        
        # Format the system prompt with available data sources
        formatted_system_prompt = self.system_prompt.format(
            available_data_sources=", ".join(self.available_data_sources)
        )
        
        # Create the messages for the chat completion
        messages = [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": f"Query: {query}"}
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
        result = self.extract_json_from_text(content)
        
        # Ensure data_sources only contains available sources
        result["data_sources"] = [
            ds for ds in result["data_sources"] 
            if ds in self.available_data_sources
        ]
        
        if "reasoning" not in result:
            result["reasoning"] = "Sources selected based on query analysis."
        
        return result
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to route an analytics query to appropriate data sources.
        
        Args:
            query: The natural language analytics query
            
        Returns:
            Dictionary with data_sources and reasoning
        """
        if not self.agent_endpoint or not self.agent_key:
            raise ValueError("Missing agent_endpoint or agent_key for Digital Ocean GenAI")
            
        return self.route_query_with_llm(query)