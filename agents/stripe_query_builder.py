#!/usr/bin/env python3
"""
Stripe Mock Data Generator Agent that creates realistic mock data
in response to natural language queries about payment/subscription data.
Uses Digital Ocean GenAI for data generation.
"""

import json
import re
from typing import Dict, Any
from openai import OpenAI

class StripeMockAgent:
    """
    Stripe Mock Data Generator Agent that creates realistic mock data
    in response to natural language queries about payment/subscription data.
    Uses Digital Ocean GenAI for data generation.
    """
    
    def __init__(self, agent_endpoint: str, agent_key: str):
        """
        Initialize the Stripe Mock Agent with Digital Ocean GenAI credentials.
        
        Args:
            agent_endpoint: URL endpoint for the Digital Ocean GenAI agent
            agent_key: API key for Digital Ocean GenAI service
        """
        self.agent_endpoint = agent_endpoint
        self.agent_key = agent_key
        
        # System prompt template for the LLM
        self.system_prompt = """
        You are a Stripe data simulator that generates realistic mock payment and subscription data.
        
        When given a query about payment data, generate a realistic JSON response that mimics what
        would come from a Stripe API but with completely fabricated data. The data should be consistent
        with typical e-commerce patterns and realistic for the query.
        
        Follow these guidelines when generating mock data:
        
        1. Create data that matches the specific request (e.g., subscription revenue, refunds, new customers)
        2. Use realistic patterns (daily/weekly cycles, growth trends, seasonal effects when appropriate)
        3. Include appropriate metadata and fields that would appear in real Stripe data
        4. Generate an appropriate number of records based on the query (5-20 records is usually sufficient)
        5. Use realistic values for money (appropriate ranges for the business type)
        6. Include realistic user/customer IDs, transaction IDs, etc.
        
        The generated data should be complete enough to answer the query comprehensively.
        
        All responses should be in this JSON format:
        {
            "data": [...array of records...],
            "has_more": boolean,
            "object": "list",
            "url": "/v1/endpoint",
            "count": number of records
        }
        
        Each record should include appropriate fields for the data type being requested.
        
        ALWAYS include appropriate date and time fields in your records.
        NEVER include explanations or anything outside the JSON structure.
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
    
    def generate_mock_data(self, query: str) -> Dict[str, Any]:
        """
        Generate mock Stripe data based on a natural language query using Digital Ocean GenAI.
        
        Args:
            query: The natural language query about Stripe data
            
        Returns:
            Dictionary with mock Stripe data
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
            temperature=0.7,  # Allow some randomness for varied mock data
        )
        
        # Extract the content from the response
        content = response.choices[0].message.content
        
        # Extract JSON from the content
        return self.extract_json_from_text(content)