import json
import re
from typing import Dict, Any, Optional
from openai import OpenAI

class SynthesizerAgent:
    """
    Analytics Synthesizer Agent that combines data from multiple sources
    to provide comprehensive answers to user queries.
    """
    
    def __init__(
        self, 
        agent_endpoint: Optional[str] = None,
        agent_key: Optional[str] = None
    ):
        """
        Initialize the Synthesizer agent with Digital Ocean GenAI credentials.
        
        Args:
            agent_endpoint: URL endpoint for the Digital Ocean GenAI agent
            agent_key: API key for Digital Ocean GenAI service
        """
        self.agent_endpoint = agent_endpoint
        self.agent_key = agent_key
        
        # Improved system prompt template for the LLM
        self.system_prompt = """
        You are an analytics data interpreter specializing in Google Analytics and e-commerce data. Your job is to provide accurate, data-driven answers based solely on the data provided to you.

        When analyzing analytics data, follow these rules:

        1. AGGREGATE METRICS CORRECTLY:
           - For "active users" or similar metrics across multiple days/weeks, SUM the values unless the metric is already a cumulative value
           - For conversions, transactions, and revenue data, always SUM across the time period
           - For rates (conversion rate, bounce rate), calculate the proper average

        2. TIME PERIOD PRECISION:
           - When asked about "last month", verify the data actually covers the previous calendar month
           - When asked about "last week", check if the data covers the previous 7 days
           - NEVER extrapolate for time periods not covered in the data

        3. DATA INTEGRITY:
           - When faced with missing or erroneous data, acknowledge the limitations clearly
           - If one data source fails but others succeed, still provide insights from available data
           - If critical data is missing, clearly state that the query cannot be fully answered

        4. RESPONSE FORMAT:
           - Begin with a direct, concise answer to the question
           - Support the answer with exact numbers from the data
           - Clearly state the time period your answer covers
           - Note any limitations or data gaps that impact your answer
           - Add relevant insights only if directly supported by the data

        5. HANDLING ERRORS:
           - If a data source contains an error message, acknowledge it but continue with available data
           - If all data sources failed, clearly explain that no data is available to answer the query

        STRICTLY AVOID:
        - Making assumptions about data not present
        - Extrapolating trends beyond the available data
        - Inventing or estimating values not in the dataset
        - Providing vague or hedged answers when precise ones are possible

        Remember: Your value comes from providing accurate, data-supported insights, not from making educated guesses.
        """
    
    def get_openai_client(self):
        """Create and return an OpenAI client configured for Digital Ocean GenAI."""
        if not self.agent_endpoint or not self.agent_key:
            raise ValueError("Missing agent_endpoint or agent_key for Digital Ocean GenAI")
            
        return OpenAI(
            base_url=self.agent_endpoint,
            api_key=self.agent_key,
        )
    
    def synthesize_data(self, query: str, data_sources: Dict[str, Any]) -> str:
        """
        Synthesize data from multiple sources to answer the user's query.
        
        Args:
            query: The original natural language query
            data_sources: Dictionary with data from different sources (e.g., GA, Stripe)
            
        Returns:
            Synthesized answer to the query
        """
        # Create the OpenAI client for Digital Ocean
        client = self.get_openai_client()
        
        # Format the data sources into a string
        data_string = ""
        for source_name, source_data in data_sources.items():
            data_string += f"\n\n{source_name.upper()} DATA:\n"
            
            # Check if the source has an error
            if isinstance(source_data, dict) and "error" in source_data:
                data_string += f"ERROR: {source_data['error']}"
                continue
                
            if isinstance(source_data, dict):
                # JSON data (like from Stripe)
                data_string += json.dumps(source_data, indent=2)
            elif hasattr(source_data, 'to_string'):
                # DataFrame (like from GA)
                data_string += source_data.to_string()
            else:
                # Other data types
                data_string += str(source_data)
        
        # Create the messages for the chat completion
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Original query: {query}\n\nAvailable data:{data_string}\n\nPlease provide a comprehensive answer to the original query based on this data."}
        ]
        
        # Call the GenAI API
        response = client.chat.completions.create(
            model="n/a",  # Model is determined by the Digital Ocean GenAI Platform
            messages=messages,
            temperature=0.1,  # Lower temperature for more factual responses
        )
        
        # Extract the content from the response
        return response.choices[0].message.content