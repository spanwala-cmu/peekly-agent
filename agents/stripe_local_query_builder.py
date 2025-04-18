import json
import re
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from openai import OpenAI

class StripeMockAgent:
    """
    Stripe Mock Data Generator Agent that creates realistic mock data
    in response to natural language queries about payment/subscription data.
    """
    
    def __init__(
        self, 
        agent_endpoint: Optional[str] = None,
        agent_key: Optional[str] = None
    ):
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
            # If no Digital Ocean credentials, generate mock data locally
            return None
            
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
    
    def generate_fallback_mock_data(self, query: str) -> Dict[str, Any]:
        """
        Generate simple mock data locally when the LLM service is not available.
        
        Args:
            query: The natural language query about Stripe data
            
        Returns:
            Dictionary with mock Stripe data
        """
        # Identify the type of data needed based on keyword analysis
        query = query.lower()
        
        # Default fields for all records
        current_time = datetime.now()
        
        # Generate 5-10 records
        record_count = random.randint(5, 10)
        
        # Prepare basic response structure
        response = {
            "object": "list",
            "url": "/v1/charges",
            "has_more": False,
            "count": record_count,
            "data": []
        }
        
        # Generate different mock data based on query keywords
        if any(word in query for word in ["subscription", "recurring"]):
            # Generate subscription data
            response["url"] = "/v1/subscriptions"
            
            for i in range(record_count):
                created_date = current_time - timedelta(days=random.randint(1, 365))
                subscription = {
                    "id": f"sub_{random.randint(10000000, 99999999)}",
                    "object": "subscription",
                    "customer": f"cus_{random.randint(10000000, 99999999)}",
                    "created": int(created_date.timestamp()),
                    "current_period_start": int((created_date + timedelta(days=30 * (i % 4))).timestamp()),
                    "current_period_end": int((created_date + timedelta(days=30 * (i % 4 + 1))).timestamp()),
                    "status": random.choice(["active", "active", "active", "canceled", "past_due"]),
                    "plan": {
                        "id": random.choice(["plan_basic", "plan_pro", "plan_enterprise"]),
                        "amount": random.choice([999, 1999, 4999]),
                        "currency": "usd",
                        "interval": "month"
                    },
                    "items": {
                        "object": "list",
                        "data": [
                            {
                                "id": f"si_{random.randint(10000000, 99999999)}",
                                "price": {
                                    "id": f"price_{random.randint(10000000, 99999999)}",
                                    "product": f"prod_{random.randint(10000000, 99999999)}",
                                    "unit_amount": random.choice([999, 1999, 4999]),
                                    "currency": "usd"
                                },
                                "quantity": 1
                            }
                        ]
                    }
                }
                response["data"].append(subscription)
                
        elif any(word in query for word in ["customer", "user"]):
            # Generate customer data
            response["url"] = "/v1/customers"
            
            for i in range(record_count):
                created_date = current_time - timedelta(days=random.randint(1, 365))
                customer = {
                    "id": f"cus_{random.randint(10000000, 99999999)}",
                    "object": "customer",
                    "created": int(created_date.timestamp()),
                    "email": f"customer{i+1}@example.com",
                    "name": f"Customer {i+1}",
                    "currency": "usd",
                    "delinquent": random.choice([False, False, False, True]),
                    "metadata": {
                        "user_id": str(random.randint(1000, 9999))
                    }
                }
                response["data"].append(customer)
                
        elif any(word in query for word in ["invoice", "bill"]):
            # Generate invoice data
            response["url"] = "/v1/invoices"
            
            for i in range(record_count):
                created_date = current_time - timedelta(days=random.randint(1, 90))
                amount = random.randint(500, 10000)
                status = random.choice(["paid", "paid", "paid", "open", "uncollectible"])
                
                invoice = {
                    "id": f"in_{random.randint(10000000, 99999999)}",
                    "object": "invoice",
                    "customer": f"cus_{random.randint(10000000, 99999999)}",
                    "created": int(created_date.timestamp()),
                    "status": status,
                    "amount_due": amount,
                    "amount_paid": amount if status == "paid" else 0,
                    "amount_remaining": 0 if status == "paid" else amount,
                    "currency": "usd",
                    "subscription": f"sub_{random.randint(10000000, 99999999)}",
                    "paid": status == "paid",
                    "period_start": int((created_date - timedelta(days=30)).timestamp()),
                    "period_end": int(created_date.timestamp())
                }
                response["data"].append(invoice)
                
        else:
            # Default to charge/payment data
            response["url"] = "/v1/charges"
            
            for i in range(record_count):
                created_date = current_time - timedelta(days=random.randint(1, 90))
                amount = random.randint(500, 10000)
                
                charge = {
                    "id": f"ch_{random.randint(10000000, 99999999)}",
                    "object": "charge",
                    "amount": amount,
                    "amount_captured": amount,
                    "amount_refunded": 0,
                    "currency": "usd",
                    "customer": f"cus_{random.randint(10000000, 99999999)}",
                    "created": int(created_date.timestamp()),
                    "paid": True,
                    "refunded": False,
                    "status": "succeeded",
                    "payment_method_details": {
                        "card": {
                            "brand": random.choice(["visa", "mastercard", "amex"]),
                            "last4": f"{random.randint(1000, 9999)}"[-4:]
                        },
                        "type": "card"
                    }
                }
                response["data"].append(charge)
        
        response["count"] = len(response["data"])
        return response
    
    def generate_mock_data(self, query: str) -> Dict[str, Any]:
        """
        Generate mock Stripe data based on a natural language query.
        
        Args:
            query: The natural language query about Stripe data
            
        Returns:
            Dictionary with mock Stripe data
        """
        # Try to use the LLM service if available
        client = self.get_openai_client()
        
        if client:
            try:
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
                
            except Exception as e:
                print(f"Error using LLM service: {str(e)}")
                # Fall back to local mock data generation
                return self.generate_fallback_mock_data(query)
        else:
            # If no LLM service is available, generate mock data locally
            return self.generate_fallback_mock_data(query)