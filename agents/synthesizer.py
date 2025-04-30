import json
import re
from typing import Dict, Any, Optional, Union, List
from openai import OpenAI
import pandas as pd
from utils.calculator import CalculationHelper

class SynthesizerAgent:
    """
    Analytics Synthesizer Agent that combines data from multiple sources
    to provide comprehensive answers to user queries with improved handling
    of metrics like active users.
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
        
        # Initialize the calculation helper
        self.calculator = CalculationHelper()
        
        # Improved system prompt template for the LLM with better guidance on metrics
        self.system_prompt = """
        You are an analytics data interpreter specializing in Google Analytics and e-commerce data. Your job is to provide accurate, data-driven answers based solely on the data provided to you.

When analyzing analytics data, follow these rules:

1. CALCULATION PRECISION:
   - For all mathematical operations, perform explicit step-by-step calculations
   - Double-check all calculations before presenting final numbers
   - Round only at the final step, maintaining full precision during intermediate calculations
   - When calculating percentage changes, use the formula ((New - Old) / Old) × 100%
   - For calculating averages, clearly specify weighted vs. simple average methodology

2. AGGREGATE METRICS CORRECTLY:
   - For "active users" or visitor metrics:
     - When reporting on active users, clearly explain that this is typically a sum of daily active users, 
       which may count the same user multiple times if they return on different days
     - Use terms like "user sessions" instead of just "users" when appropriate
     - If the data includes unique user counts, prioritize reporting these
   - For conversions, transactions, and revenue data, always SUM across the time period
   - For rates (conversion rate, bounce rate), calculate as: (Total conversions ÷ Total sessions) × 100, NOT by averaging daily rates
   - For average order value, calculate as: Total revenue ÷ Total orders

3. TIME PERIOD PRECISION:
   - When asked about "last month", verify the data covers the exact previous calendar month (all days)
   - When asked about "last week", confirm if the data means previous 7 days or previous calendar week
   - When comparing "year over year" or "month over month", ensure identical day counts and adjust for seasonality
   - NEVER extrapolate for time periods not covered in the data
   - Always specify the exact date range in your answer (e.g., "For the period Jan 1-31, 2023...")

4. DATA INTEGRITY:
   - When faced with missing or erroneous data, acknowledge the limitations clearly
   - If one data source fails but others succeed, still provide insights from available data
   - If critical data is missing, clearly state which specific data is missing and why the query cannot be fully answered
   - Flag any metric discrepancies (e.g., if sessions don't match across reports)
   - Note any sampling in the data and its potential impact on accuracy

5. RESPONSE FORMAT:
   - Begin with a direct, conversational answer (like you're chatting with a colleague)
   - Present the key finding upfront (e.g., "Looking at your data, it seems your website had about 5,823 active user sessions in the last two weeks.")
   - Explain calculations in simple, clear terms, not technical ones
   - If there are important caveats about the data (like active users potentially being counted multiple times), mention this in a friendly way
   - End with a brief insight or follow-up question that would be valuable to the user

6. HANDLING ERRORS:
   - If you encounter division by zero, clearly state this issue rather than showing "Infinity" or "NaN"
   - If a percentage increase is from zero to a positive number, state "new activity" rather than "infinite increase"
   - If a data source contains an error message, acknowledge it but continue with available data
   - If all data sources failed, clearly explain that no data is available to answer the query
   - For unexpected results (e.g., 1000% increases), verify the data before reporting

STRICTLY AVOID:
- Making assumptions about data not present
- Extrapolating trends beyond the available data
- Inventing or estimating values not in the dataset
- Providing vague or hedged answers when precise ones are possible
- Averaging percentages directly without recalculating from the base metrics
- Confusing count metrics (sum) with unique metrics (deduplicated)
- Using incorrect formulas for key metrics like conversion rate or YoY change

NOTE: For all calculations, the system will provide precise calculation results. Reference these exact values in your response and clearly indicate when calculations have been performed by the calculator system.

IMPORTANT FORMATTING INSTRUCTIONS:
- Write in a conversational, helpful tone like a data analyst speaking to a business owner
- Use clear sections with headings if multiple topics are covered
- End with a brief, actionable conclusion or insight based on the data"""
    
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
        # Update calculator context with the data sources
        self.calculator.update_context(data_sources)
        
        # Perform pre-calculations based on the query
        calculation_results = self._perform_calculations(query, data_sources)
        
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
                
            if isinstance(source_data, pd.DataFrame):
                # DataFrame (like from GA)
                data_string += source_data.to_string()
                
                # Add column description for clarity
                data_string += "\n\nColumn descriptions:"
                for col in source_data.columns:
                    if col == 'date':
                        data_string += f"\n- {col}: Date of the data point"
                    elif col == 'active_users':
                        data_string += f"\n- {col}: Number of active user sessions for that day (NOT unique users)"
                    elif col == 'sessions':
                        data_string += f"\n- {col}: Number of sessions for that day"
                    else:
                        data_string += f"\n- {col}: {col.replace('_', ' ').capitalize()}"
            elif hasattr(source_data, 'to_string'):
                # Other dataframe-like object
                data_string += source_data.to_string()
            elif isinstance(source_data, dict):
                # JSON data (like from Stripe)
                data_string += json.dumps(source_data, indent=2)
            else:
                # Other data types
                data_string += str(source_data)
        
        # Add calculation results to the context
        if calculation_results:
            data_string += "\n\nPRE-CALCULATED METRICS:\n"
            for calc_name, calc_info in calculation_results.items():
                data_string += f"\n{calc_name}:\n"
                
                # Add value with proper formatting
                value = calc_info.get('value')
                if isinstance(value, float):
                    if calc_name.endswith('_rate') or calc_name.endswith('_percentage'):
                        formatted_value = f"{value:.2f}%"
                    elif value > 1000:
                        formatted_value = f"{value:,.2f}"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                    
                data_string += f"- Result: {formatted_value}\n"
                data_string += f"- Description: {calc_info.get('description', '')}\n"
                
                # Add explanation with more detail
                explanation = calc_info.get('explanation', '')
                if explanation:
                    data_string += f"- Calculation: {explanation}\n"
                
                # Add warnings if present
                warning = calc_info.get('warning', '')
                if warning:
                    data_string += f"- IMPORTANT NOTE: {warning}\n"
        
        # Create the messages for the chat completion
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user", 
                "content": (
                    f"Original query: {query}\n\n"
                    f"Available data:{data_string}\n\n"
                    "Please provide a conversational, easy-to-understand answer to the query based on this data. "
                    "Remember to highlight any important caveats about the data (like active users potentially being "
                    "counted multiple times). End with a brief insight or follow-up question."
                )
            }
        ]
        
        # Call the GenAI API
        response = client.chat.completions.create(
            model="n/a",  # Model is determined by the Digital Ocean GenAI Platform
            messages=messages,
            temperature=0.3,  # Lower temperature for more factual responses
        )
        
        # Extract the content from the response
        return response.choices[0].message.content
    
    def _perform_calculations(self, query: str, data_sources: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Perform calculations based on the query and data sources.
        
        Args:
            query: The user query
            data_sources: Dictionary with data from different sources
            
        Returns:
            Dictionary of calculation results
        """
        results = {}
        query_lower = query.lower()
        
        # Handle active users calculation
        if "active user" in query_lower or "active users" in query_lower:
            try:
                # Check if we have GA data with active users column
                ga_data = None
                if "Google Analytics" in data_sources:
                    ga_data = data_sources["Google Analytics"]
                
                if isinstance(ga_data, pd.DataFrame) and 'active_users' in ga_data.columns:
                    # Calculate sum of daily active users
                    total_active_users = ga_data['active_users'].sum()
                    
                    # Create daily breakdown
                    daily_breakdown = {}
                    date_col = 'date' if 'date' in ga_data.columns else None
                    
                    if date_col:
                        for _, row in ga_data.iterrows():
                            date_str = str(row[date_col])
                            daily_breakdown[date_str] = row['active_users']
                    else:
                        # Use row index if no date column
                        for i, row in ga_data.iterrows():
                            daily_breakdown[f"day_{i+1}"] = row['active_users']
                    
                    # Find max and min
                    max_day = max(daily_breakdown.items(), key=lambda x: x[1])
                    min_day = min(daily_breakdown.items(), key=lambda x: x[1])
                    
                    # Format the explanation
                    daily_list = [f"{day}: {count}" for day, count in daily_breakdown.items()]
                    daily_sum = " + ".join([str(count) for day, count in daily_breakdown.items()])
                    
                    explanation = (
                        f"Daily active users: {', '.join(daily_list)}\n"
                        f"Sum calculation: {daily_sum} = {total_active_users}\n"
                        f"Highest day: {max_day[0]} with {max_day[1]} active users\n"
                        f"Lowest day: {min_day[0]} with {min_day[1]} active users"
                    )
                    
                    # Store the result with proper context
                    results["active_users"] = {
                        "value": total_active_users,
                        "description": "Sum of daily active user sessions",
                        "explanation": explanation,
                        "warning": (
                            "This is a sum of daily active users, which may count the same user "
                            "multiple times if they visited on different days. This is not a count "
                            "of unique users."
                        ),
                        "daily_breakdown": daily_breakdown,
                        "max_day": max_day,
                        "min_day": min_day
                    }
                    
                    # Try to calculate unique users if we have the right columns
                    if 'user_id' in ga_data.columns:
                        unique_users = ga_data['user_id'].nunique()
                        results["unique_active_users"] = {
                            "value": unique_users,
                            "description": "Count of unique active users",
                            "explanation": f"Counted {unique_users} unique user IDs across the time period",
                        }
                    elif 'client_id' in ga_data.columns:
                        unique_users = ga_data['client_id'].nunique()
                        results["unique_active_users"] = {
                            "value": unique_users,
                            "description": "Count of unique active users",
                            "explanation": f"Counted {unique_users} unique client IDs across the time period",
                        }
            except Exception as e:
                print(f"Error calculating active users: {e}")
        
        # 1. Conversion rate calculation
        if "conversion rate" in query_lower or "convert" in query_lower:
            try:
                # Try to find sessions and conversions in the data
                if "Google Analytics" in data_sources:
                    ga_data = data_sources["Google Analytics"]
                    if isinstance(ga_data, pd.DataFrame):
                        # DataFrame - get totals
                        transactions = 0
                        sessions = 0
                        
                        if 'transactions' in ga_data.columns:
                            transactions = ga_data['transactions'].sum()
                        
                        if 'sessions' in ga_data.columns:
                            sessions = ga_data['sessions'].sum()
                            
                        if sessions > 0:
                            conversion_rate = (transactions / sessions) * 100
                            results["conversion_rate"] = {
                                "value": conversion_rate,
                                "description": "Conversion rate (transactions/sessions)",
                                "explanation": f"Conversion rate: {transactions} transactions / {sessions} sessions = {conversion_rate:.2f}%"
                            }
            except Exception as e:
                print(f"Error calculating conversion rate: {e}")
        
        # 2. Revenue metrics
        if "revenue" in query_lower or "sales" in query_lower:
            try:
                if "Stripe" in data_sources:
                    stripe_data = data_sources["Stripe"]
                    if isinstance(stripe_data, dict) and "data" in stripe_data:
                        # Extract total revenue
                        total_revenue = 0
                        for charge in stripe_data.get("data", []):
                            if isinstance(charge, dict) and "amount" in charge:
                                total_revenue += charge["amount"]
                        
                        results["total_revenue"] = {
                            "value": total_revenue,
                            "description": "Total revenue from Stripe transactions",
                            "explanation": f"Total revenue: Sum of {len(stripe_data.get('data', []))} transactions = {total_revenue}"
                        }
                        
                        # Calculate average order value if transaction count > 0
                        if len(stripe_data.get("data", [])) > 0:
                            aov = total_revenue / len(stripe_data.get("data", []))
                            results["average_order_value"] = {
                                "value": aov,
                                "description": "Average Order Value",
                                "explanation": f"Average Order Value: {total_revenue} / {len(stripe_data.get('data', []))} transactions = {aov:.2f}"
                            }
            except Exception as e:
                print(f"Error calculating revenue metrics: {e}")
        
        # 3. Percentage changes
        if "change" in query_lower or "growth" in query_lower or "increase" in query_lower or "decrease" in query_lower:
            try:
                # Look for metrics with 'current' and 'previous' fields
                for source_name, source_data in data_sources.items():
                    if isinstance(source_data, dict) and "data" in source_data:
                        for metric_name, metric_data in source_data.get("data", {}).items():
                            if isinstance(metric_data, dict) and "current" in metric_data and "previous" in metric_data:
                                current = metric_data["current"]
                                previous = metric_data["previous"]
                                
                                if previous != 0:
                                    pct_change = ((current - previous) / previous) * 100
                                    results[f"{metric_name}_pct_change"] = {
                                        "value": pct_change,
                                        "description": f"Percentage change in {metric_name}",
                                        "explanation": f"Percentage change in {metric_name}: ({current} - {previous}) / {previous} * 100 = {pct_change:.2f}%"
                                    }
                                else:
                                    if current > 0:
                                        results[f"{metric_name}_pct_change"] = {
                                            "value": float('inf'),
                                            "description": f"Percentage change in {metric_name} (new activity)",
                                            "explanation": f"New activity: {metric_name} went from 0 to {current}, representing new activity."
                                        }
            except Exception as e:
                print(f"Error calculating percentage changes: {e}")
        
        return results