from typing import Dict, Any, Union, List
import re
import pandas as pd

class CalculationHelper:
    """
    Helper class to perform financial calculations with proper validation
    to be used by the SynthesizerAgent.
    """
    
    def __init__(self, data_context: Dict[str, Any] = None):
        """
        Initialize the calculator with an optional data context.
        
        Args:
            data_context: A dictionary containing metric values
        """
        self.data_context = data_context or {}
        self.original_data = {}
        
        # Define supported mathematical functions
        self.functions = {
            'avg': self._avg,
            'sum': self._sum,
            'max': self._max,
            'min': self._min,
            'percentage_change': self._percentage_change,
            'conversion_rate': self._conversion_rate,
            'unique_users': self._unique_users,
            'total_sessions': self._total_sessions
        }
    
    def update_context(self, data: Dict[str, Any]):
        """
        Update the data context with new data.
        
        Args:
            data: New data to add to the context
        """
        # Store original data for specialized analytics
        self.original_data = data
        
        # Extract metrics from complex data structures
        flat_metrics = self._flatten_data(data)
        self.data_context.update(flat_metrics)
    
    def _flatten_data(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Union[int, float]]:
        """
        Flatten nested data structures into a simple key-value dictionary.
        
        Args:
            data: Nested data structure
            prefix: Prefix for keys (used in recursion)
            
        Returns:
            Flattened dictionary
        """
        result = {}
        
        # Handle GA DataFrame
        if hasattr(data, 'to_dict'):
            # Convert DataFrame to dict
            df_dict = data.to_dict('records')
            for i, row in enumerate(df_dict):
                for col, val in row.items():
                    key = f"ga:{col}:{i}" if prefix == "" else f"{prefix}:{col}:{i}"
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        result[key] = val
            return result
            
        # Regular dictionary
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = key if prefix == "" else f"{prefix}:{key}"
                
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    result[new_key] = value
                elif isinstance(value, dict):
                    result.update(self._flatten_data(value, new_key))
                elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    for i, item in enumerate(value):
                        item_key = f"{new_key}:{i}"
                        result.update(self._flatten_data(item, item_key))
        
        return result
    
    def evaluate(self, expression: str) -> Union[int, float]:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: Expression to evaluate
            
        Returns:
            Calculated result
        """
        # Special case for active users
        if "active_users" in expression.lower() and "sum" in expression.lower():
            return self._handle_active_users()
            
        return self.evaluate_expression(expression)
    
    def _handle_active_users(self) -> Dict[str, Any]:
        """
        Special handler for active users calculation.
        
        Returns:
            Dictionary with calculation result and metadata
        """
        # Check if we have GA data with active users
        if "Google Analytics" in self.original_data:
            ga_data = self.original_data["Google Analytics"]
            
            if isinstance(ga_data, pd.DataFrame) and 'active_users' in ga_data.columns:
                # Get active users by day
                daily_active = {}
                date_col = None
                
                # Try to find date column
                for col in ga_data.columns:
                    if 'date' in col.lower():
                        date_col = col
                        break
                
                # Create a dictionary of active users by day
                if date_col:
                    for _, row in ga_data.iterrows():
                        daily_active[str(row[date_col])] = row['active_users']
                else:
                    # If no date column, use row index
                    for i, row in ga_data.iterrows():
                        daily_active[f"day_{i}"] = row['active_users']
                
                # Calculate total (sum of daily active users)
                total_active = sum(daily_active.values())
                
                return {
                    "result": total_active,
                    "daily_breakdown": daily_active,
                    "metric_type": "active_users",
                    "calculation_type": "sum_of_daily",
                    "warning": "This is a sum of daily active users, which may count some users multiple times if they were active on different days."
                }
        
        # Fallback to regular calculation
        return None
    
    def _resolve_metric(self, metric: str) -> Union[int, float]:
        """
        Resolve a metric from the data context.
        
        Args:
            metric: Metric identifier
            
        Returns:
            Metric value
        """
        if metric in self.data_context:
            return self.data_context[metric]
        
        # Try to convert to float/int
        try:
            return float(metric)
        except ValueError:
            raise ValueError(f"Metric '{metric}' not found in data context")
    
    def evaluate_expression(self, expression: str) -> Union[int, float]:
        """
        Evaluate a mathematical expression with support for metrics and functions.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Calculated result
        """
        # First, try to parse as a direct metric
        if re.match(r'^[A-Za-z0-9:_]+$', expression):
            return self._resolve_metric(expression)
        
        # Try to parse as a function call
        func_match = re.match(r'^(\w+)\((.*)\)$', expression)
        if func_match:
            function = func_match.group(1)
            args_str = func_match.group(2)
            
            # Split arguments, handling potential nested structures
            args = [arg.strip() for arg in args_str.split(',')]
            
            # Resolve each argument (either a metric or a constant)
            resolved_args = []
            for arg in args:
                if re.match(r'^[A-Za-z0-9:_]+$', arg):
                    resolved_args.append(self._resolve_metric(arg))
                else:
                    try:
                        resolved_args.append(float(arg))
                    except ValueError:
                        raise ValueError(f"Cannot resolve argument: {arg}")
            
            # Call the function
            if function not in self.functions:
                raise ValueError(f"Unsupported function: {function}")
            
            return self.functions[function](*resolved_args)
        
        # Try to parse as a numeric constant
        try:
            return float(expression)
        except ValueError:
            raise ValueError(f"Unable to parse expression: {expression}")
    
    def explain_calculation(self, expression: str, result: Union[int, float]) -> str:
        """
        Generate an explanation for a calculation.
        
        Args:
            expression: Expression that was evaluated
            result: Result of the calculation
            
        Returns:
            Human-readable explanation
        """
        # Special case for active users
        if "active_users" in expression.lower() and isinstance(result, dict):
            daily_breakdown = result.get("daily_breakdown", {})
            total = result.get("result", 0)
            
            # Create explanation
            explanation = "Active users by day:\n"
            for day, count in daily_breakdown.items():
                explanation += f"- {day}: {count}\n"
            
            explanation += f"\nTotal active users (sum of daily): {total}"
            
            if "warning" in result:
                explanation += f"\n\nNote: {result['warning']}"
                
            return explanation
        
        # Try to parse as a function call
        func_match = re.match(r'^(\w+)\((.*)\)$', expression)
        if func_match:
            function = func_match.group(1)
            args_str = func_match.group(2)
            
            # Split arguments
            args = [arg.strip() for arg in args_str.split(',')]
            
            # Resolve arguments for explanation
            resolved_args = []
            for arg in args:
                if re.match(r'^[A-Za-z0-9:_]+$', arg):
                    try:
                        value = self._resolve_metric(arg)
                        resolved_args.append(f"{arg} ({value})")
                    except ValueError:
                        resolved_args.append(arg)
                else:
                    resolved_args.append(arg)
            
            # Generate explanation based on function
            if function == 'avg':
                return f"Average of {', '.join(resolved_args)} = {result}"
            elif function == 'sum':
                return f"Sum of {', '.join(resolved_args)} = {result}"
            elif function == 'max':
                return f"Maximum of {', '.join(resolved_args)} = {result}"
            elif function == 'min':
                return f"Minimum of {', '.join(resolved_args)} = {result}"
            elif function == 'percentage_change':
                return f"Percentage change from {resolved_args[1]} to {resolved_args[0]} = {result:.2f}%"
            elif function == 'conversion_rate':
                return f"Conversion rate: {resolved_args[0]} / {resolved_args[1]} = {result:.4f}"
            elif function == 'unique_users':
                return f"Unique users across the date range = {result}"
            else:
                return f"{function}({', '.join(resolved_args)}) = {result}"
        
        # Direct metric or constant
        return f"Value of {expression} = {result}"
    
    # Helper functions for mathematical operations
    def _avg(self, *args):
        """Calculate average of given arguments"""
        if not args:
            raise ValueError("No arguments provided")
        return sum(args) / len(args)
    
    def _sum(self, *args):
        """Calculate sum of given arguments"""
        return sum(args)
    
    def _max(self, *args):
        """Find maximum of given arguments"""
        if not args:
            raise ValueError("No arguments provided")
        return max(args)
    
    def _min(self, *args):
        """Find minimum of given arguments"""
        if not args:
            raise ValueError("No arguments provided")
        return min(args)
    
    def _percentage_change(self, new_value, old_value):
        """Calculate percentage change between two values"""
        if old_value == 0:
            if new_value == 0:
                return 0
            return float('inf')  # Handle division by zero
        return ((new_value - old_value) / abs(old_value)) * 100
    
    def _conversion_rate(self, conversions, sessions):
        """Calculate conversion rate"""
        if sessions == 0:
            return 0
        return (conversions / sessions)
        
    def _unique_users(self, *args):
        """Calculate unique users from GA data"""
        # Implementation would depend on your data structure
        # This is a placeholder
        if "Google Analytics" in self.original_data:
            ga_data = self.original_data["Google Analytics"]
            
            if isinstance(ga_data, pd.DataFrame):
                # If you have user_id column, count unique values
                if 'user_id' in ga_data.columns:
                    return ga_data['user_id'].nunique()
                    
                # If you have client_id column, count unique values
                if 'client_id' in ga_data.columns:
                    return ga_data['client_id'].nunique()
                    
        return 0
        
    def _total_sessions(self, *args):
        """Calculate total sessions from GA data"""
        if "Google Analytics" in self.original_data:
            ga_data = self.original_data["Google Analytics"]
            
            if isinstance(ga_data, pd.DataFrame) and 'sessions' in ga_data.columns:
                return ga_data['sessions'].sum()
                
        return 0