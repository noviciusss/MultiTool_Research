import math
import statistics
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result as a string.
    supports -basic maths - sqrt,cos,tan,tan,sin,log 
    statistics - mean, median, mode, stdev
    
    args : expression : str : mathematical expression in str(eg "sqrt(144)")
    return : str:result or error message
    
    """
    try :
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "mean": statistics.mean,
            "median": statistics.median,
            "stdev": statistics.stdev,
        }
        result = eval(expression, {"__builtins__": None}, safe_dict)
        return str(result)
        
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"
    
if __name__ == "__main__":
    print("Testing Calculator Tool:")
    print(f"2 + 2 = {calculator.invoke('2 + 2')}")
    print(f"sqrt(144) = {calculator.invoke('sqrt(144)')}")
    print(f"mean([10, 20, 30]) = {calculator.invoke('mean([10, 20, 30])')}")