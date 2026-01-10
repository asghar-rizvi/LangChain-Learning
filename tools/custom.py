from langchain_core.tools import tool

# Decorator way -> the easiest way

# @tool
# def multiply_number(a: int, b:int) -> int:
#     """Multiply 2 numbers"""
#     return a*b

# result = multiply_number.invoke({"a":3, "b":4})

# print(multiply_number.args_schema.model_json_schema())
# print(result)


# Structured Tool -> Just use pydantic model for stricter constraints

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyModel(BaseModel):
    a: int = Field(description="First number to add")
    b: int = Field(description="Second number to add")

def multiply_number(a: int, b:int) -> int:
    return a*b

multiply_tool = StructuredTool.from_function(
    func = multiply_number,
    name = "multiply",
    description = "Multiply 2 numbers",
    args_schema=MultiplyModel
)

result = multiply_tool.invoke({"a":2,"b":44})
print(result)