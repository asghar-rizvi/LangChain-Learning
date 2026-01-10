from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    
# new_person : Person = {"name": "Asghar", "age":21} # Works perfect
new_person : Person = {"name":"Asghar", "age":"21"} # it also works perfect because it doesnt give any error, cos it does no validation
print(new_person)