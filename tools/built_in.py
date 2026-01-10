from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

# search = DuckDuckGoSearchRun()

# result = search.invoke("What is the temperature of karachi right now")

# print(result)

tool = ShellTool()
result = tool.invoke('whoami')
# result = tool.invoke('ls')
print(result)