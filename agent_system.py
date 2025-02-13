from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_community.tools import TavilySearchResults
import os

# Initialize Tavily
search = TavilySearchResults(Tavily_API_KEY=os.getenv("TAVILY_API_KEY"))
#search_api = SearchApiAPIWrapper(search_api_api_key=os.getenv("SEARCHAPI_API_KEY"))



tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)


# Initialize Ollama model with step-by-step thinking
def create_ollama_chain(system_prompt: str):
    template = """System: {system}
Question: {question}
Answer: Let's think step by step."""
    
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOllama(model="llama3.1:8b")
    return prompt | model

# Define tools
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def web_search(query: str) -> str:
    """Search the web for information using Tavily API."""
    try:
        search_results = search.run(query)
        return search_results
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# Create base model
base_model = ChatOllama(model="llama3.1:8b")

# Create specialized agents
math_agent = create_react_agent(
    model=base_model,
    tools=[add, multiply],
    name="math_expert",
    prompt=(
        "You are a math expert. Always use one tool at a time. "
        "Think step by step when solving problems."
    )
)

research_agent = create_react_agent(
    model=base_model,
    tools=[web_search],
    name="research_expert",
    prompt=(
        "You are a world class researcher with access to web search. "
        "Do not do any math. Think step by step when analyzing information."
    )
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=base_model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent. "
        "Think step by step when deciding which agent to use."
    )
)

# Compile workflow
app = workflow.compile()

def process_query(query: str) -> Dict[str, Any]:
    """Process a query through the multi-agent system."""
    try:
        result = app.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        })
        return result
    except Exception as e:
        return {"error": f"Error processing query: {str(e)}"}

# Make the multi-agent system available for import
system = {
    "app": app,
    "process_query": process_query,
    "math_agent": math_agent,
    "research_agent": research_agent
}