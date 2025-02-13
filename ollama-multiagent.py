from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

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
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )

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
    name="research_agent",
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

# Example usage
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

# Test the system
if __name__ == "__main__":
    # Example queries
    queries = [
        "what's the combined headcount of the FAANG companies in 2024?",
        "what's 25 times 16?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = process_query(query)
        print(f"Result: {result}")