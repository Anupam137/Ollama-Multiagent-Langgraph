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

def draft_email(subject: str, content: str) -> str:
    """Draft an email with given subject and content."""
    return f"""
Subject: {subject}
---
{content}
---
Best regards,
[Your Name]
    """

def code_review(code: str) -> str:
    """Review code and provide suggestions."""
    return f"Code Review Results:\n{code}\n[Analysis and suggestions would be generated here]"

def task_planning(task: str) -> str:
    """Create a task plan with steps and timeline."""
    return f"Task Plan for: {task}\n[Detailed plan would be generated here]"

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

utility_agent = create_react_agent(
    model=base_model,
    tools=[draft_email, code_review, task_planning],
    name="utility_expert",
    prompt=(
        "You are an expert in handling day-to-day tasks like email drafting, "
        "code review, and task planning. Break down each task systematically "
        "and provide detailed, actionable outputs."
    )
)

# Create enhanced supervisor that acts as both supervisor and manager
workflow = create_supervisor(
    [research_agent, math_agent, utility_agent],
    model=base_model,
    prompt=(
        "You are a senior manager and supervisor overseeing a team of experts:\n"
        "1. Research expert (research_agent) for information gathering and analysis\n"
        "2. Math expert (math_expert) for calculations and numerical analysis\n"
        "3. Utility expert (utility_expert) for day-to-day tasks including:\n"
        "   - Email drafting\n"
        "   - Code review\n"
        "   - Task planning\n\n"
        "Your responsibilities:\n"
        "1. Analyze incoming queries and route them to the appropriate expert(s)\n"
        "2. For complex queries requiring multiple experts, coordinate their efforts\n"
        "3. Ensure high-quality output by reviewing expert responses\n"
        "4. Provide strategic guidance when needed\n"
        "5. Handle task prioritization and resource allocation\n\n"
        "Think strategically about each query and use the most appropriate expert(s).\n"
        "For math questions: Use math_expert\n"
        "For research/information: Use research_agent\n"
        "For practical tasks: Use utility_expert\n"
        "For complex queries: Coordinate multiple experts as needed"
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

# Make the system available for import
system = {
    "app": app,
    "process_query": process_query,
    "math_agent": math_agent,
    "research_agent": research_agent,
    "utility_agent": utility_agent
}