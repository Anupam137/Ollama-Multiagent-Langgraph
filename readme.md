# Ollama Multi-Agent System Langraph Supervisor

## Overview
Langraph Supervisor is an advanced framework designed to manage and supervise a team of agents using Ollama models. This project represents a significant improvement in the way multi-agent systems are deployed and managed, providing enhanced capabilities and responsibilities for the supervising agent.

**Original Repository:** [Langraph Supervisor](https://github.com/langchain-ai/langgraph-supervisor)

## Features(in this build):
- **Ollama Models Integration**: Utilize Ollama models for running agents locally, offering a robust alternative to OpenAI models.
- **Enhanced Multi-Agent System**: The `enhanced_multiagent.py` file introduces a larger set of agents with expanded responsibilities for the supervisor, allowing for more complex interactions and tasks.
- **Tavily Integration (In Development)**: This feature aims to improve search result accuracy by integrating Tavily into the agent team.
- **Scalable Agent Framework**: Designed to allow easy extension and customization of agent functionalities.

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Ollama installed and configured on your local machine.

### Running the Agents
To run the team of agents locally using Ollama models, follow these steps:

1. **Clone the Repository**: Ensure you have the latest version of Langraph Supervisor.
   ```bash
   git clone https://github.com/langchain-ai/langgraph-supervisor.git
   cd langgraph-supervisor
   ```

2. **Install Dependencies**: Install all required packages. It is recommended to create a virtual environment before proceeding.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Agents**: Execute the `ollama-multiagent.py` script to start the agents.
   ```bash
   python ollama-multiagent.py
   ```

4. **Testing Enhanced Multi-Agent System**: To test the enhanced system, run:
   ```bash
   python run_enhanced.py
   ```

## Future Work
- Full integration of Tavily for improved search accuracy.
- Enhancements to agent coordination and supervisor decision-making capabilities.
- Expansion of supported local models for increased flexibility.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for discussion.


