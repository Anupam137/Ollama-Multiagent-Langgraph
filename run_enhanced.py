from enhanced_multiagent import system

result = system["process_query"]("What is the code logic behind fibonacci?")

# Ensure result is a dictionary and contains 'messages'
if isinstance(result, dict) and "messages" in result:
    for message in result["messages"]:
        role = getattr(message, "name", "user")  
        content = getattr(message, "content", "")

        print(f"\n🟢 Role: {role}")
        print(f"📜 Content: {content}")

        # tool calls will also be printed
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool in message.tool_calls:
                if isinstance(tool, dict):  
                    print(f"  🔧 Tool Call: {tool.get('name')} with Args: {tool.get('args')}")
                else:  
                    print(f"  🔧 Tool Call: {getattr(tool, 'name', 'Unknown')} with Args: {getattr(tool, 'args', {})}")

else:
    print("Unexpected response format:", result)
