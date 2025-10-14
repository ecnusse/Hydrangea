
import json
import openai

def process_user_query(query):
    """Main application function"""
    # Call LLM
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}]
    )
    
    llm_output = response.choices[0].message.content
    
    # Process the output
    if query.startswith("json"):
        return process_json_response(llm_output)
    elif query.startswith("code"):
        return execute_code(llm_output)
    else:
        return process_text(llm_output)

def process_json_response(json_text):
    """Process JSON output from LLM"""
    try:
        data = json.loads(json_text)
        return data.get("result", "No result")
    except json.JSONDecodeError:
        return "Invalid JSON"

def execute_code(code_text):
    """Execute Python code from LLM"""
    try:
        compile(code_text, "<string>", "exec")
        exec(code_text)
        return "Code executed successfully"
    except SyntaxError:
        return "Syntax error in code"

def process_text(text):
    """Process text output"""
    sentences = text.split(".")
    return [s.strip() for s in sentences if s.strip()]
