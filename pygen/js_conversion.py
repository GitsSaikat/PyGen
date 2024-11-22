import os
from groq import Groq

def generate_content_with_groq(client: Groq, model_name: str, prompt: str) -> str:
    """Generate content using the specified model via Groq."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating content: {str(e)}")
        return ""

def translate_python_to_javascript(python_code: str, client: Groq, model_name: str) -> str:
    """Translate Python code to JavaScript using Groq model."""
    prompt = f"""
Translate the following Python code to JavaScript:

{python_code}
"""
    try:
        response_text = generate_content_with_groq(client, model_name, prompt)
        return response_text
    except Exception as e:
        print(f"Error translating Python code: {str(e)}")
        return ""

def main():
    # Get user input for Python code
    python_code = input("Enter the Python code to translate to JavaScript: ").strip()
    while not python_code:
        print("Python code cannot be empty. Please try again.")
        python_code = input("Enter the Python code to translate to JavaScript: ").strip()

    # Set the environment variable or get the API key from the user
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = input("Please enter your Groq API key: ").strip()
        os.environ["GROQ_API_KEY"] = api_key

    # Initialize Groq client
    client = Groq(api_key=api_key)

    # Prompt the user for the model name
    print("\nPlease enter the model name you wish to use.")
    print("Example model names: 'llama3-8b-8192', 'llama3-13b-8192', etc.")
    model_name = input("Enter the model name to use: ").strip()
    while not model_name:
        print("Model name cannot be empty. Please try again.")
        model_name = input("Enter the model name to use: ").strip()

    print("\nTranslating Python code to JavaScript...")
    javascript_code = translate_python_to_javascript(python_code, client, model_name)

    if not javascript_code:
        print("Failed to translate the Python code. Exiting.")
        return

    print("\nTranslated JavaScript Code:")
    print(javascript_code)

if __name__ == "__main__":
    main()
