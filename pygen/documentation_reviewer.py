import os
from groq import Groq

def load_single_documentation_file(doc_file_path: str) -> str:
    """
    Load the content of a single documentation file.

    Args:
        doc_file_path (str): Path to the documentation file.

    Returns:
        str: The content of the documentation file.
    """
    try:
        with open(doc_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading {doc_file_path}: {e}")
        return ""

def generate_documentation_review(client: Groq, model_name: str, doc_content: str, project_name: str) -> str:
    """
    Generate a review of the documentation using the Groq model.

    Args:
        client (Groq): Initialized Groq client.
        model_name (str): Name of the Groq model to use.
        doc_content (str): Content of the documentation.
        project_name (str): Name of the project.

    Returns:
        str: The generated review.
    """
    prompt = f"""
You are a professional documentation reviewer. Analyze the following documentation for the project '{project_name}' and provide a concise review highlighting its clarity, completeness, structure, Readability and any improvements needed. Score it out of 10 based on these criteria.

Here is the documentation content:

{doc_content}

Review:
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
            max_tokens=300,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        review = chat_completion.choices[0].message.content.strip()
        return review
    except Exception as e:
        print(f"Error generating review: {e}")
        return "Failed to generate review."

def main():
    # Path to the documentation file
    doc_file_path = "/content/DOCUMENTATION.md"
    if not os.path.isfile(doc_file_path):
        print(f"Documentation file not found at {doc_file_path}")
        return

    # Load documentation content
    doc_content = load_single_documentation_file(doc_file_path)
    if not doc_content:
        print("Failed to load documentation content.")
        return

   
    project_name = "AutoML"  

    # Get Groq API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = input("Please enter your Groq API key: ").strip()
        if not api_key:
            print("Groq API key is required.")
            return
        os.environ["GROQ_API_KEY"] = api_key

    # Initialize Groq client
    client = Groq(api_key=api_key)

    # Prompt the user for the model name
    print("\nPlease enter the Groq model name you wish to use.")
    print("Example model names: 'llama3-8b-8192', 'llama3-13b-8192', etc.")
    model_name = input("Enter the Groq model name to use: ").strip()
    while not model_name:
        print("Model name cannot be empty. Please try again.")
        model_name = input("Enter the Groq model name to use: ").strip()

    # Generate the review
    print("\nGenerating documentation review...")
    review = generate_documentation_review(client, model_name, doc_content, project_name)

    print("\n=== Documentation Review ===")
    print(review)
    print("============================")

if __name__ == "__main__":
    main()
