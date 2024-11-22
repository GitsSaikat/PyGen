import os
from typing import Dict
from groq import Groq

def load_package_files(package_dir: str) -> Dict[str, str]:
    """
    Recursively load all relevant files in the package directory into a dictionary.

    Args:
        package_dir (str): Path to the root of the Python package.

    Returns:
        Dict[str, str]: A dictionary mapping relative file paths to their contents.
    """
    package_files = {}
    for root, _, files in os.walk(package_dir):
        for file in files:
            if file.endswith(('.py', '.md', '.txt')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Store relative paths for better context
                    relative_path = os.path.relpath(file_path, package_dir)
                    package_files[relative_path] = content
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return package_files

def generate_review_with_groq(client: Groq, model_name: str, package_files: Dict[str, str], package_name: str) -> str:
    """
    Generate a 100-word review of the Python package using the Groq model.

    Args:
        client (Groq): Initialized Groq client.
        model_name (str): Name of the Groq model to use.
        package_files (Dict[str, str]): Dictionary of file paths and their contents.
        package_name (str): Name of the Python package.

    Returns:
        str: The generated 100-word review.
    """
    # Concatenate all file contents with file paths for context
    concatenated_content = ""
    for path, content in package_files.items():
        concatenated_content += f"### {path}\n{content}\n\n"

    # Define the prompt for the review
    prompt = f"""
You are a professional code reviewer. Analyze the following Python package named '{package_name}' and provide a concise 100-word review highlighting its structure, code quality, documentation, and any improvements needed. Score it out of 10 based on package structure, code quality, coherence.

Here is the package content:

{concatenated_content}

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
            max_tokens=250,  # Adjust to ensure approximately 100 words
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
    # Get user input for the package directory
    package_dir = input("Enter the path to your generated Python package: ").strip()
    if not os.path.isdir(package_dir):
        print("Invalid package directory.")
        return

    # Load package files
    package_files = load_package_files(package_dir)
    if not package_files:
        print("No valid files found in the package.")
        return

    # Get package name (assuming the package directory name is the package name)
    package_name = os.path.basename(os.path.abspath(package_dir))

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
    print("\nGenerating 100-word review...")
    review = generate_review_with_groq(client, model_name, package_files, package_name)

    print("\n=== Package Review ===")
    print(review)
    print("======================")

if __name__ == "__main__":
    main()
