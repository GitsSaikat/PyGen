# Ollama Installation

 # curl https://ollama.ai/install.sh | sh
 # ollama serve & ollama pull mistral
 # ollama run llama3.1:70b llama3.1:8b llama3.2:latest

# %pip install -U langchain-ollama

import os
import re
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

def parse_generated_content(content: str) -> Dict[str, str]:
    """
    Parse the generated content into a dictionary of file paths and contents.
    Expects the format:
    <file_path>
    <begin_content>
    file content here
    <end_content>
    """
    files = {}
    current_file = None
    current_content = []
    state = 'expect_file'

    lines = content.strip().split('\n')
    line_num = 0  # Initialize line number
    while line_num < len(lines):
        line = lines[line_num]
        stripped = line.strip()
        line_num += 1

        if state == 'expect_file':
            if stripped.startswith('<') and stripped.endswith('>'):
                current_file = stripped[1:-1].strip()
                if not current_file:
                    print(f"Warning: Empty file path at line {line_num}")
                    current_file = None
                    continue
                state = 'expect_begin_content'
            elif stripped == '':
                continue  # Skip empty lines
            else:
                # Ignore any text before the first <file_path>
                continue
        elif state == 'expect_begin_content':
            if stripped == '<begin_content>':
                current_content = []
                state = 'collect_content'
            else:
                print(f"Warning: Expected <begin_content> after <{current_file}> at line {line_num}, got: {stripped}")
                current_file = None
                state = 'expect_file'
        elif state == 'collect_content':
            if stripped == '<end_content>':
                if current_file in files:
                    print(f"Warning: Duplicate file path '{current_file}' at line {line_num}. Overwriting previous content.")
                files[current_file] = '\n'.join(current_content).strip()
                current_file = None
                current_content = []
                state = 'expect_file'
            else:
                current_content.append(line)
    # Check if any file was not properly closed
    if state == 'collect_content' and current_file:
        print(f"Warning: Incomplete file definition for {current_file}")
        files[current_file] = '\n'.join(current_content).strip()  # Save what was collected
    return files

def generate_content_with_ollama(prompt: str, model_name: str) -> str:
    """Generate content using the Ollama model via LangChain."""
    template = """{prompt}"""
    chat_prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model=model_name)
    chain = chat_prompt | model
    response = chain.invoke({"prompt": prompt})
    return response

def generate_python_package(package_name: str, package_description: str, features: List[str], model_name: str) -> Optional[Dict[str, str]]:
    """Generate a Python package structure and content using the Ollama model."""

    # Format the features as a bulleted list
    if features:
        features_formatted = "\n".join(f"- {feature.strip()}" for feature in features)
        features_section = f"\n\nInclude the following features:\n{features_formatted}"
    else:
        features_section = ""

    # Updated prompt with clearer instructions
    prompt = f"""
You are to generate the structure and content of a Python package named `{package_name}` based on the following description:
{package_description}{features_section}

Provide the package structure and content **strictly** in the following format for each file:

<file_path>
<begin_content>
file_content
<end_content>

Include **only** the files specified below, in the given order:

- setup.py
- README.md
- requirements.txt
- examples/example_{package_name}.py
- tests/test_{package_name}.py
- {package_name}/__init__.py
- {package_name}/main.py

Ensure there are no extra lines or spaces between the tags and the content.

**Important Instructions:**

- Do **not** include any text outside the specified format.
- Do **not** provide any explanations, introductions, or conclusions.
- Ensure that every `<begin_content>` has a matching `<end_content>`.
- Use the exact file paths and filenames as specified.
- The content inside `<begin_content>` and `<end_content>` should be the code or text for that file.

**Example:**

<setup.py>
<begin_content>
from setuptools import setup, find_packages

setup(
    name='{package_name}',
    version='0.1.0',
    description='An example package',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        # Add dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
<end_content>

<README.md>
<begin_content>
# {package_name}

An example package for demonstration purposes.
<end_content>

Please begin the response now.
"""

    try:
        response_text = generate_content_with_ollama(prompt, model_name)
        print("=== Raw AI Response ===")
        print(response_text)
        print("========================\n")
        parsed_content = parse_generated_content(response_text)

        if not parsed_content:
            print("Warning: The model's response could not be parsed into a valid package structure.")
            return None

        return parsed_content
    except Exception as e:
        print(f"Error generating package content: {str(e)}")
        return None

def create_package_files(package_structure: Dict[str, str]) -> None:
    """Create the package files and directories based on the generated structure."""
    for file_path, content in package_structure.items():
        if not file_path or not content:
            print(f"Warning: Skipping file with empty path or content")
            continue
        try:
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Created: {file_path}")
        except Exception as e:
            print(f"Error creating {file_path}: {str(e)}")
    print(f"\nPackage file creation process completed.")

def get_user_input() -> tuple:
    """Get user input for package name, description, and features."""
    package_name = input("Enter the name for your Python package: ").strip()
    while not package_name:
        print("Package name cannot be empty. Please try again.")
        package_name = input("Enter the name for your Python package: ").strip()

    package_description = input("Enter a brief description of your package: ").strip()
    while not package_description:
        print("Package description cannot be empty. Please try again.")
        package_description = input("Enter a brief description of your package: ").strip()

    features = []
    print("Enter the features you want to include in your package.")
    print("Type each feature and press Enter. When done, just press Enter on an empty line.")
    while True:
        feature = input("Enter a feature (or press Enter to finish): ").strip()
        if not feature:
            break
        features.append(feature)

    return package_name, package_description, features

def create_fallback_structure(package_name: str) -> Dict[str, str]:
    """Create a fallback package structure if the model fails to generate one."""
    return {
        f"{package_name}/__init__.py": "# Package initialization\n",
        f"{package_name}/main.py": f"# Main module for {package_name}\n",
        "setup.py": f"""from setuptools import setup, find_packages

setup(
    name='{package_name}',
    version='0.1.0',
    description='A brief description of your package',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
""",
        "README.md": f"# {package_name}\n\nA brief description of the package and its functionality.\n",
        "requirements.txt": "# Add your package dependencies here\n",
        f"tests/test_{package_name}.py": f"""import unittest
from {package_name}.main import {package_name.capitalize()}Class

class Test{package_name.capitalize()}Class(unittest.TestCase):
    def setUp(self):
        self.instance = {package_name.capitalize()}Class()

    def test_example(self):
        self.assertTrue(True)  # Replace with actual tests

if __name__ == '__main__':
    unittest.main()
"""
    }

def validate_package_structure(package_structure: Dict[str, str]) -> bool:
    """Validate that the package structure has non-empty file paths and contents."""
    valid = True
    for file_path, content in package_structure.items():
        if not file_path:
            print(f"Error: Found empty file path.")
            valid = False
        if not content:
            print(f"Error: File '{file_path}' has empty content.")
            valid = False
    return valid

def generate_markdown_documentation(package_structure: Dict[str, str], package_name: str, features: List[str]) -> Optional[str]:
    """
    Generate a Markdown documentation file (DOCUMENTATION.md) based on the generated package structure.
    """
    documentation_lines = []

    # Package Overview
    readme_content = package_structure.get("README.md", "")
    if readme_content:
        # Extract the first paragraph as the description
        lines = readme_content.split('\n')
        description = ""
        for line in lines:
            if line.strip() and not line.startswith("#"):
                description = line.strip()
                break
        documentation_lines.append(f"# {package_name}\n")
        documentation_lines.append(f"{description}\n")
    else:
        documentation_lines.append(f"# {package_name}\n")
        documentation_lines.append("A brief description of the package.\n")

    # Features
    if features:
        documentation_lines.append("## Features\n")
        for feature in features:
            documentation_lines.append(f"- {feature}")
        documentation_lines.append("")  # Add an empty line

    # [Rest of the function remains unchanged]

    # Write the documentation to DOCUMENTATION.md
    documentation_content = "\n".join(documentation_lines)
    documentation_path = os.path.join(os.getcwd(), "DOCUMENTATION.md")

    try:
        with open(documentation_path, 'w', encoding='utf-8') as doc_file:
            doc_file.write(documentation_content)
        print(f"Documentation generated at {documentation_path}")
        return documentation_path
    except Exception as e:
        print(f"Error generating documentation: {str(e)}")
        return None

def main():
    package_name, package_description, features = get_user_input()

    # Prompt the user for the model name
    print("\nPlease enter the model name you wish to use.")
    print("Example model names: 'llama3.2:latest', etc.")
    model_name = input("Enter the model name to use: ").strip()
    while not model_name:
        print("Model name cannot be empty. Please try again.")
        model_name = input("Enter the model name to use: ").strip()

    print("\nGenerating Python Package...")
    package_structure = generate_python_package(package_name, package_description, features, model_name)

    if not package_structure or not validate_package_structure(package_structure):
        print("Failed to generate valid package structure. Using fallback structure.")
        package_structure = create_fallback_structure(package_name)

    print("\nGenerated Python Package Structure:")
    for file_path in package_structure.keys():
        print(f"- {file_path}")

    create = input("\nDo you want to create these files? (y/n): ").strip().lower()
    if create == 'y':
        create_package_files(package_structure)
    else:
        print("Package files were not created.")

    # Generate Documentation
    generate_doc = input("\nDo you want to generate documentation? (y/n): ").strip().lower()
    if generate_doc == 'y':
        documentation_path = generate_markdown_documentation(package_structure, package_name, features)
        if documentation_path:
            print(f"Documentation successfully created at {documentation_path}")
        else:
            print("Failed to create documentation.")
    else:
        print("Documentation was not created.")

if __name__ == "__main__":
    main()
