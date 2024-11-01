# package_groq.py
import os
import re
from typing import Dict, List, Optional
from groq import Groq

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

    lines = content.split('\n')
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
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
                print(f"Warning: Expected <file_path> at line {line_num}, got: {stripped}")
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
    if state != 'expect_file':
        print(f"Warning: Incomplete file definition for {current_file}")
    return files

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

def generate_python_package(package_name: str, package_description: str, features: List[str], client: Groq, model_name: str) -> Optional[Dict[str, str]]:
    """Generate a Python package structure and content using the specified LLM via Groq."""

    # Format the features as a bulleted list
    if features:
        features_formatted = "\n".join(f"- {feature.strip()}" for feature in features)
        features_section = f"\n\nInclude the following features:\n{features_formatted}"
    else:
        features_section = ""

    prompt = f"""
Create a Python package named `{package_name}` based on the following description:
{package_description}{features_section}

Provide the package structure and content strictly in the following format for each file:

<file_path>
<begin_content>
file_content
<end_content>

Include at least the following files in this order:
- setup.py
- README.md
- requirements.txt
- examples/example_{package_name}.py
- tests/test_{package_name}.py
- {package_name}/__init__.py
- {package_name}/main.py

Ensure there are no extra lines or spaces between the tags and the content.

**Example:**

<setup.py>
<begin_content>
from setuptools import setup, find_packages

setup(
    name='examplepkg',
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
"""

    try:
        response_text = generate_content_with_groq(client, model_name, prompt)
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

    Args:
        package_structure (Dict[str, str]): Dictionary mapping file paths to their contents.
        package_name (str): Name of the Python package.
        features (List[str]): List of features provided by the user.

    Returns:
        Optional[str]: Path to the generated DOCUMENTATION.md file or None if failed.
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

    # Installation
    documentation_lines.append("## Installation\n")
    documentation_lines.append("```bash")
    documentation_lines.append(f"pip install {package_name}")
    documentation_lines.append("```")
    documentation_lines.append("")

    # Usage Examples
    example_file = f"examples/example_{package_name}.py"
    if example_file in package_structure:
        example_content = package_structure[example_file]
        documentation_lines.append("## Usage Example\n")
        documentation_lines.append("```python")
        documentation_lines.append(example_content)
        documentation_lines.append("```")
        documentation_lines.append("")

    # API Reference
    main_file = f"{package_name}/main.py"
    if main_file in package_structure:
        main_content = package_structure[main_file]
        # Attempt to extract classes and functions
        classes = re.findall(r"class\s+(\w+)\(.*\):", main_content)
        functions = re.findall(r"def\s+(\w+)\(.*\):", main_content)

        if classes or functions:
            documentation_lines.append("## API Reference\n")
            if classes:
                documentation_lines.append("### Classes\n")
                for cls in classes:
                    documentation_lines.append(f"- `{cls}`")
                documentation_lines.append("")
            if functions:
                documentation_lines.append("### Functions\n")
                for func in functions:
                    documentation_lines.append(f"- `{func}()`")
                documentation_lines.append("")

    # Testing
    test_file = f"tests/test_{package_name}.py"
    if test_file in package_structure:
        documentation_lines.append("## Testing\n")
        documentation_lines.append("To run the tests, execute:\n")
        documentation_lines.append("```bash")
        documentation_lines.append("python -m unittest")
        documentation_lines.append("```")
        documentation_lines.append("")

    # Dependencies
    requirements = package_structure.get("requirements.txt", "")
    if requirements:
        documentation_lines.append("## Dependencies\n")
        documentation_lines.append("The package requires the following dependencies:")
        documentation_lines.append("```")
        documentation_lines.append(requirements)
        documentation_lines.append("```")
        documentation_lines.append("")

    # Contribution Guidelines
    documentation_lines.append("## Contributing\n")
    documentation_lines.append("Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.\n")

    # License
    documentation_lines.append("## License\n")
    documentation_lines.append("This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.\n")

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

    print("\nGenerating Python Package...")
    package_structure = generate_python_package(package_name, package_description, features, client, model_name)

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
