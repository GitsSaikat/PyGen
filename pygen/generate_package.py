import os
import google.generativeai as genai
from typing import Dict, List, Optional

def get_api_key() -> str:
    """Retrieve the Google API key from an environment variable or user input."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Please enter your Google API key: ").strip()
    return api_key

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

def generate_python_package(package_name: str, package_description: str, features: List[str]) -> Optional[Dict[str, str]]:
    """Generate a Python package structure and content using Google Gemini Pro 002 model."""
    api_key = get_api_key()
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-1.5-pro-002')

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
        response = model.generate_content(prompt)
        print("=== Raw AI Response ===")
        print(response.text)
        print("========================\n")
        parsed_content = parse_generated_content(response.text)
        
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

def main():
    package_name, package_description, features = get_user_input()
    
    print("\nGenerating Python Package...")
    package_structure = generate_python_package(package_name, package_description, features)
    
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

if __name__ == "__main__":
    main()
