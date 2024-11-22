import os
import generate_package
from typing import Dict, Optional

def generate_markdown_documentation(package_structure: Dict[str, str], package_name: str) -> Optional[str]:
    """
    Generate a Markdown documentation file (DOCUMENTATION.md) based on the generated package structure.

    Args:
        package_structure (Dict[str, str]): Dictionary mapping file paths to their contents.
        package_name (str): Name of the Python package.

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
    
    # Features (Assuming features are listed in setup.py or elsewhere)
    setup_content = package_structure.get("setup.py", "")
    features = []
    if setup_content:
        # Attempt to extract features from install_requires or classifiers
        lines = setup_content.split('\n')
        in_install_requires = False
        for line in lines:
            if "install_requires" in line:
                in_install_requires = True
                continue
            if in_install_requires:
                if ']' in line:
                    in_install_requires = False
                    continue
                feature = line.strip().strip(',').strip("'\"")
                if feature:
                    features.append(feature)
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
        import re
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
    
    # License (Assuming MIT License as per setup.py example)
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

# Example Integration with Existing Script

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
    
    # Generate Documentation
    generate_doc = input("\nDo you want to generate documentation? (y/n): ").strip().lower()
    if generate_doc == 'y':
        documentation_path = generate_markdown_documentation(package_structure, package_name)
        if documentation_path:
            print(f"Documentation successfully created at {documentation_path}")
        else:
            print("Failed to create documentation.")
    else:
        print("Documentation was not created.")

if __name__ == "__main__":
    main()
