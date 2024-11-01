import os
import io
import zipfile
import warnings
import streamlit as st
import google.generativeai as genai 
from groq import Groq
from typing import Dict, Optional, List
import re
import datetime
import time

# Suppress all warnings
warnings.filterwarnings("ignore")

def parse_generated_content(content: str) -> Dict[str, str]:
    """
    Parse the generated content into a dictionary of file paths and contents.
    Expects the format:

    <file_path>
    <begin_content>
    file_content here
    <end_content>
    """
    files = {}
    current_file = None
    current_content = []
    state = 'expect_file'

    lines = content.strip().split('\n')
    line_num = 0
    while line_num < len(lines):
        line = lines[line_num]
        stripped = line.strip()
        line_num += 1

        if state == 'expect_file':
            if stripped.startswith('<') and stripped.endswith('>'):
                current_file = stripped[1:-1].strip()
                if not current_file:
                    current_file = None
                    continue
                state = 'expect_begin_content'
            else:
                continue  # Ignore any text before the first <file_path>
        elif state == 'expect_begin_content':
            if stripped == '<begin_content>':
                current_content = []
                state = 'collect_content'
            else:
                current_file = None
                state = 'expect_file'
        elif state == 'collect_content':
            if stripped == '<end_content>':
                if current_file:
                    files[current_file] = '\n'.join(current_content).strip()
                current_file = None
                current_content = []
                state = 'expect_file'
            else:
                current_content.append(line)
    # Handle incomplete file definitions
    if state == 'collect_content' and current_file:
        files[current_file] = '\n'.join(current_content).strip()
    return files

def generate_feature_descriptions(client: Groq, model_name: str, features: List[str], max_retries: int = 5) -> Dict[str, str]:
    """
    Generate feature descriptions using the Groq model with retry logic.

    Args:
        client (Groq): The Groq client instance.
        model_name (str): The model name to use.
        features (List[str]): List of feature names.
        max_retries: Maximum number of retries for each request.

    Returns:
        Dict[str, str]: A dictionary mapping feature names to their descriptions.
    """
    feature_descriptions = {}
    for feature in features:
        prompt = f"Provide a detailed, user-friendly description for the following feature of an AutoML system, focusing on implementation details. Include elaborate explanations and pseudocode for implementation to help generate the code.\n\nFeature: {feature}\n\nDescription:"
        retries = 0
        while retries < max_retries:
            try:
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model_name,
                )
                description = response.choices[0].message.content.strip()
                feature_descriptions[feature] = description
                break  # Break out of the retry loop if successful
            except Exception as e:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff
                print(f"Error generating description for feature '{feature}': {str(e)}")
                print(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(wait_time)
        else:
            print(f"Failed to generate description for feature '{feature}' after {max_retries} attempts.")
            feature_descriptions[feature] = ""  # Fallback to empty description
    return feature_descriptions

def enhance_package_description(client: Groq, model_name: str, package_description: str, max_retries: int = 5) -> str:
    """
    Enhance the package description using the Groq model with retry logic.

    Args:
        client (Groq): The Groq client instance.
        model_name (str): The model name to use.
        package_description (str): Original package description.
        max_retries (int): Maximum number of retries for the request.

    Returns:
        str: Enhanced package description.
    """
    prompt = f"Enhance the following package description to make it more comprehensive and user-friendly, focusing on implementation details and including elaborate explanations to help with code generation.\n\n{package_description}\n\nEnhanced Description:"
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )
            enhanced_description = response.choices[0].message.content.strip()
            return enhanced_description
        except Exception as e:
            retries += 1
            wait_time = 2 ** retries
            print(f"Error enhancing package description: {str(e)}")
            print(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
            time.sleep(wait_time)
    else:
        print(f"Failed to enhance package description after {max_retries} attempts.")
        return package_description

def generate_python_package(package_name: str, package_description: str, features: Optional[List[str]], api_provider: str, api_key: str, model_name: str) -> Optional[Dict[str, str]]:
    """Generate a Python package structure and content using Groq or GEMINI models."""
    if not api_key:
        st.error("🔑 API key is required.")
        return None

    if api_provider == "GEMINI":
        try:
            # genai.configure(api_key=api_key)  # Uncomment if using GEMINI
            # model = genai.GenerativeModel(model_name)  # Uncomment if using GEMINI
            client = None  # No Groq client
            st.error("GEMINI API integration is currently not implemented.")
            return None
        except Exception as e:
            st.error(f"⚠️ Error initializing GEMINI model: {e}")
            return None
    elif api_provider == "Groq":
        try:
            client = Groq(api_key=api_key)
            model = None  # No GEMINI model
        except Exception as e:
            st.error(f"⚠️ Error initializing Groq client: {e}")
            return None
    else:
        st.error("⚠️ Invalid API provider selected.")
        return None

    # Enhance package description if using Groq
    if api_provider == "Groq":
        st.info("Enhancing package description...")
        package_description = enhance_package_description(client, model_name, package_description)
        # Generate detailed feature descriptions
        if features:
            st.info("Generating detailed feature descriptions...")
            feature_descriptions = generate_feature_descriptions(client, model_name, features)
            # Use the enhanced feature descriptions
            features_formatted = ""
            for feature in features:
                description = feature_descriptions.get(feature, "")
                features_formatted += f"- {feature.strip()}:\n{description}\n"
        else:
            features_formatted = ""
    else:
        # For GEMINI, use the original descriptions
        # Format the features as a bulleted list
        if features:
            features_formatted = "\n".join(f"- {feature.strip()}" for feature in features if feature.strip())
        else:
            features_formatted = ""

    features_section = ""
    if features_formatted:
        features_section = f"\n\nInclude the following features with detailed descriptions:\n{features_formatted}\nFor each feature, generate the following files:\n- {package_name}/{{feature_lower}}.py\n- examples/example_{{feature_lower}}.py\n- tests/test_{{feature_lower}}.py"

    # Updated prompt with explicit instructions
    prompt = f"""
You are to generate the structure and content of a Python package named `{package_name}` based on the following description, Also remember to be verbose like you're life depends on it, :
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

{"Additionally, for each feature provided, include the following files:\n- " + "\n- ".join([f"{package_name}/{{feature_lower}}.py", f"examples/example_{{feature_lower}}.py", f"tests/test_{{feature_lower}}.py"]) if features else ""}

Ensure there are no extra lines or spaces between the tags and the content.

**Important Instructions:**

- Do **not** include any text outside the specified format.
- Do **not** provide any explanations, introductions, or conclusions.
- Ensure that every `<begin_content>` has a matching `<end_content>`.
- Use the exact file paths and filenames as specified.
- The content inside `<begin_content>` and `<end_content>` should be the code or text for that file.
- Replace `{{feature_lower}}` with the lowercase version of the feature name, replacing spaces with underscores.

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

Please begin the response now. Be verbose, Try to write at least 100 lines of code in each file you're generating.
"""

    try:
        with st.spinner(f"🤖 Generating package using {api_provider} model {model_name}..."):
            if api_provider == "GEMINI":
                # response = model.generate_content(prompt)  # Uncomment if using GEMINI
                st.error("GEMINI API integration is currently not implemented.")
                return None
                # if not hasattr(response, 'text') or not response.text.strip():
                #     st.error("⚠️ The model did not return any content. Please try again with different inputs.")
                #     return None
                # response_text = response.text
            elif api_provider == "Groq":
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model_name,
                )
                response_text = response.choices[0].message.content.strip()
                if not response_text:
                    st.error("⚠️ The model did not return any content. Please try again with different inputs.")
                    return None
            else:
                st.error("⚠️ Invalid API provider selected.")
                return None

            st.success("✅ Package generated successfully!")
            st.text_area("📄 Raw AI Response", value=response_text, height=300)
            parsed_content = parse_generated_content(response_text)

            if not parsed_content:
                st.warning("⚠️ The model's response could not be parsed into a valid package structure.")
                # Optionally, you can retry or provide more feedback
                return None

            return parsed_content
    except Exception as e:
        st.error(f"Sorry, an error occurred while generating the package: {e}")
        return None

def create_fallback_structure(package_name: str, features: Optional[List[str]] = None) -> Dict[str, str]:
    """Create a fallback package structure if the model fails to generate one."""
    fallback = {
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
""",
    }

    # Add fallback feature files if features are provided
    if features:
        for feature in features:
            feature_lower = feature.lower().replace(" ", "_")
            fallback[f"{package_name}/{feature_lower}.py"] = f"""# {feature} feature implementation

class {feature.replace(' ', '')}:
    def __init__(self):
        self.state = None

    def perform_action(self):
        \"\"\"Perform the {feature} action.\"\"\"
        # Implement the action here
        self.state = "Action performed"
        print("Action performed successfully.")

    def additional_method_1(self):
        # Implement additional functionality here
        pass  # Replace with actual code

    def additional_method_2(self):
        # Implement additional functionality here
        pass  # Replace with actual code

    def additional_method_3(self):
        # Implement additional functionality here
        pass  # Replace with actual code

    # Continue adding methods as needed
"""

            fallback[f"examples/example_{feature_lower}.py"] = f"""# Example usage of the {feature} feature

from {package_name}.{feature_lower} import {feature.replace(' ', '')}

def main():
    feature = {feature.replace(' ', '')}()
    feature.perform_action()
    # Add more example usage as needed

if __name__ == "__main__":
    main()
"""

            fallback[f"tests/test_{feature_lower}.py"] = f"""import unittest
from {package_name}.{feature_lower} import {feature.replace(' ', '')}

class Test{feature.replace(' ', '')}(unittest.TestCase):
    def setUp(self):
        self.feature = {feature.replace(' ', '')}()

    def test_perform_action(self):
        self.feature.perform_action()
        self.assertEqual(self.feature.state, "Action performed")
    
    def test_additional_method_1(self):
        # Implement actual tests here
        self.assertIsNone(self.feature.additional_method_1())  # Replace with actual tests
    
    def test_additional_method_2(self):
        # Implement actual tests here
        self.assertIsNone(self.feature.additional_method_2())  # Replace with actual tests
    
    def test_additional_method_3(self):
        # Implement actual tests here
        self.assertIsNone(self.feature.additional_method_3())  # Replace with actual tests
    
    # Continue adding test methods as needed

if __name__ == '__main__':
    unittest.main()
"""

    return fallback

def validate_package_structure(package_structure: Dict[str, str]) -> bool:
    """Validate that the package structure has non-empty file paths and contents."""
    valid = True
    for file_path, content in package_structure.items():
        if not file_path:
            valid = False
        if not content:
            valid = False
    return valid

def create_zip(package_structure: Dict[str, str], package_name: str) -> bytes:
    """Create a zip file from the package structure with a different but relevant name."""
    # Generate a different zip file name by appending '_package' and current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    zip_filename = f"{package_name}_package_{timestamp}.zip"

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path, content in package_structure.items():
            zipf.writestr(file_path, content)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def generate_markdown_documentation(client: Groq, model_name: str, package_structure: Dict[str, str], package_name: str, max_retries: int = 5) -> Optional[str]:
    """
    Generate an enhanced Markdown documentation file (DOCUMENTATION.md) using the AI model.

    Args:
        client (Groq): The AI client instance.
        model_name (str): The model name to use.
        package_structure (Dict[str, str]): Dictionary mapping file paths to their contents.
        package_name (str): Name of the Python package.
        max_retries (int): Maximum number of retries for the request.

    Returns:
        Optional[str]: The content of the generated DOCUMENTATION.md file or None if failed.
    """
    # Prepare the prompt
    prompt = f"""
You are to generate comprehensive documentation for a Python package named '{package_name}'. The package contains the following files and their contents:

"""

    for file_path, content in package_structure.items():
        prompt += f"File: {file_path}\nContent:\n\"\"\"\n{content}\n\"\"\"\n\n"

    prompt += f"""
Based on the above files and their contents, generate detailed and user-friendly documentation in Markdown format. The documentation should include an overview, installation instructions, usage examples, API references, and any other relevant sections that would help users understand and utilize the package effectively. Do not include any code that is not part of the package. Ensure the documentation is well-structured and written in clear, professional language. Be and act like technical writer. Try to be verbose as much as possible. Write the documentation in a tutorial like style and explain how to use each file.

Begin the documentation below:
"""

    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    }
                ],
                model=model_name,
            )
            documentation_content = response.choices[0].message.content.strip()
            return documentation_content
        except Exception as e:
            retries += 1
            wait_time = 2 ** retries
            print(f"Error generating documentation: {str(e)}")
            print(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
            time.sleep(wait_time)
    else:
        print(f"Failed to generate documentation after {max_retries} attempts.")
        return None

def main():
    st.set_page_config(page_title="🛠️ Python Package Generator", layout="wide")
    st.title("🛠️ Python Package Generator")
    st.write("Generate a Python package structure using Groq or GEMINI models and download it as a ZIP file.")

    # API Configuration
    st.header("🔑 API Configuration")
    api_provider = st.selectbox("Select API Provider", ["Groq"], key="api_provider_select")

    if api_provider == "Groq":
        api_key = st.text_input("🔐 Enter your Groq API Key", type="password", key="groq_api_key_input")

        # Define Groq models
        groq_models = [
            'llama3-8b-8192',
            'llama3-13b-8192',
            'llama3-70b-8192',
            'llama-3.1-70b-versatile',
            'llama3-groq-8b-8192-tool-use-preview',
            'llama3-groq-70b-8192-tool-use-preview',
            'gemma2-9b-it',
            'gemma-7b-it',
            'mixtral-8x7b-32768',
            # Add more Groq models as needed
        ]
        model_list = groq_models

    # elif api_provider == "GEMINI":
    #     api_key = st.text_input("🔐 Enter your GEMINI API Key", type="password", key="gemini_api_key_input")

    #     # Define GEMINI models
    #     gemini_models = [
    #         'gemini-1.5-pro-002',
    #         'gemini-1.5-pro',
    #         'gemini-1.5-flash',
    #         'gemini-1.5-flash-002',
    #         'gemini-1.5-flash-8b',
    #         # Add more GEMINI models as needed
    #     ]
    #     model_list = gemini_models

    else:
        api_key = ''
        model_list = []

    # Model Selection
    model_name = st.selectbox("Select Model", model_list, key="model_select")

    # User Inputs
    st.header("📦 Package Details")
    package_name = st.text_input("📝 Enter the name for your Python package:", "", key="package_name_input").strip()
    package_description = st.text_area("📝 Enter a brief description of your package:", "", height=150, key="package_description_input").strip()

    # Features Input
    st.subheader("✨ Features")
    features_input = st.text_area("Enter features (one per line):", value="", height=150, key="features_input")
    features = [f.strip() for f in features_input.split('\n') if f.strip()]

    # Documentation Option
    st.subheader("📄 Documentation")
    generate_doc = st.checkbox("Generate Documentation (DOCUMENTATION.md)", value=True, key="generate_doc_checkbox")

    # Generate Package Button
    if st.button("🚀 Generate Package"):
        if not api_key:
            st.error("🔑 API key cannot be empty.")
            return
        if not package_name:
            st.error("📦 Package name cannot be empty.")
            return
        if not package_description:
            st.error("📝 Package description cannot be empty.")
            return
        if not model_name:
            st.error("⚠️ Model name cannot be empty.")
            return

        # Initialize the AI client
        if api_provider == "Groq":
            try:
                client = Groq(api_key=api_key)
            except Exception as e:
                st.error(f"⚠️ Error initializing Groq client: {e}")
                return
        else:
            st.error("⚠️ Invalid API provider selected.")
            return

        package_structure = generate_python_package(package_name, package_description, features, api_provider, api_key, model_name)

        if not package_structure or not validate_package_structure(package_structure):
            st.warning("⚠️ Failed to generate a valid package structure. Using fallback structure.")
            package_structure = create_fallback_structure(package_name, features)

        # Generate Documentation if selected
        documentation_content = None
        if generate_doc:
            st.info("Generating documentation using the AI model...")
            documentation_content = generate_markdown_documentation(client, model_name, package_structure, package_name)
            if documentation_content:
                package_structure["DOCUMENTATION.md"] = documentation_content
                st.success("✅ Documentation generated successfully!")
                st.text_area("📝 DOCUMENTATION.md Content", value=documentation_content, height=600)
            else:
                st.warning("⚠️ Failed to generate documentation.")

        st.header("📁 Generated Package Structure")
        st.write("Here is the structure of your generated Python package:")
        for file_path in package_structure.keys():
            st.write(f"- `{file_path}`")

        # Create ZIP
        with st.spinner("📦 Creating ZIP file..."):
            zip_data = create_zip(package_structure, package_name)

        # Provide Download Button for ZIP
        st.success("✅ Package structure is ready for download!")
        zip_filename = f"{package_name}_package_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
        st.download_button(
            label="📥 Download Package as ZIP",
            data=zip_data,
            file_name=zip_filename,
            mime="application/zip"
        )

        # Provide Separate Download Button for Documentation if generated
        if generate_doc and documentation_content:
            st.download_button(
                label="📥 Download Documentation (DOCUMENTATION.md)",
                data=documentation_content,
                file_name="DOCUMENTATION.md",
                mime="text/markdown"
            )

        # Optionally, display the contents
        with st.expander("📄 View Package Files"):
            for file_path, content in package_structure.items():
                st.subheader(f"📄 {file_path}")
                st.text_area("", value=content, height=400)

if __name__ == "__main__":
    main()
