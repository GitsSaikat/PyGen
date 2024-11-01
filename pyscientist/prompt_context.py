import os
import time
from typing import List, Dict
from groq import Groq

def load_feature_descriptions(feature_files: List[str]) -> Dict[str, str]:
    """
    Load feature descriptions from specified text files.
    """
    feature_descriptions = {}
    for file_path in feature_files:
        if not os.path.exists(file_path):
            print(f"Warning: Feature file {file_path} does not exist.")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Assuming the file contains multiple features in the format:
                # Feature: <feature_name>
                # Description:
                # <description>
                entries = content.strip().split('\n\n')
                for entry in entries:
                    lines = entry.strip().split('\n')
                    if len(lines) >= 2 and lines[0].startswith('Feature:'):
                        feature_name = lines[0][len('Feature:'):].strip()
                        description = '\n'.join(lines[1:]).replace('Description:', '').strip()
                        feature_descriptions[feature_name] = description
        except Exception as e:
            print(f"Error loading feature file {file_path}: {str(e)}")
    return feature_descriptions

def load_code_examples(example_files: List[str]) -> Dict[str, str]:
    """
    Load code examples from specified text files.
    """
    code_examples = {}
    for file_path in example_files:
        if not os.path.exists(file_path):
            print(f"Warning: Code example file {file_path} does not exist.")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                code_examples[os.path.basename(file_path)] = content
        except Exception as e:
            print(f"Error loading code example file {file_path}: {str(e)}")
    return code_examples

def generate_enhanced_feature_descriptions(client: Groq, model_name: str, feature_descriptions: Dict[str, str], max_retries: int = 5) -> Dict[str, str]:
    """
    Use the Groq model to generate enhanced descriptions for features.
    """
    enhanced_descriptions = {}
    for feature, description in feature_descriptions.items():
        prompt = f"Provide a concise summary of the following feature for an AutoML system, focusing on key points:\n\nFeature: {feature}\nDescription:\n{description}\n\nSummary:"
        retries = 0
        while retries < max_retries:
            try:
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=model_name,
                )
                enhanced_description = response.choices[0].message.content.strip()
                enhanced_descriptions[feature] = enhanced_description
                break
            except Exception as e:
                retries += 1
                wait_time = 2 ** retries
                print(f"Error generating enhanced description for feature '{feature}': {str(e)}")
                print(f"Retrying in {wait_time} seconds... (Attempt {retries})")
                time.sleep(wait_time)
        else:
            print(f"Failed to generate enhanced description for feature '{feature}' after {max_retries} attempts.")
            enhanced_descriptions[feature] = description  # Fallback to original description
    return enhanced_descriptions

def generate_code_summaries(client: Groq, model_name: str, code_examples: Dict[str, str], max_retries: int = 5) -> Dict[str, str]:
    """
    Use the Groq model to generate summaries for code examples.
    """
    code_summaries = {}
    for filename, code in code_examples.items():
        prompt = f"Summarize the following code snippet, explaining its purpose and functionality in the context of an AutoML system:\n\n```python\n{code}\n```\n\nSummary:"
        retries = 0
        while retries < max_retries:
            try:
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=model_name,
                )
                summary = response.choices[0].message.content.strip()
                code_summaries[filename] = summary
                break
            except Exception as e:
                retries += 1
                wait_time = 2 ** retries
                print(f"Error generating summary for code example '{filename}': {str(e)}")
                print(f"Retrying in {wait_time} seconds... (Attempt {retries})")
                time.sleep(wait_time)
        else:
            print(f"Failed to generate summary for code example '{filename}' after {max_retries} attempts.")
            code_summaries[filename] = ""  # Fallback to empty summary
    return code_summaries

def prepare_prompt_context(enhanced_feature_descriptions: Dict[str, str], code_summaries: Dict[str, str]) -> str:
    """
    Prepare additional context to be included in the prompt for the model.
    """
    context_lines = []

    # Include enhanced feature descriptions
    if enhanced_feature_descriptions:
        context_lines.append("Enhanced Feature Descriptions:")
        for feature, description in enhanced_feature_descriptions.items():
            context_lines.append(f"Feature: {feature}")
            context_lines.append(f"{description}")
            context_lines.append("")  # Add an empty line

    # Include code summaries
    if code_summaries:
        context_lines.append("Code Summaries:")
        for filename, summary in code_summaries.items():
            context_lines.append(f"### {filename}")
            context_lines.append(summary)
            context_lines.append("")  # Add an empty line

    return "\n".join(context_lines)

def save_prompt_context(context: str, output_file: str) -> None:
    """
    Save the prepared context to a file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(context)
        print(f"Context saved to {output_file}")
    except Exception as e:
        print(f"Error saving context to {output_file}: {str(e)}")

def main():
    # Paths to data files, it also supports .py and .json files
    feature_files = ['feature_descriptions_enhanced.txt']
    code_example_files = ['package_descriptions_enhanced.txt']

    # Load data
    feature_descriptions = load_feature_descriptions(feature_files)
    code_examples = load_code_examples(code_example_files)

    # Set the environment variable or get the API key from the user
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = input("Please enter your Groq API key: ").strip()
        os.environ["GROQ_API_KEY"] = api_key

    # Initialize Groq client
    client = Groq(api_key=api_key)

    # Prompt the user for the model name
    print("\nPlease enter the model name you wish to use for data preparation.")
    print("Example model names: 'llama3-8b-8192', 'llama3-70b-8192', etc.")
    model_name = input("Enter the model name to use: ").strip()
    while not model_name:
        print("Model name cannot be empty. Please try again.")
        model_name = input("Enter the model name to use: ").strip()

    # Generate enhanced feature descriptions
    print("\nGenerating enhanced feature descriptions...")
    enhanced_feature_descriptions = generate_enhanced_feature_descriptions(client, model_name, feature_descriptions)

    # Generate code summaries
    print("Generating code summaries...")
    code_summaries = generate_code_summaries(client, model_name, code_examples)

    # Prepare prompt context
    context = prepare_prompt_context(enhanced_feature_descriptions, code_summaries)

    # Save context to a file
    output_file = 'prompt_context.txt'
    save_prompt_context(context, output_file)

if __name__ == "__main__":
    main()
