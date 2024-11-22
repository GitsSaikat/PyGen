# prompt_generator.py

import os
import json
import time
from typing import List, Dict
from groq import Groq

def get_user_input() -> tuple:
    """
    Get user input for package description and features.
    """
    package_description = input("Enter a brief description of your package: ").strip()
    while not package_description:
        print("Package description cannot be empty. Please try again.")
        package_description = input("Enter a brief description of your package: ").strip()

    features = []
    print("\nEnter the features you want to include in your package.")
    print("Type each feature and press Enter. When done, just press Enter on an empty line.")
    while True:
        feature = input("Enter a feature (or press Enter to finish): ").strip()
        if not feature:
            break
        features.append(feature)

    return package_description, features

def generate_feature_descriptions(client: Groq, model_name: str, features: List[str], max_retries: int = 5) -> Dict[str, str]:
    """
    Generate feature descriptions using the Groq model with retry logic.

    Args:
        client (Groq): The Groq client instance.
        model_name (str): The model name to use.
        features (List[str]): List of feature names.
        max_retries (int): Maximum number of retries for each request.

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
        return package_description  # Fallback to original description

def save_feature_descriptions(feature_descriptions: Dict[str, str], json_output_file: str, txt_output_file: str) -> None:
    """
    Save the feature descriptions to a JSON file and a text file.

    Args:
        feature_descriptions (Dict[str, str]): Feature descriptions.
        json_output_file (str): Output JSON file path.
        txt_output_file (str): Output text file path.
    """
    try:
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(feature_descriptions, f, indent=4)
        print(f"Feature descriptions saved to {json_output_file}")
    except Exception as e:
        print(f"Error saving feature descriptions to {json_output_file}: {str(e)}")

    try:
        with open(txt_output_file, 'w', encoding='utf-8') as f:
            for feature, description in feature_descriptions.items():
                f.write(f"Feature: {feature}\n")
                f.write(f"Description:\n{description}\n")
                f.write("\n")
        print(f"Feature descriptions saved to {txt_output_file}")
    except Exception as e:
        print(f"Error saving feature descriptions to {txt_output_file}: {str(e)}")

def save_package_description(enhanced_description: str, output_file: str) -> None:
    """
    Save the enhanced package description to a text file.

    Args:
        enhanced_description (str): Enhanced package description.
        output_file (str): Output file path.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_description)
        print(f"Enhanced package description saved to {output_file}")
    except Exception as e:
        print(f"Error saving package description to {output_file}: {str(e)}")

def main():
    # Set the environment variable or get the API key from the user
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = input("Please enter your Groq API key: ").strip()
        os.environ["GROQ_API_KEY"] = api_key

    # Initialize Groq client
    client = Groq(api_key=api_key)

    # Prompt the user for the model name
    print("\nPlease enter the model name you wish to use.")
    print("Example model names: 'llama3-8b-8192', 'llama3-70b-8192', 'llama-3.1-70b-versatile', 'llama3-groq-8b-8192-tool-use-preview', 'llama3-groq-70b-8192-tool-use-preview', 'gemma2-9b-it', 'gemma-7b-it', 'mixtral-8x7b-32768', etc.")
    model_name = input("Enter the model name to use: ").strip()
    while not model_name:
        print("Model name cannot be empty. Please try again.")
        model_name = input("Enter the model name to use: ").strip()

    # Get user input
    package_description, features = get_user_input()

    # Enhance package description
    print("\nEnhancing package description...")
    enhanced_package_description = enhance_package_description(client, model_name, package_description)

    # Generate feature descriptions
    print("Generating feature descriptions...")
    feature_descriptions = generate_feature_descriptions(client, model_name, features)

    # Save enhanced package description
    package_description_file = 'package_descriptions_enhanced.txt'
    save_package_description(enhanced_package_description, package_description_file)

    # Save feature descriptions to JSON and txt
    feature_descriptions_json_file = 'feature_descriptions_enhanced.json'
    feature_descriptions_txt_file = 'feature_descriptions_enhanced.txt'
    save_feature_descriptions(feature_descriptions, feature_descriptions_json_file, feature_descriptions_txt_file)

if __name__ == "__main__":
    main()
