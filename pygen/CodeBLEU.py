# ! pip install codebleu
# ! pip install tree-sitter==0.23.1 tree-sitter-python==0.23.2

from codebleu import calc_codebleu
import ast
import asttokens
from collections import Counter

# Define your generated and reference code
generated_code = """

"""

reference_code = """

"""

# Function to calculate Token Match Score
def calculate_token_match(generated_code: str, reference_code: str) -> float:
    import re
    def tokenize(code):
        # Remove comments and docstrings
        code = re.sub(r'(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        tokens = re.findall(r'\b\w+\b', code)
        return set(tokens)
    
    gen_tokens = tokenize(generated_code)
    ref_tokens = tokenize(reference_code)
    
    intersection = gen_tokens.intersection(ref_tokens)
    union = gen_tokens.union(ref_tokens)
    
    if not union:
        return 0.0
    return len(intersection) / len(union)

# Function to calculate Identifier Match Score
def calculate_identifier_match(generated_code: str, reference_code: str) -> float:
    import ast
    import asttokens
    def extract_identifiers(code):
        atok = asttokens.ASTTokens(code, parse=True)
        identifiers = set()
        for node in ast.walk(atok.tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.FunctionDef):
                identifiers.add(node.name)
            elif isinstance(node, ast.ClassDef):
                identifiers.add(node.name)
        return identifiers
    
    gen_identifiers = extract_identifiers(generated_code)
    ref_identifiers = extract_identifiers(reference_code)
    
    intersection = gen_identifiers.intersection(ref_identifiers)
    union = gen_identifiers.union(ref_identifiers)
    
    if not union:
        return 0.0
    return len(intersection) / len(union)

# Calculate Token Match Score and Identifier Match Score
token_match_score = calculate_token_match(generated_code, reference_code)
identifier_match_score = calculate_identifier_match(generated_code, reference_code)

# Calculate CodeBLEU score
result = calc_codebleu(
    references=[reference_code],
    predictions=[generated_code],
    lang='python',
    weights=(0.25, 0.25, 0.25, 0.25)) 

# Display the results
print(f"CodeBLEU Score: {result['codebleu']:.4f}")
print(f"N-gram Match Score: {result['ngram_match_score']:.4f}")
print(f"Weighted N-gram Match Score: {result['weighted_ngram_match_score']:.4f}")
print(f"Syntax Match Score: {result['syntax_match_score']:.4f}")
print(f"Dataflow Match Score: {result['dataflow_match_score']:.4f}")
print(f"Token Match Score: {token_match_score:.4f}")
print(f"Identifier Match Score: {identifier_match_score:.4f}")
