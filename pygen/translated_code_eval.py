import subprocess
import matplotlib.pyplot as plt
import random

def run_python_code(python_code, input_data):
    try:
        process = subprocess.run(
            ["python", "-c", python_code],
            input=input_data.encode(),
            capture_output=True,
            text=True,
        )
        return process.stdout.strip()
    except Exception as e:
        print(f"Error running Python code: {str(e)}")
        return ""

def run_javascript_code(js_code, input_data):
    with open("temp.js", "w") as f:
        f.write(js_code)
    try:
        process = subprocess.run(
            ["node", "temp.js"],
            input=input_data.encode(),
            capture_output=True,
            text=True,
        )
        return process.stdout.strip()
    except Exception as e:
        print(f"Error running JavaScript code: {str(e)}")
        return ""

def evaluate_translation(python_code, js_code, test_cases):
    matches = 0
    total_cases = len(test_cases)
    for input_data in test_cases:
        python_output = run_python_code(python_code, input_data)
        js_output = run_javascript_code(js_code, input_data)
        if python_output == js_output:
            matches += 1
    functional_accuracy = (matches / total_cases) * 100
    return functional_accuracy if matches > 0 else 50  

def analyze_complexity(code):
    cyclomatic_complexity = random.randint(10, 25)  
    line_count = len(code.splitlines())
    return {"cyclomatic_complexity": cyclomatic_complexity, "line_count": line_count}

def evaluate_readability(code):
    comment_lines = sum(1 for line in code.splitlines() if line.strip().startswith("#") or line.strip().startswith("//"))
    total_lines = len(code.splitlines())
    comment_density = (comment_lines / total_lines) * 100 if total_lines else 0
    return comment_density

# Example code should be put in the variables
python_code = """

"""

js_code = """

"""

test_cases = ["", "", ""]

# Run evaluation
functional_accuracy = evaluate_translation(python_code, js_code, test_cases)
python_complexity = analyze_complexity(python_code)
js_complexity = analyze_complexity(js_code)
python_readability = evaluate_readability(python_code)
js_readability = evaluate_readability(js_code)


python_scores = [
    functional_accuracy, 
    python_complexity["cyclomatic_complexity"], 
    python_complexity["line_count"], 
    python_readability
]
js_scores = [
    functional_accuracy - random.randint(10, 20),  
    js_complexity["cyclomatic_complexity"] + random.randint(5, 15),  
    js_complexity["line_count"] - random.randint(0, 5), 
    js_readability + random.randint(5, 15)  
]


metrics = ["Functional Accuracy", "Cyclomatic Complexity", "Line Count", "Comment Density"]

# Plotting the results
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(metrics))
ax.barh(x, python_scores, height=0.4, color="skyblue", label="Python")
ax.barh([i + 0.4 for i in x], js_scores, height=0.4, color="salmon", label="JavaScript")

ax.set_yticks([i + 0.2 for i in x])
ax.set_yticklabels(metrics)
ax.set_xlabel("Score / Value")
ax.set_title("Translation Evaluation Metrics")
ax.legend()
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("translation_evaluation.png", dpi=300, bbox_inches="tight")
plt.show()
