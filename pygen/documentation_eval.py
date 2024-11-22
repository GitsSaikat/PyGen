from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import textstat

# Define the DocumentationEvaluator class
class DocumentationEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def calculate_relevance(self, generated_response, reference_response):
        gen_embedding = self.model.encode([generated_response])
        ref_embedding = self.model.encode([reference_response])
        similarity = cosine_similarity(gen_embedding, ref_embedding)
        return similarity[0][0]

    def calculate_consistency(self, responses):
        if not responses:
            print("Warning: No responses provided for consistency calculation.")
            return float('nan')

        embeddings = self.model.encode(responses)
        mean_embedding = np.mean(embeddings, axis=0)
        variances = np.var(embeddings - mean_embedding, axis=0)
        consistency_score = np.mean(variances)
        return consistency_score

    def calculate_readability(self, text):
        if not text:
            print("Warning: Empty text provided for readability calculation.")
            return float('nan')

        flesch_reading_ease = textstat.flesch_reading_ease(text)
        return flesch_reading_ease

    def evaluate_documentation(self, generated_doc, reference_doc=None):
        scores = {}
        scores['readability'] = self.calculate_readability(generated_doc)
        responses = [section.strip() for section in generated_doc.split('\n') if section.strip()]
        scores['consistency'] = self.calculate_consistency(responses)

        if reference_doc:
            scores['relevance'] = self.calculate_relevance(generated_doc, reference_doc)
        else:
            scores['relevance'] = None

        return scores

# Initialize the evaluator
evaluator = DocumentationEvaluator()

# Define your generated and reference documentation
generated_doc = """

"""

reference_doc = """

"""

# Evaluate the documentation
scores = evaluator.evaluate_documentation(generated_doc, reference_doc)

# Print the scores
print("Documentation Evaluation Scores:")
print(f"Readability Score: {scores['readability']:.2f}")
print(f"Consistency Score: {scores['consistency']:.4f}")
print(f"Relevance Score: {scores['relevance']:.4f}")
