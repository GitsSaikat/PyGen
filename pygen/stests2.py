import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


base_data = {
    
}


metrics = ['CodeBLEU', 'Identifier_Match', 'Ngram_Match', 'Weighted_Ngram', 'Token_Match', 'Dataflow_Match', 'Syntax_Match']


np.random.seed(0)  
data = {
    'Model': [],
    'CodeBLEU': [],
    'Identifier_Match': [],
    'Ngram_Match': [],
    'Weighted_Ngram': [],
    'Token_Match': [],
    'Dataflow_Match': [],
    'Syntax_Match': []
}

for model, base_scores in base_data.items():
    for _ in range(60):  
        data['Model'].append(model)
        for i, metric in enumerate(metrics):
           
            score = max(0, min(1, np.random.normal(base_scores[i], 0.02))) 
            data[metric].append(score)

# Convert to DataFrame
df = pd.DataFrame(data)


for metric in metrics:
    print(f"\nOne-Way ANOVA for {metric}")
    model = ols(f'{metric} ~ C(Model)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # Tukey's HSD Post-Hoc Test
    print(f"\nTukey's HSD Post-Hoc Test for {metric}")
    tukey = pairwise_tukeyhsd(endog=df[metric], groups=df['Model'], alpha=0.05)
    print(tukey)
