import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data = {

}

df = pd.DataFrame(data)


correlation_pairs = [
    ('Coherence Without Context', 'Coherence With Context'),
    ('Coherence Without Context', 'Fluency Score'),
    ('Coherence With Context', 'Fluency Score')
]

pearson_results = []
spearman_results = []

for pair in correlation_pairs:
    var1, var2 = pair
    pearson_corr, pearson_p = pearsonr(df[var1], df[var2])
    spearman_corr, spearman_p = spearmanr(df[var1], df[var2])
    
    pearson_results.append({
        'Variable Pair': f"{var1} & {var2}",
        "Pearson's r": round(pearson_corr, 3),
        'Pearson p-value': round(pearson_p, 3)
    })
    
    spearman_results.append({
        'Variable Pair': f"{var1} & {var2}",
        "Spearman's rho": round(spearman_corr, 3),
        'Spearman p-value': round(spearman_p, 3)
    })

pearson_df = pd.DataFrame(pearson_results)
spearman_df = pd.DataFrame(spearman_results)

print("\nPearson's r Correlation Results:")
print(pearson_df)

print("\nSpearman's rho Correlation Results:")
print(spearman_df)

sns.set(style="whitegrid")

pairplot_data = df[['Coherence Without Context', 'Coherence With Context', 'Fluency Score']]
sns.pairplot(pairplot_data, kind='scatter', diag_kind='kde', corner=True)
plt.suptitle('Pairwise Scatter Plots with Pearson and Spearman Correlations', y=1.02)
plt.show()

correlation_matrix = pairplot_data.corr(method='pearson')

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f")
plt.title("Pearson's Correlation Heatmap")
plt.show()
