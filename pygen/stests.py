import pandas as pd
import numpy as np
from pingouin import cronbach_alpha, intraclass_corr
import statsmodels.stats.inter_rater as irr
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

data = {

}

df = pd.DataFrame(data)
print("Sample Documentation Review Scores:")
print(df)

def calculate_cronbach_alpha(df, metric_columns):
    alpha_results = {}
    for metric in metric_columns:
        scores = df[metric].to_frame()
        alpha, _ = cronbach_alpha(scores)
        alpha_results[metric] = round(alpha, 3)
    return alpha_results

def calculate_icc(df, metric_columns):
    icc_results = {}
    for metric in metric_columns:
        df_icc_long = df[['Item', metric]].melt(id_vars='Item', var_name='Rater', value_name='Score')
        icc = intraclass_corr(data=df_icc_long, targets='Item', raters='Rater', ratings='Score')
        icc_2_1 = icc[icc['Type'] == 'ICC2']['ICC'].values[0]
        icc_results[metric] = round(icc_2_1, 3)
    return icc_results

def calculate_fleiss_kappa(df, metric_columns, categories=[1, 2, 3]):
    kappa_results = {}
    for metric in metric_columns:
        bins = [0, 7, 8, 10]
        labels = [1, 2, 3]
        df_fleiss_metric = df[['Item', 'Rater1', 'Rater2', 'Rater3', 'Rater4']].copy()
        for rater in ['Rater1', 'Rater2', 'Rater3', 'Rater4']:
            df_fleiss_metric[rater] = pd.cut(df_fleiss_metric[rater], bins=bins, labels=labels, include_lowest=True).astype(int)
        
        fleiss_matrix = []
        for _, row in df_fleiss_metric.iterrows():
            counts = [sum(row == category) for category in categories]
            fleiss_matrix.append(counts)
        fleiss_matrix = np.array(fleiss_matrix)
        
        kappa = irr.fleiss_kappa(fleiss_matrix, method='fleiss')
        kappa_results[metric] = round(kappa, 3)
    return kappa_results

def calculate_average_agreement(df, metric_columns):
    agreement_results = {}
    for metric in metric_columns:
        def pairwise_agreement(row):
            raters = row[1:]
            total_pairs = 0
            agreements = 0
            for r1, r2 in combinations(raters, 2):
                total_pairs += 1
                if r1 == r2:
                    agreements += 1
            return agreements / total_pairs
        
        df_metric = df[['Item', 'Rater1', 'Rater2', 'Rater3', 'Rater4']].copy()
        df_metric['Pairwise_Agreement'] = df_metric.apply(pairwise_agreement, axis=1)
        average_agreement = df_metric['Pairwise_Agreement'].mean()
        agreement_results[metric] = round(average_agreement, 3)
    return agreement_results

metrics = ['Clarity', 'Completeness', 'Structure', 'Readability']
for metric in metrics:
    for rater in ['Rater1', 'Rater2', 'Rater3', 'Rater4']:
        df[f"{metric}_{rater}"] = df[rater]

cronbach_results = {}
icc_results = {}
fleiss_kappa_results = {}
average_agreement_results = {}

for metric in metrics:
    metric_columns = [f"{metric}_Rater1", f"{metric}_Rater2", f"{metric}_Rater3", f"{metric}_Rater4"]
    df_metric = df[['Item'] + metric_columns].copy()
    
    alpha, _ = cronbach_alpha(df_metric[metric_columns])
    cronbach_results[metric] = round(alpha, 3)
    
    df_icc_long = df_metric.melt(id_vars='Item', var_name='Rater', value_name='Score')
    icc = intraclass_corr(data=df_icc_long, targets='Item', raters='Rater', ratings='Score')
    icc_2_1 = icc[icc['Type'] == 'ICC2']['ICC'].values[0]
    icc_results[metric] = round(icc_2_1, 3)
    
    bins = [0, 7, 8, 10]
    labels = [1, 2, 3]
    df_fleiss_metric = df_metric[['Item'] + metric_columns].copy()
    for rater in metric_columns:
        df_fleiss_metric[rater] = pd.cut(df_fleiss_metric[rater], bins=bins, labels=labels, include_lowest=True).astype(int)
    
    fleiss_matrix = []
    for _, row in df_fleiss_metric.iterrows():
        counts = [sum(row == category) for category in labels]
        fleiss_matrix.append(counts)
    fleiss_matrix = np.array(fleiss_matrix)
    
    kappa = irr.fleiss_kappa(fleiss_matrix, method='fleiss')
    fleiss_kappa_results[metric] = round(kappa, 3)
    
    def pairwise_agreement(row):
        raters = row[1:]
        total_pairs = 0
        agreements = 0
        for r1, r2 in combinations(raters, 2):
            total_pairs += 1
            if r1 == r2:
                agreements += 1
        return agreements / total_pairs
    
    df_metric_agreement = df_metric.copy()
    df_metric_agreement['Pairwise_Agreement'] = df_metric_agreement.apply(pairwise_agreement, axis=1)
    average_agreement = df_metric_agreement['Pairwise_Agreement'].mean()
    average_agreement_results[metric] = round(average_agreement, 3)

summary_data = {
    'Metric': metrics,
    'Cronbach Alpha': [cronbach_results[m] for m in metrics],
    'ICC (2,1)': [icc_results[m] for m in metrics],
    'Fleiss Kappa': [fleiss_kappa_results[m] for m in metrics],
    'Average Agreement': [average_agreement_results[m] for m in metrics]
}

summary_df = pd.DataFrame(summary_data)
print("\nReliability and Agreement Metrics Summary:")
print(summary_df)

for metric in metrics:
    print(f"\n--- {metric} ---")
    print(f"Cronbach's Alpha: {cronbach_results[metric]}")
    print(f"ICC (2,1): {icc_results[metric]}")
    print(f"Fleiss' Kappa: {fleiss_kappa_results[metric]}")
    print(f"Average Agreement: {average_agreement_results[metric]}")
