import warnings
warnings.filterwarnings('ignore')

import pickle
from collections import defaultdict
import os

# Path to the pickled fitted data
fit_file_path = 'datas/datas_fit.pkl'

# Load the existing fitted data
with open(fit_file_path, 'rb') as f:
    datas_fit = pickle.load(f)

# Dictionary to group each dataset's values by date
grouped_data = defaultdict(list)

# Populate grouped_data with the relevant attributes
for data in datas_fit:
    row = [
        data.level,
        data.sigma2,
        data.l1,
        data.l2,
        data.l3,
        data.tau2
    ]
    grouped_data[data.date].append(row)

latex_table = (
    "\\begin{table}[ht]\n"
    "\\centering\n"
    "\\begin{tabular}{llrrrrr}\n"   # 7 columns (2 'l' + 5 'r')
    "\\hline\n"
    "{Date} & {Level} & $\\sigma^2$ & $l_1$ & $l_2$ & $l_3$ & $\\tau^2$ \\\\\n"
    "\\hline\n"
)

# Populate the table rows, grouped by date
for date, rows in sorted(grouped_data.items()):
    latex_table += f"\\multirow{{{len(rows)}}}{{*}}{{{date}}} "
    for i, row in enumerate(rows):
        if i == 0:
            latex_table += f"& {row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} \\\\\n"
        else:
            latex_table += f" & {row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} \\\\\n"
    latex_table += "\\hline\n"
    
latex_table += (
    "\\end{tabular}\n"
    "\\caption{}\n"
    "\\label{tab:cov_est}\n"
    "\\end{table}"
)

# Ensure the results directory exists, then save the LaTeX table to a file
results_path = "results/table.txt"
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, "w") as f:
    f.write(latex_table)
