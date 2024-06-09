import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('titanic.csv', index_col=0)

# One-Hot Encoding
df_encoded = pd.get_dummies(df)

# Apply Apriori algorithm
freq_items = apriori(df_encoded, min_support=0.005, use_colnames=True, verbose=1)
print(freq_items.head(7))

# Generate association rules
rules = association_rules(freq_items, metric="confidence", min_threshold=0.8)
print(rules.head())

# Filter rules related to 'Survived_Yes'
survived_rules = rules[(rules['antecedents'].apply(lambda x: 'Survived_Yes' in x) |
                        rules['consequents'].apply(lambda x: 'Survived_Yes' in x))]
print(survived_rules)

# Check if we have any survived rules
if not survived_rules.empty:
    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(survived_rules['support'], survived_rules['confidence'], alpha=0.5)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence for Survived Rules')

    plt.subplot(1, 2, 2)
    plt.scatter(survived_rules['support'], survived_rules['lift'], alpha=0.5)
    plt.xlabel('Support')
    plt.ylabel('Lift')
    plt.title('Support vs Lift for Survived Rules')

    plt.tight_layout()
    plt.show()

    # Fit a line to the lift vs confidence
    fit = np.polyfit(survived_rules['lift'], survived_rules['confidence'], 1)
    fit_fn = np.poly1d(fit)
    plt.plot(survived_rules['lift'], survived_rules['confidence'], 'yo', survived_rules['lift'], fit_fn(survived_rules['lift']))
    plt.xlabel('Lift')
    plt.ylabel('Confidence')
    plt.title('Lift vs Confidence for Survived Rules')
    plt.show()
else:
    print("No association rules related to survival found.")

# Processing 90 combinations | Sampling itemseProcessing 210 combinations | Sampling itemsProcessing 180 combinations | Sampling itemsProcessing 40 combinations | Sampling itemset size 5
#     support      itemsets
# 0  0.147660   (Class_1st)
# 1  0.129487   (Class_2nd)
# 2  0.320763   (Class_3rd)
# 3  0.402090  (Class_Crew)
# 4  0.213539  (Sex_Female)
# 5  0.786461    (Sex_Male)
# 6  0.950477   (Age_Adult)
#     antecedents  ... zhangs_metric
# 0   (Class_1st)  ...      0.037128
# 1   (Class_2nd)  ...     -0.041697
# 2   (Class_3rd)  ...     -0.093712
# 3  (Class_Crew)  ...      0.322047
# 4  (Class_Crew)  ...      0.082827

# [5 rows x 10 columns]
#                                antecedents  ... zhangs_metric
# 9                           (Survived_Yes)  ...     -0.046906
# 11                 (Class_1st, Sex_Female)  ...      0.714898
# 15               (Survived_Yes, Class_1st)  ...      0.022665
# 17                 (Sex_Female, Class_2nd)  ...      0.663777
# 22                  (Class_2nd, Age_Child)  ...      0.684428
# 28               (Survived_Yes, Class_3rd)  ...     -0.115847
# 31                (Class_Crew, Sex_Female)  ...      0.635147
# 36              (Survived_Yes, Class_Crew)  ...      0.145645
# ...      0.652022
# 56      (Sex_Female, Age_Child, Class_2nd)  ...      0.680987
# 64   (Survived_Yes, Sex_Female, Class_3rd)  ...     -0.115763
# 69     (Survived_Yes, Sex_Male, Class_3rd)  ...     -0.107163
# 71  (Survived_Yes, Class_Crew, Sex_Female)  ...      0.049977
# 72     (Age_Adult, Class_Crew, Sex_Female)  ...      0.635147
# 73                (Class_Crew, Sex_Female)  ...      0.665243
# 77   (Survived_Yes, Age_Adult, Class_Crew)  ...      0.145645
# 78    (Survived_Yes, Class_Crew, Sex_Male)  ...      0.054256
# 79              (Survived_Yes, Class_Crew)  ...      0.181174

# [27 rows x 10 columns]