from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Sample dataset
dataset = [
    ['blue shirt A', 'blue shirt B'],
    ['blue shirt C', 'blue shirt A'],
    ['blue shirt A', 'blue shirt B']
]

# Convert to one-hot encoded DataFrame
df = pd.DataFrame(dataset)
df = df.stack().str.get_dummies().sum(level=0)

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Show complementary recommendations
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
