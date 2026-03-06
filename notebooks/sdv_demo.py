import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# Create small dataset
real = pd.DataFrame({
    "age": [23, 45, 31, 52, 28, 34, 41, 37, 60, 22],
    "income": [35000, 88000, 54000, 120000, 46000, 62000, 73000, 58000, 140000, 30000],
    "gender": ["F","M","F","M","F","M","F","F","M","F"],
    "approved": [0,1,0,1,0,0,1,0,1,0]
})

# Create metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real)

# Create model
model = GaussianCopulaSynthesizer(metadata)
model.fit(real)

# Generate synthetic data
synthetic = model.sample(20)

print("Synthetic Data Preview:")
print(synthetic.head())

synthetic.to_csv("results/synthetic_demo.csv", index=False)

print("\nReal age stats:")
print(real["age"].describe())

print("\nSynthetic age stats:")
print(synthetic["age"].describe())

print("\nReal approval rate by gender:")
print(real.groupby("gender")["approved"].mean())

print("\nSynthetic approval rate by gender:")
print(synthetic.groupby("gender")["approved"].mean())

print(real.groupby("gender")["approved"].mean())
print(synthetic.groupby("gender")["approved"].mean())
print("\nReal income stats:")
print(real["income"].describe())

print("\nSynthetic income stats:")
print(synthetic["income"].describe())

import matplotlib.pyplot as plt

real["income"].hist(alpha=0.5, label="Real")
synthetic["income"].hist(alpha=0.5, label="Synthetic")
plt.legend()
plt.title("Income Distribution Comparison")
plt.show()

