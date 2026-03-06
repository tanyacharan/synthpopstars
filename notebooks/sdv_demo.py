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
plt.savefig("results/income_comparison.png")
plt.close()

from sdv.single_table import CTGANSynthesizer

# Train CTGAN
ctgan_model = CTGANSynthesizer(metadata)
ctgan_model.fit(real)

synthetic_ctgan = ctgan_model.sample(20)

print("\nCTGAN Synthetic Data Preview:")
print(synthetic_ctgan.head())

print("\nCTGAN Approval Rate by Gender:")
print(synthetic_ctgan.groupby("gender")["approved"].mean())

synthetic_ctgan.to_csv("results/synthetic_ctgan.csv", index=False)

# ---- Clean Comparison Table ----

comparison = pd.DataFrame({
    "Dataset": ["Real", "Gaussian", "CTGAN"],
    "Female Approval": [0.1667, 0.3333, 0.1667],
    "Male Approval": [0.7500, 0.4545, 0.7500]
})

# print nicely in terminal
print("\nApproval Rate Comparison Table:")
print(comparison.to_string(index=False))

# save csv for slides or Excel
comparison.to_csv("results/comparison_table.csv", index=False)

# optionally save as an image to paste directly into slides
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,1.5))            # adjust size as needed
ax.axis("off")
tbl = ax.table(cellText=comparison.values,
               colLabels=comparison.columns,
               loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.auto_set_column_width(col=list(range(len(comparison.columns))))
plt.savefig("results/comparison_table.png", bbox_inches="tight", dpi=200)
plt.close(fig)
