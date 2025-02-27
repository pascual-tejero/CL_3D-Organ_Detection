import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ["Single dataset\n(AbdomenCT-1K)", "Single dataset\n(WORD)", "ANCL", "CL Replay"]
datasets = ["AbdomenCT-1K", "WORD"]
values = np.array([
    [0.813, 0.001],  # AbdomenCT-1K performance
    [0.004, 0.539],  # WORD performance
    [0.269, 0.555],  # ANCL  
    [0.772, 0.634]   # CL replay
])

# Define x-axis positions
x = np.arange(len(categories))  # Base positions
width = 0.25  # Reduce width slightly to avoid overlap

fig, ax = plt.subplots(figsize=(7, 5), dpi=175)  # High DPI for print quality
bars1 = ax.bar(x - width * 0.60, values[:, 0], width, label="AbdomenCT-1K", color='dodgerblue', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width * 0.60, values[:, 1], width, label="WORD", color='orange', edgecolor='black', linewidth=0.8)

# Add value labels above bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=14, fontweight='bold')

# Labels & Titles
ax.set_xlabel("Training Method", fontsize=16, fontweight='bold', labelpad=12)
ax.set_ylabel("Performance (mAP)", fontsize=16, fontweight='bold', labelpad=12)
ax.set_title("Performance Across Datasets (Test Set)", fontsize=18, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14)
ax.set_ylim(0, 1)  # Set y-axis limit

# Customize ticks
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

# Legend
ax.legend(fontsize=14, loc="upper center", frameon=True, edgecolor='black', ncol=2)

# Grid for readability
ax.yaxis.grid(True, linestyle="--", alpha=0.5)

# Adjust layout for better spacing
plt.tight_layout()

# Show plot
plt.show()
