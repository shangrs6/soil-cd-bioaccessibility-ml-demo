from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Project root directory.
# This script is expected to be located in the `scripts/` folder.
BASE_DIR = Path(__file__).resolve().parents[1]

processed_path = BASE_DIR / "data" / "processed_data" / "processed_bioaccessibility.csv"
cd_source_path = BASE_DIR / "data" / "processed_data" / "cd_source_analysis.csv"

processed_bio = pd.read_csv(processed_path, encoding="utf-8-sig")
cd_source = pd.read_csv(cd_source_path, encoding="utf-8-sig")

fig_dir = BASE_DIR / "figures"      # 定义保存图片的文件夹路径
fig_dir.mkdir(parents=True, exist_ok=True) # 如果文件夹不存在，则创建 

print(processed_bio.head())
print(cd_source.head())
print(processed_bio.shape)              # 检查 processed_bio 数据有多少行多少列
print(cd_source.shape)             # 检查 cd_source 数据有多少行多少列  

print(processed_bio["BA_GP"].describe())
print(processed_bio["Method"].value_counts())
print(processed_bio["Type"].value_counts())

# 绘制 BA_GP 的分布直方图
plt.figure(figsize=(7, 5))
plt.hist(processed_bio["BA_GP"], bins=25, edgecolor="black")
plt.xlabel("Cd bioaccessibility in gastric phase, BA_GP (%)")
plt.ylabel("Frequency")
plt.title("Distribution of Cd Bioaccessibility in Gastric Phase")
plt.tight_layout()
plt.savefig(fig_dir / "ba_gp_distribution.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(processed_bio["T_Cd"], bins=30, edgecolor="black")
plt.xlabel("Total Cd concentration (mg/kg)")
plt.ylabel("Frequency")
plt.title("Distribution of Total Cd Concentration")
plt.tight_layout()
plt.savefig(fig_dir / "total_cd_distribution.png", dpi=300)
plt.show()
# 绘制 log1p(T_Cd) 的分布直方图
plt.figure(figsize=(7, 5))
plt.hist(processed_bio["log_T_Cd"], bins=30, edgecolor="black")
plt.xlabel("log1p(Total Cd concentration)")
plt.ylabel("Frequency")
plt.title("Log-transformed Distribution of Total Cd")
plt.tight_layout()
plt.savefig(fig_dir / "log_total_cd_distribution.png", dpi=300)
plt.show()
# 绘制 log1p(GP) 的分布直方图
method_order = processed_bio["Method"].value_counts().index.tolist()
data_by_method = [processed_bio.loc[processed_bio["Method"] == m, "BA_GP"] for m in method_order]
plt.figure(figsize=(8, 5))
plt.boxplot(data_by_method, labels=method_order, showfliers=True)
plt.xlabel("In vitro digestion method")
plt.ylabel("BA_GP (%)")
plt.title("Cd Bioaccessibility by In Vitro Method")
plt.tight_layout()
plt.savefig(fig_dir / "ba_gp_by_method.png", dpi=300)
plt.show()
# 绘制 log1p(IP) 的分布直方图
type_order = processed_bio["Type"].value_counts().index.tolist()
data_by_type = [processed_bio.loc[processed_bio["Type"] == t, "BA_GP"] for t in type_order]
plt.figure(figsize=(9, 5))
plt.boxplot(data_by_type, labels=type_order, showfliers=True)
plt.xticks(rotation=20, ha="right")
plt.xlabel("Soil source type")
plt.ylabel("BA_GP (%)")
plt.title("Cd Bioaccessibility by Soil Source Type")
plt.tight_layout()
plt.savefig(fig_dir / "ba_gp_by_type.png", dpi=300)
plt.show()
# 绘制 log1p(T_Cd) 的分布直方图
type_order_cd = cd_source["Type"].value_counts().index.tolist()
data_cd_by_type = [cd_source.loc[cd_source["Type"] == t, "Cd"] for t in type_order_cd]
plt.figure(figsize=(9, 5))
plt.boxplot(data_cd_by_type, labels=type_order_cd, showfliers=True)
plt.yscale("log")
plt.xticks(rotation=20, ha="right")
plt.xlabel("Soil source type")
plt.ylabel("Cd concentration (mg/kg, log scale)")
plt.title("Soil Cd Concentration by Source Type")
plt.tight_layout()
plt.savefig(fig_dir / "cd_concentration_by_type_logscale.png", dpi=300)
plt.show()
# 绘制相关性矩阵
num_cols = ["pH", "SOM", "Clay", "Silt", "Sand", "Fe", "log_T_Cd", "BA_GP"]

corr = processed_bio[num_cols].corr()

plt.figure(figsize=(8, 6))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
plt.yticks(range(len(num_cols)), num_cols)

for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

plt.title("Correlation Matrix of Numeric Variables")
plt.tight_layout()
plt.savefig(fig_dir / "correlation_matrix_numeric_variables.png", dpi=300)
plt.show()
# 绘制 log1p(T_Cd) 与 BA_GP 的散点图
plt.figure(figsize=(7, 5))
plt.scatter(processed_bio["log_T_Cd"], processed_bio["BA_GP"], alpha=0.7, edgecolor="black")
plt.xlabel("log1p(Total soil Cd concentration)")
plt.ylabel("BA_GP (%)")
plt.title("Relationship between Total Cd and Gastric-phase Bioaccessibility")
plt.tight_layout()
plt.savefig(fig_dir / "ba_gp_vs_log_total_cd.png", dpi=300)
plt.show()
