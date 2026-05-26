from pathlib import Path

import pandas as pd
import numpy as np

# Project root directory.
# This script is expected to be located in the `scripts/` folder.
BASE_DIR = Path(__file__).resolve().parents[1]

# Prefer storing raw data under data/raw/.
# If the raw Excel file is kept in the project root during local development,
# the fallback path below will also work.
raw_file = BASE_DIR / "data" / "raw" / "Data-Cd.xlsx"
fallback_raw_file = BASE_DIR / "Data-Cd.xlsx"

if raw_file.exists():
    file_path = raw_file
elif fallback_raw_file.exists():
    file_path = fallback_raw_file
else:
    raise FileNotFoundError(
        "Raw data file not found. Please place Data-Cd.xlsx either in "
        "`data/raw/` or in the project root directory."
    )

xls = pd.ExcelFile(file_path)
#processed_Data数据处理
print(xls.sheet_names)
df = pd.read_excel(xls, 'Processed_Data')
print(df.head())
print(df.shape)
print(df.columns)
cd_source = pd.read_excel(xls, "Cd_Source")
print(cd_source.head())
print(cd_source.shape)
print(cd_source.columns)

processed_cleaned_data = df.copy()
processed_cleaned_data[["Number", "Site"]] = processed_cleaned_data[["Number", "Site"]].ffill()
processed_cleaned_data = processed_cleaned_data.rename(columns={
    "SOM (g/kg)": "SOM",
    "Fe (g/kg)": "Fe",
    "T-Cd (mg/kg)": "T_Cd",
    "GP (mg/kg)": "GP",
    "IP (mg/kg)": "IP",
    "BA(GP)": "BA_GP",
    "BA(IP)": "BA_IP"
})
numeric_cols = ["pH", "SOM", "Clay", "Silt", "Sand", "Fe", "T_Cd", "GP", "IP", "BA_GP", "BA_IP"]
for col in numeric_cols:
    processed_cleaned_data[col] = pd.to_numeric(processed_cleaned_data[col], errors='coerce')

processed_cleaned_data["log_T_Cd"] = np.log1p(processed_cleaned_data["T_Cd"])
processed_cleaned_data["log_GP"] = np.log1p(processed_cleaned_data["GP"])
processed_cleaned_data["log_IP"] = np.log1p(processed_cleaned_data["IP"])

print(processed_cleaned_data.head())
print(processed_cleaned_data.info())
print(processed_cleaned_data[numeric_cols].isna().sum())

output_dir = BASE_DIR / "data" / "processed_data"
output_dir.mkdir(parents=True, exist_ok=True)

processed_cleaned_data.to_csv(str(output_dir / "processed_bioaccessibility.csv"), index=False, encoding="utf-8-sig")

#cd_source数据处理
cd_source_cleaned = cd_source.copy()
cd_source_cleaned[["Province"]] = cd_source_cleaned[["Province"]].ffill()
cd_source_cleaned = cd_source_cleaned.rename(columns={
    "Cd (mg/kg)": "Cd",
    "发表时间": "Year",
    "参考文献": "Reference"
})
cd_source_cleaned["Cd"] = (
    cd_source_cleaned["Cd"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .str.strip()
)
cd_source_cleaned["Cd"] = pd.to_numeric(cd_source_cleaned["Cd"], errors='coerce')
cd_source_cleaned["Year"] = pd.to_numeric(cd_source_cleaned["Year"], errors="coerce")
print(cd_source_cleaned.head())
print(cd_source_cleaned.info())
print(cd_source_cleaned.isna().sum())
print(cd_source_cleaned[cd_source_cleaned["Cd"].isna()].head(20))

cd_source_cleaned.to_csv(
    str(output_dir / "cd_source_cleaned.csv"),
    index=False,
    encoding="utf-8-sig"
)

cd_source_analysis = cd_source_cleaned.dropna(subset=["Cd"]).copy()
cd_source_analysis.to_csv(
    str(output_dir / "cd_source_analysis.csv"),
    index=False,
    encoding="utf-8-sig"
)