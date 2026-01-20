#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pathlib import Path
from typing import Dict, List

COLUMN_MAPPING = {
    "日期": "date",
}

def build_synthetic_contract_data(
    data_dir: Path,
    window: int = 2,
) -> pd.DataFrame:
    all_data = []

    for file_path in data_dir.glob("*.xlsx"):
        df = pd.read_excel(file_path)
        df = df.rename(columns=COLUMN_MAPPING)

        contract_code = file_path.stem[-4:]
        delivery_month = parse_contract_month(contract_code)
        target_months = get_pre_delivery_months(delivery_month, window)

        df["date"] = pd.to_datetime(df["date"])
        all_data.append(df[df["date"].dt.month.isin(target_months)])

    return pd.concat(all_data, ignore_index=True)

