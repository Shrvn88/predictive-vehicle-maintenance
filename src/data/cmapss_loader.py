import pandas as pd
from pathlib import Path
from ..config import RAW_DATA

# CMAPSS column definition (NASA spec)
COL_NAMES = (
    ["unit", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

def load_cmapss(fd="FD001"):
    """
    Loads CMAPSS dataset and computes RUL.

    Returns:
        train_df, test_df
    """

    base = RAW_DATA / "CMAPSS"

    train_path = base / f"train_{fd}.txt"
    test_path = base / f"test_{fd}.txt"
    rul_path = base / f"RUL_{fd}.txt"

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])
    

    train_df.columns = COL_NAMES
    test_df.columns = COL_NAMES

    # -------- TRAIN RUL --------
    max_cycles = train_df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]

    train_df = train_df.merge(max_cycles, on="unit")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    train_df["RUL"] = train_df["RUL"].clip(upper=125)
    train_df.drop("max_cycle", axis=1, inplace=True)

    # -------- TEST RUL --------
    max_test_cycles = test_df.groupby("unit")["cycle"].max().reset_index()
    max_test_cycles.columns = ["unit", "max_cycle"]

    test_df = test_df.merge(max_test_cycles, on="unit")
    test_df = test_df.merge(rul_df, left_on="unit", right_index=True)

    test_df["RUL"] = test_df["RUL"] + test_df["max_cycle"] - test_df["cycle"]
    test_df["RUL"] = test_df["RUL"].clip(upper=125)
    test_df.drop("max_cycle", axis=1, inplace=True)

    return train_df, test_df

print(load_cmapss()[0].shape)