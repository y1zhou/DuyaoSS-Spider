import json
import pandas as pd
import re
from typing import Dict, List
from pathlib import Path

COLNAMES = [
    "remarks",
    "loss",
    "ping",
    "google_ping",
    "avg_speed",
    "udp_nat_type",
]


def load_crawl_results(json_file: Path):
    with open(json_file, "r") as f:
        x = json.load(f)
    return x


def remove_provider_numbering(provider: str) -> str:
    return re.sub(r"^\d+\.", "", provider)


def csv_to_df(rows: List[List[str]], provider: str, save: Path = None) -> pd.DataFrame:

    df = pd.DataFrame(rows)
    df.columns = COLNAMES[: df.shape[1]]
    df.insert(loc=0, column="group", value=provider)

    if save:
        df.to_csv(save, index=False)
    return df
