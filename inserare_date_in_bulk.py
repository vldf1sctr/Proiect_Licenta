import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()
CSV_PATH = os.getenv("CSV_PATH")

df = pd.read_csv(CSV_PATH)
df.rename(columns={"Valoare in RON (EUR)": "eur_ron", "Valoare in RON (USD)": "usd_ron", "IRCC (%)": "ircc", "ROBOR (%)": "robor", "Data": "data"}, inplace=True)
df["data"] = pd.to_datetime(df["data"], format="%d.%m.%Y")
df.sort_values(by="data", inplace=True)
df.drop(columns=["ID"], inplace=True)


df["data"] = df["data"].dt.strftime(date_format="%Y-%m-%d")
df.to_json("data.json", orient="records", indent=2)

print(df.head())