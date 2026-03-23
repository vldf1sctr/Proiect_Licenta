import pandas as pd
import json

df = pd.read_csv('/Users/vlad/Downloads/prognoza cursului de schimb prin automatizarea analizei statistice a seriilor de timp - bulk (1).csv')

df = df.drop(columns=['id'])

records = df.to_dict(orient='records')

with open('data.json', 'w') as f:
    json.dump(records, f, indent=2)