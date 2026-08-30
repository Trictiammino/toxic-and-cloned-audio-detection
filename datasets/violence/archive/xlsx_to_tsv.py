import pandas as pd

df = pd.read_excel("VSD.xlsx")

df.to_csv("VSD.tsv", sep="\t", index=False)