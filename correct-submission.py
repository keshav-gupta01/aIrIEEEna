import pandas as pd
import numpy as np

df = pd.read_csv("FINAL.csv")
df.head()
df["Class"] = (df["Class"] > 0.5).astype(int)
df["ID"] = np.arange(1, len(df) + 1)
from google.colab import files
files.download("FINAL_corrected.csv")