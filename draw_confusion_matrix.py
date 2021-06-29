import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[5836, 158, 1286],
         [114, 4368, 68],
         [67, 5, 6892]]

df_cm = pd.DataFrame(array, index=["Glass", "Mask", "Normal"],
                     columns=["Glass", "Mask", "Normal"])
plt.figure(figsize=(3, 3))
sn.heatmap(df_cm, annot=True, fmt=".5g")
plt.show()
