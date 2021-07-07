import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[7487, 4, 488],
         [53, 6274, 38],
         [232, 33, 7452]]

df_cm = pd.DataFrame(array, index=["Glass", "Mask", "Normal"],
                     columns=["Glass", "Mask", "Normal"])
plt.figure(figsize=(3, 3))
sn.heatmap(df_cm, annot=True, fmt=".5g")
plt.show()
