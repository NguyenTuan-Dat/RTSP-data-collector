import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[5332, 75, 1873],
         [159, 4344, 47],
         [704, 113, 6147]]

df_cm = pd.DataFrame(array, index=["Glass", "Mask", "Normal"],
                     columns=["Glass", "Mask", "Normal"])
plt.figure(figsize=(3, 3))
sn.heatmap(df_cm, annot=True, fmt=".5g")
plt.show()
