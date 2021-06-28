import os
import shutil
import numpy as np

content = None

with open("/Users/ntdat/Downloads/meta.txt") as f:
    content = f.read()

if not os.path.exists("/Users/ntdat/Downloads/MeGlass_120x120/Glass"):
    os.mkdir("/Users/ntdat/Downloads/MeGlass_120x120/Glass")
if not os.path.exists("/Users/ntdat/Downloads/MeGlass_120x120/Normal"):
    os.mkdir("/Users/ntdat/Downloads/MeGlass_120x120/Normal")

if content is not None:
    for line in content.split("\n"):
        try:
            print(line.split(" "))
            name, is_glasses = np.array(line.split(" "))
            save_dir = "/Users/ntdat/Downloads/MeGlass_120x120/Glass" if is_glasses == '1' else "/Users/ntdat/Downloads/MeGlass_120x120/Normal"
            shutil.move(os.path.join("/Users/ntdat/Downloads/MeGlass_120x120", name), os.path.join(save_dir, name))
        except Exception as ex:
            print(ex)
