# A hacky wrapper for running celerotation code on a list of KIC stars.

import numpy as np
import subprocess

kics = np.loadtxt("names.txt")
print(kics)

for star in kics:
    print("python ruth_demo.py {}".format(int(star)))
    subprocess.call("python ruth_demo.py {}".format(int(star)), shell=True)
