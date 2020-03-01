#
# This runs Hoplite on imagenet
#

import os
import subprocess

main_dir = "/home/ndg0068/DNN/imagenet/EntireDataset/"
dirs = {}
count = 0

for dirpath, dirnames, filenames in os.walk(main_dir):
    if count > 26:
        break
    for subdir in dirnames:
        if count > 26:
            break
        # check if enough items
        if (
            len(
                [
                    n
                    for n in os.listdir(os.path.join(dirpath, subdir))
                    if os.path.isfile(n)
                ]
            )
            >= 1000
        ):
            subprocess.Popen(
                "./vgg16.py -o {} -m {} -d {}".format(
                    "output" + count + ".csv", 1000, os.path.join(dirpath, subdir)
                )
            )
            count += 1
