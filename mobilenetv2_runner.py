#
# This runs Hoplite on imagenet
#

import os
import subprocess

#TODO change dir for talon
main_dir = "/storage/scratch2/share/pi_ndg0068"
dirs = {}
count = 0

MAX = 100

for dirpath, dirnames, filenames in os.walk(main_dir):
    if count > MAX:
        break
    for subdir in dirnames:
        if count > MAX:
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
                "./.mobilenetv2.py -o {} -m {} -d {}".format(
                    "output" + count + ".csv", 1000, os.path.join(dirpath, subdir)
                )
            )
            count += 1
