#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Normalizes dictionary data in the json file passed as first arg.
See example data format in this directory.
"""
import json
import sys

with open(file=sys.argv[1], mode="r") as f:
    keyboard = json.load(fp=f)

max_val = max(keyboard.values())

for key in keyboard.keys():
    keyboard[key] = keyboard[key] / max_val

with open(file=sys.argv[1], mode="w") as f:
    json.dump(obj=keyboard, fp=f)
