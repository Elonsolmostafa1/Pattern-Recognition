# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import splitfolders

male_data = "Dataset\\Males\\"
female_data = "Dataset\\Females\\"

splitfolders.ratio(male_data,"Data_split",seed=1337,ratio=(0.8,0.0,0.2))
splitfolders.ratio(female_data,"Data_split",seed=1337,ratio=(0.8,0.0,0.2))
