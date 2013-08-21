# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:12:28 2013

@author: davidsabater
"""

import sys, csv

input_file = sys.argv[1]
output_file = sys.argv[2]

reader = csv.reader( open( input_file ))
o = csv.writer(open( output_file, 'wb' ))

for line in reader:
    print line[0]
    print line[1]
    if line[0] > '0':
        Survived = 1
    else:
        Survived = 0
    o.writerow([Survived,line[1]])