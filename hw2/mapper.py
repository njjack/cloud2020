#!/usr/bin/env python

import sys

monthmap = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}

for line in sys.stdin:
    line = line.strip()
    words = line.split()

    for word in words:
        # get timestamps included in '[', ']'
        if word[0]=='[' :
            date, month, year = word[1:12].split('/')
            hr = word[13:].split(':')[0]
            print '%s-%s-%s T %s:00:00.000\t%s' % (year, monthmap[month], date, hr, 1)


