#!/usr/bin/env python

from operator import itemgetter
import sys

current_timestamp = None
current_count = 0

for line in sys.stdin:
    line = line.strip()
    timestamp, count = line.split('\t', 1)

    try:
        count = int(count)
    except ValueError:
        continue

    if current_timestamp == timestamp:
        current_count += count
    else:
        if current_timestamp:
            print '%s\t%s' % (current_timestamp, current_count)
        current_count = count
        current_timestamp = timestamp

if current_timestamp == timestamp:
    print '%s\t%s' % (current_timestamp, current_count)
