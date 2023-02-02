import sys, os, io, readline
import re
from tools import once

@once
def getkeys():
   cname = ''
   results = {'':[]}
   comments = re.compile('#(.+)$')
   section = re.compile('^([\w\._]+):$')
   
   with open('.keys', 'r') as f:
      lines = f.readlines()
      for line in lines:
         t = section.match(line)
         if t is not None:
            # nonlocal cname #pylint: disable
            cname = t.group(1)
            results[cname] = []
            # cname = name
         elif len(line) > 1:
            line = comments.split(line)[0].strip()
            if len(line) > 0:
               results[cname].append(line)
      return results      
         