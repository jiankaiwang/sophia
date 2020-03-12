#!/bin/python3

from os.path import exists
from logging import Logger
from os import remove, rename
import codecs
import argparse
import re

# In[]

def main(INPUTFILE, OUTPUTFILE, replaced=False):
  """main: the rewriting flow of parsing lines in the README.md file"""
  if not exists(INPUTFILE): raise Exception("Can't find inputfile.")
  if exists(OUTPUTFILE):
    Logger.warning("Remove the existing file.")    
    remove(OUTPUTFILE)
  
  text = ""
  with open(INPUTFILE, "r") as fin:
    for line in fin:
      text += line
  
  patterns = r"]\(\S+\)"
  match = re.findall(patterns, text)
  
  for m in match:
    if m[:6] != "](http":
      replaced = "](https://github.com/jiankaiwang/sophia/blob/master/" + m[2:]
      text = text.replace(m, replaced)
  
  with codecs.open(OUTPUTFILE, "w", "UTF-8") as fout:
    fout.write(text)
    
  if replaced:
    remove(INPUTFILE)
    rename(OUTPUTFILE, INPUTFILE)

# In[]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', type=str, default='../README.md')
  parser.add_argument('--output', type=str, default='../P_README.md')
  FLAGS, unparsed = parser.parse_known_args()
  
  INPUTFILE = FLAGS.input
  OUTPUTFILE = FLAGS.output
  
  main(INPUTFILE, OUTPUTFILE, replaced=True)
