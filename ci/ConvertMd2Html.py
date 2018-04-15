# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 09:35:57 2018
@author: JianKai Wang
"""

import os
from os.path import isfile, join, basename, splitext
from pathlib import Path
import sys
import getopt
import logging
import subprocess

# help message
def helpMessgae():
    print("""Usage: python ConvertMd2Html.py [options]
[options]
-e              the path of sophia project (necessary)
-i              input directory path (necessary)
-o              output directory path (necessary)
-h, --help      the help message
    """)

# parse opts
def parseOpts(get_opts):
    return(dict((k,v) for k,v in get_opts))
    
# check file path
def checkFInOut(get_opts):
    if '-e' in opts.keys() and '-i' in opts.keys() and '-o' in opts.keys():
        logging.debug(opts['-i'])
        if Path(opts['-e']).is_dir() and Path(opts['-i']).is_dir() and Path(opts['-o']).is_dir():
            return 1, get_opts['-e'], get_opts['-i'], get_opts['-o']
    logging.debug("not existing parameter")
    return -1, "", "", ""

# convert from *.md to *.html
def getConvertList(fin, fout):
    onlyfiles = {}
    whiteList = ["model.md","framework.md","maker.md","uitools.md","statistics.md"]
    
    for f in os.listdir(fin):
        filepath = join(fin, f)
        if isfile(filepath) and filepath.lower().endswith(('.md',)) and f not in whiteList:
            name = splitext(basename(filepath))[0]
            onlyfiles.setdefault(f,name + ".html")
    
    return onlyfiles
    
# 
def convertToHtml(fexec, fin, fout, getfiles):
    templateHtml = join(fexec,"ci","template","default.html")
    headerHtml = join(fexec,"ci","template","header.html")
    
    for all_md in getfiles.keys():
        bashCommand = """pandoc +RTS -K512m -RTS {} --to html4 --from markdown+autolink_bare_uris+ascii_identifiers+tex_math_single_backslash+smart --output {} --email-obfuscation none --self-contained --standalone --section-divs --template {} --no-highlight --variable highlightjs=1 --variable "theme:bootstrap" --mathjax --variable "mathjax-url:https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" --include-in-header {} """.format(join(fin,all_md), join(fin,getfiles[all_md]), templateHtml, headerHtml)
         
        logging.debug(bashCommand)
        try:        
            process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
            output, error = process.communicate()
            print('Output:{},Error:{}'.format(output.decode('utf-8'), error.decode('utf-8')))
        except:
            print("Failure to convert markdown {}.".format(all_md))
    
    
# main entry
logging.basicConfig(level=logging.WARNING)

opts, args = getopt.getopt(sys.argv[1:], "fhe:i:o:", ["help"])
opts = parseOpts(opts)

if len(opts) < 1:
    logging.debug("opts < 1")
    helpMessgae()
elif '--help' in opts.keys() or '-h' in opts.keys():
    logging.debug("opts exists help")
    helpMessgae()
else:
    logging.debug("others")
    flag, fexec, fin, fout = checkFInOut(opts)
    if flag != 1:
        print("Error: Parameter is not valid.")
        helpMessgae()
    else:
        os.chdir(fin)
        logging.debug("precheck ok")
        fileNameList = getConvertList(fin, fout)
        convertToHtml(fexec, fin, fout, fileNameList)
    
    
    
    
    
    
    
    
    
    
    