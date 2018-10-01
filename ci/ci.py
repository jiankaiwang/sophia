# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 09:35:57 2018
@description: Check the link inside the markdown file is available.
@author: JianKai Wang
"""

import os
from os.path import isfile, join, basename, splitext
import sys
import getopt
import logging
import re
import codecs
import requests

# pre-defined variabled
allFiles = ["ml.md", "dl.md", "statistics.md", "uitools.md", "maker.md", "framework.md"]

# help message
def helpMessgae():
    print("""Usage: python ci.py [options]
[options]
-d              the 'data' path of sophia project (necessary)
-h, --help      the help message
    """)
    
# parse opts
def parseOpts(get_opts):
    return(dict((k,v) for k,v in get_opts))
    
# check file path
def checkFInOut(get_opts):
    if '-d' in opts.keys():
        logging.debug(opts['-d'])
        return 1, get_opts['-d']
    logging.debug("not existing parameter")
    return -1, ""

# check necessary file existing
def __checkRemoteRepoExists(filename):
    request = requests.get(\
            'https://raw.githubusercontent.com/jiankaiwang/sophia/master/data/'\
            + filename)
    if request.status_code == 200:
        return True
    else:
        return False

def checkMdFileExist(getPath):
    global allFiles
    
    retFlag = True
    lossFileName = ""
    for file in allFiles:
        tmpPath = join(getPath, file)
        if not isfile(tmpPath):
            remoteExists = __checkRemoteRepoExists(file)
            if remoteExists:
                # would be updated
                retFlag, lossFileName = True, ""
                continue
            else:
                lossFileName = file
                retFlag = False
            break
    return retFlag, lossFileName
    
# get all content is file
def getContentFromFile(fileName):
    content = ""
    with codecs.open(fileName, 'r', 'utf-8') as fin:
        for line in fin:
            content += line.strip()
    return content
    
# check the link is vaild
def checkMdLink(getPath):
    global allFiles
    
    retFlag = True
    lossLinkName = []
    allHtmlFiles = []
    for file in allFiles:
        tmpPath = join(getPath, file)
        logging.debug("check link in " + tmpPath)
        dataContent = getContentFromFile(tmpPath)
        
        # get all []() format in markdown
        searchObj = re.finditer(r'\[.*?\]\(data[/|\\](\S*?)\)', dataContent, re.I)
        if searchObj != None:
            for mobj in searchObj:
                if (not isfile(join(getPath, mobj.group(1)))) \
                    or splitext(basename(mobj.group(1)))[1] != ".html":
                    logging.debug('not html: ' + mobj.group(1))
                    retFlag = False
                    lossLinkName.append(mobj.group(1))
                else:
                    allHtmlFiles.append(mobj.group(1))
                    #logging.debug(mobj.group(1))
    
    return retFlag, lossLinkName, allHtmlFiles

# check whether html files are on markdown or not
def checkHtmlNotListOnMd(getPath, getMdLink):
    
    allPhysicalHtmlFiles = []
    
    for f in os.listdir(getPath):
        filepath = join(getPath, f)
        if isfile(filepath) and filepath.lower().endswith(('.html',)):
            allPhysicalHtmlFiles.append(f)
    
    notListedHtmlData = set(allPhysicalHtmlFiles).difference(set(getMdLink)) 
    return len(notListedHtmlData), notListedHtmlData
        
    
# main entry
logging.basicConfig(level=logging.DEBUG)

opts, args = getopt.getopt(sys.argv[1:], "fhd:", ["help"])
opts = parseOpts(opts)

if len(opts) < 1:
    logging.debug("opts < 1")
    helpMessgae()
elif '--help' in opts.keys() or '-h' in opts.keys():
    logging.debug("opts exists help")
    helpMessgae()
else:
    logging.debug("others")
    
    # check the data directory is vaild
    status, dataPath = checkFInOut(opts)
    if status != 1:
        logging.debug("can not locate data path")
        sys.exit(1)
    
    # check the whole main markdown is available
    status, lossFileName = checkMdFileExist(dataPath)
    if not status:
        logging.debug("markdown file is not exist: " + lossFileName)
        sys.exit(1)
        
    # check the link in markdown file is available
    status, lossMDLink, htmlFiles = checkMdLink(dataPath)
    if not status:
        logging.debug("some links in markdown file are lost: " + ','.join(lossMDLink))
        sys.exit(1)
            
    # check all the link is list
    logging.debug(','.join(htmlFiles))
    objLen, notListHtmlFile = checkHtmlNotListOnMd(dataPath, htmlFiles)
    if objLen > 0:
        logging.debug("some physical files not listed in markdown : " + ','.join(list(notListHtmlFile)))
        sys.exit(1)
    
    # all check points are correct
    sys.exit(0)



