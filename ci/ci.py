# -*- coding:utf-8 -*-

from os import listdir
from os.path import isfile, join
import subprocess
import re

mypath = "../data"

# *.rmd and *.md
def checkFileExt(name):
    if "md" in name:
        return True
    else:
        return False
def transform2html(file_name):
    bashCommand = 'Rscript ./gmd.r {}'.format(mypath + "/" + file_name)
    try:        
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        searchObj = re.search(r'Error', output.decode('utf-8'), re.M | re.I)
        if len(searchObj.group()) > 4:
            return False
        else:
            return True
    except:
        return False
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and checkFileExt(join(mypath, f))]
for item in onlyfiles:
    if not transform2html(item):
        print("Error in rendering file " + item)