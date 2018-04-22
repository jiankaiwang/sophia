# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:04:40 2018
@author: JianKai Wang
"""

# coding: utf-8

import sys
import getopt
import logging
import requests
import json

# help message
def helpMessgae():
    print("""Usage: python github_api.py [options]
[options]
-o              operations, e.g. master-lastest-commit (mlc) (necessary)
-p              the repository name, e.g. jiankaiwang/sophia (necessary)
-h, --help      the help message
    """)

# parse opts
def parseOpts(get_opts):
    return(dict((k,v) for k,v in get_opts))
    
#################################################
# the github api operation
#################################################
def api_fetchMasterLastestCommit(repository_name):    
    res_url = "https://api.github.com/repos/{}/commits".format(repository_name)
    try:
        req_raw = requests.get(res_url)
        if req_raw.status_code != 200:
            logging.warning("Repository information from github api is error.")
            return {"status_code":-1}

        req_raw = {"status_code":200, "raw_info":req_raw}
        return(req_raw)
    except:
        logging.warning("Parsing the information fetching from github api is error.")
        return {"status_code":-1}
    
#################################################

# main entry
allowedOperations = ["master-lastest-commit", "mlc"]
logging.basicConfig(level=logging.WARNING)

opts, args = getopt.getopt(sys.argv[1:], "fho:p:", ["help"])
opts = parseOpts(opts)

if len(opts) < 1:
    logging.debug("opts < 1")
    helpMessgae()
elif '--help' in opts.keys() or '-h' in opts.keys():
    logging.debug("opts exists help")
    helpMessgae()
else:
    if "-p" not in opts.keys() or len(opts["-p"]) < 1:
        logging.warning("No assigned respoitory.")
        sys.exit(1)
    
    if opts["-o"] not in allowedOperations:
        logging.warning("No allowed operation.")
        sys.exit(1)
        
    if opts["-o"] in ["master-lastest-commit", "mlc"]:
        apidata = api_fetchMasterLastestCommit(opts["-p"])
        if apidata["status_code"] == 200:
            req_data = json.loads(apidata["raw_info"].text)
            print(req_data[0]["sha"])
        else:
            print('-1')









    
    