#!/bin/bash

# usage example (run as jenkins)
# sh ./cicd.sh /tmp /home/jkw/sophia/portal

# get the parameter and set the directory
ciTestPath=""
cdPath=""
if [ "$#" -ne 2 ] || ! [ -d "$1" ] || ! [ -d "$2" ]; then
    echo "Usage: Default DIRECTORY"
    ciTestPath=/tmp
    cdPath=/home/jkw/sophia/portal
else
    echo "Usage: Assigned Path"
    ciTestPath=$1
    cdPath=$2
fi

# check current vc status
cd $cdPath
originVer=$(git rev-parse HEAD)
githubVer=$(python $cdPath/ci/github_api.py -o mlc -p jiankaiwang/sophia)
if [ $githubVer = "-1" ]; then
    echo "Error parsing repository information from Github API."
    exit 1
fi
if [ $githubVer = $originVer ]; then
    echo "No need to start the cicd process."
    exit 0
fi

# git clone to check
testEnv=$ciTestPath/cicd_test
mkdir -p $testEnv
cd $testEnv
git clone https://github.com/jiankaiwang/sophia.git
testFolder=$testEnv/sophia

# convert md ro html
cd $testFolder/ci
python ConvertMd2Html.py -e $testFolder -i $testFolder/data -o $testFolder/data

# remove all test file
removeTestFile() {
    rm -rf $testEnv
}

# CI vaildation
cd $testFolder/ci
python ci.py -d $testFolder/data
if [ $? = 1 ]; then
    echo "Not passing the CI test."
    removeTestFile;
    exit 1
else
    echo "Pass all check points."
    removeTestFile;
fi

# continuous delivery & deployment
cdlog=$cdPath/cdlog.txt
cd $cdPath

git checkout master
git pull --rebase
lastVer=$(git rev-parse HEAD)
echo $(date),origin:$originVer,latest:$lastVer >> $cdlog

exit 0








