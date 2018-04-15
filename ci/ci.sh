#!/bin/bash

# convert md ro html
cd /c/inetpub/wwwroot/sophia/ci
python ConvertMd2Html.py -e /c/inetpub/wwwroot/sophia/ -i /c/inetpub/wwwroot/sophia/data -o /c/inetpub/wwwroot/sophia/data

# CI vaildation
cd /c/inetpub/wwwroot/sophia/ci
python ci.py -d /c/inetpub/wwwroot/sophia/data
if [ $? = 1 ]; then
    echo "Not passing the CI test."
    exit 1
else
    echo "Pass all check points."
fi

exit 0

