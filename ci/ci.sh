#!/bin/bash

# convert md ro html
cd /home/jkw/sophia/portal/ci
python ConvertMd2Html.py -e /home/jkw/sophia/portal -i /home/jkw/sophia/portal/data -o /home/jkw/sophia/portal/data

# CI vaildation
cd /home/jkw/sophia/portal/ci
python ci.py -d /home/jkw/sophia/portal/data
if [ $? = 1 ]; then
    echo "Not passing the CI test."
    exit 1
else
    echo "Pass all check points."
fi

exit 0


