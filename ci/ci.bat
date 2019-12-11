@echo off

REM convert md ro html
cd C:\inetpub\wwwroot\sophia\ci
python ConvertMd2Html.py -e C:\inetpub\wwwroot\sophia -i C:\inetpub\wwwroot\sophia\data -o C:\inetpub\wwwroot\sophia\data

REM CI vaildation
cd C:\inetpub\wwwroot\sophia\ci
python ci.py -d C:\inetpub\wwwroot\sophia\data
if errorlevel 1 (
    echo "Not passing the CI test."
    EXIT \b 1
) else (
    echo "Pass all check points."
)

EXIT \b 0