@echo off

cd C:\inetpub\wwwroot\sophia\ci
python ci.py -e C:\inetpub\wwwroot\sophia -i C:\inetpub\wwwroot\sophia\data -o C:\inetpub\wwwroot\sophia\data

PAUSE