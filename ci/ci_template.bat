@echo off

cd C:\inetpub\wwwroot\sophia\data

#pandoc +RTS -K512m -RTS movidius_quickstart.md --to html4 --from markdown+autolink_bare_uris+ascii_identifiers+tex_math_single_backslash+smart --output movidius_quickstart.html --email-obfuscation none --self-contained --standalone --section-divs --template "C:\inetpub\wwwroot\sophia\ci\template\default.html" --no-highlight --variable highlightjs=1 --variable "theme:bootstrap" --mathjax --variable "mathjax-url:https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" --include-in-header "..\ci\template\header.html"

## movidius_quickstart
pandoc +RTS -K512m -RTS movidius_quickstart.md --to html4 --from markdown+autolink_bare_uris+ascii_identifiers+tex_math_single_backslash+smart --output movidius_quickstart.html --email-obfuscation none --self-contained --standalone --section-divs --template "C:\inetpub\wwwroot\sophia\ci\template\default.html" --no-highlight --variable highlightjs=1 --variable "theme:bootstrap" --mathjax --variable "mathjax-url:https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" --include-in-header "..\ci\template\header.html"

## jetson_tx2_quickstart
pandoc +RTS -K512m -RTS jetson_tx2_quickstart.md --to html4 --from markdown+autolink_bare_uris+ascii_identifiers+tex_math_single_backslash+smart --output jetson_tx2_quickstart.html --email-obfuscation none --self-contained --standalone --section-divs --template "C:\inetpub\wwwroot\sophia\ci\template\default.html" --no-highlight --variable highlightjs=1 --variable "theme:bootstrap" --mathjax --variable "mathjax-url:https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" --include-in-header "C:\inetpub\wwwroot\sophia\ci\template\header.html"

## Neighbor_Joining
pandoc +RTS -K512m -RTS Neighbor_Joining.md --to html4 --from markdown+autolink_bare_uris+ascii_identifiers+tex_math_single_backslash+smart --output Neighbor_Joining.html --email-obfuscation none --self-contained --standalone --section-divs --template "C:\inetpub\wwwroot\sophia\ci\template\default.html" --no-highlight --variable highlightjs=1 --variable "theme:bootstrap" --mathjax --variable "mathjax-url:https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" --include-in-header "C:\inetpub\wwwroot\sophia\ci\template\header.html"

## vott_cntk_flow
pandoc +RTS -K512m -RTS vott_cntk_flow.md --to html4 --from markdown+autolink_bare_uris+ascii_identifiers+tex_math_single_backslash+smart --output vott_cntk_flow.html --email-obfuscation none --self-contained --standalone --section-divs --template "C:\inetpub\wwwroot\sophia\ci\template\default.html" --no-highlight --variable highlightjs=1 --variable "theme:bootstrap" --mathjax --variable "mathjax-url:https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" --include-in-header "C:\inetpub\wwwroot\sophia\ci\template\header.html"

PAUSE