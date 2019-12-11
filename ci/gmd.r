# author: JianKai Wang (https://jiankaiwang.no-ip.biz)
# desc: render the *.md or *.rmd to *.html
# notice: Please make sure "pandoc" is installed. (https://pandoc.org/installing.html)

Sys.setlocale(category = "LC_ALL", locale = "cht")
require("rmarkdown")
sysArgs <- commandArgs(trailingOnly = TRUE)

if(length(sysArgs) > 0) {
  for(i in 1:length(sysArgs)) {
    tryCatch(
      rmarkdown::render(sysArgs[i], "html_document")
      , error = function(e) {
        print(paste("Error: can not render doc ", sysArgs[i], ".", sep=""))
      }
    )
  }
} else {
  print("Usage: Rscript gmd.r <inputfilename1.md> [<inputfilename2.md>]")
  print("Output: inputfilename1.html [inputfilename2.html]")
}