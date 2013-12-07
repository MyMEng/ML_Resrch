#! /bin/bash

# multimarkdown ../README.md -t latex -o readme.tex
pdflatex report
bibtex report
pdflatex report

# $ latex myarticle
# $ bibtex myarticle
# $ latex myarticle
# $ latex myarticle
