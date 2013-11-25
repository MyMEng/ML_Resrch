#! /bin/bash

pdflatex report
bibtex report
pdflatex report

# $ latex myarticle
# $ bibtex myarticle
# $ latex myarticle
# $ latex myarticle
