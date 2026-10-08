#!/bin/bash
# Build report.docx and report.html from report.md. Figures are read from ../results/.
# Requires pypandoc_binary (bundles pandoc): .venv/bin/pip install pypandoc_binary==1.15
# For the PDF: open report.docx in Word (File > Save As > PDF) or report.html in a browser (Print > PDF).
set -e
cd "$(dirname "$0")"
PANDOC=$(../.venv/bin/python -c "import pypandoc; print(pypandoc.get_pandoc_path())")
"$PANDOC" report.md -o report.docx --resource-path=.:../results
"$PANDOC" report.md -o report.html --standalone --embed-resources --resource-path=.:../results \
  --css style.css --metadata pagetitle="CISC 451 Assignment 1"
echo "wrote report.docx report.html"
