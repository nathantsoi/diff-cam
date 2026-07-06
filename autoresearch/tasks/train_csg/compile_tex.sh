#!/bin/bash
# Compile method.tex + experiments.tex with pdflatex by wrapping them in a
# standalone LaTeX document.

# Set working directory to the script's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "Creating temporary LaTeX wrapper..."
cat << 'EOF' > wrapper.tex
\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}

\begin{document}
\input{method.tex}
\input{experiments.tex}
\end{document}
EOF

echo "Running pdflatex (Pass 1)..."
pdflatex -jobname=method -interaction=nonstopmode wrapper.tex > /dev/null

echo "Running pdflatex (Pass 2 for cross-references)..."
pdflatex -jobname=method -interaction=nonstopmode wrapper.tex > /dev/null

echo "Cleaning up temporary wrapper..."
rm -f wrapper.tex

if [ -f method.pdf ]; then
  echo "Success: Compiled method.tex + experiments.tex to method.pdf"
else
  echo "Error: Failed to compile method.pdf. Check method.log for details."
  exit 1
fi
