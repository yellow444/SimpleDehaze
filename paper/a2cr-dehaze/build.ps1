$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $perl = Get-Command perl -ErrorAction SilentlyContinue
    if ($perl) {
        latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    }
    else {
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
        bibtex main
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
    }
}
finally {
    Pop-Location
}
