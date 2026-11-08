# Build helpers for the UPC 2026 Problem B paper.
#   make          compile, then run the submission checks
#   make pdf      also copy to <team><problem>.pdf for submission
#   make figs     regenerate the figures that need no external data
#   make clean    remove LaTeX by-products (keeps main.pdf)
# TEAM and PROBLEM come from config.tex, so there is one source of truth.

MAIN    := main.tex
TEAM    := $(shell sed -n 's/^\\newcommand{\\teamnum}{\([^}]*\)}.*/\1/p' config.tex)
PROBLEM := $(shell sed -n 's/^\\newcommand{\\problemltr}{\([^}]*\)}.*/\1/p' config.tex)
SUBMIT  := $(TEAM)$(PROBLEM).pdf
SKILL   ?= scripts
LATEXMK_FLAGS := -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1

.PHONY: all build check words figs pdf clean

all: build check

build:
	latexmk $(LATEXMK_FLAGS) $(MAIN)

check: build
	python3 $(SKILL)/check_pdf.py main.pdf --team $(TEAM) --problem $(PROBLEM)
	python3 $(SKILL)/count_summary.py main.tex --tex

words:
	python3 $(SKILL)/count_summary.py $(MAIN) --tex

figs:
	@echo "regenerating the figure files that need no external data"
	cd code && python3 mkfig_a.py
	cd code && python3 mkfig_c.py
	@echo "other figure scripts read UPC_ROOT; see README"

pdf: check
	cp main.pdf $(SUBMIT)
	@echo "wrote $(SUBMIT)"

clean:
	latexmk -c $(MAIN)
	rm -f main.bbl main.run.xml main.synctex.gz
