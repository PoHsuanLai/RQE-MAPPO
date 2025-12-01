# Makefile for RQE-MAPPO Proposal Slides

# Main targets
.PHONY: all clean view short

# Default: build short version (recommended)
all: proposal_short.pdf

# Full version (29 slides)
proposal.pdf: proposal.tex
	pdflatex proposal.tex
	pdflatex proposal.tex

# Short version (10 slides, recommended)
proposal_short.pdf: proposal_short.tex
	pdflatex proposal_short.tex
	pdflatex proposal_short.tex

short: proposal_short.pdf

full: proposal.pdf

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

distclean: clean
	rm -f proposal.pdf proposal_short.pdf

view: proposal_short.pdf
	open proposal_short.pdf

view-full: proposal.pdf
	open proposal.pdf

# Quick rebuild (single pass)
quick:
	pdflatex proposal_short.tex
