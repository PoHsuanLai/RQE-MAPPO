# RQE-MAPPO Research Proposal Slides

This directory contains the Beamer presentation slides for the RQE-MAPPO research proposal.

## Contents

- `proposal_short.tex` - **Short version** (10 slides, RECOMMENDED)
- `proposal_short.pdf` - Compiled short presentation
- `proposal.tex` - Full version (29 slides, detailed)
- `proposal.pdf` - Compiled full presentation
- `Makefile` - Build automation

## Building the Slides

### Using Make (recommended)

```bash
make              # Build short version (default, recommended)
make view         # Build and open short version
make short        # Build short version explicitly
make full         # Build full version (29 slides)
make view-full    # Build and open full version
make quick        # Quick rebuild (single pass)
make clean        # Remove auxiliary files
make distclean    # Remove all generated files including PDFs
```

### Using pdflatex directly

```bash
pdflatex proposal.tex
pdflatex proposal.tex  # Run twice for proper references
```

## Theme

Uses the **CleanEasy** Beamer theme located in `~/Library/texmf/tex/latex/beamer/`

## Required LaTeX Packages

- `beamer` - Presentation class
- `cmbright` - CM Bright fonts
- `amsmath`, `amssymb` - Mathematical symbols
- `booktabs` - Professional tables
- `tikz` - Graphics and diagrams

All packages should be installed via TeX Live.

## Presentation Structure

### Short Version (10 slides - RECOMMENDED)
Follows the 8-point structure for concise research proposals:

1. **Core Idea** - One-sentence take-away message
2. **Problem** - Visual examples, What/Why/Impact
3. **Prior Work Limitations** - What's missing, how far from solution
4. **Our Insight** - Key differences, computational tractability
5. **Method Components** - Algorithm details, architecture
6. **Experimental Setup** - Environments, baselines, backbone
7. **Experiments & Results** - Expected outcomes and why
8. **Timeline & Milestones** - 8-week plan with success criteria

### Full Version (29 slides)
Comprehensive presentation with all details, experiments, and background.

## Notes

- **Short**: 10 slides, 16:9 aspect ratio, focused for class presentations
- **Full**: 29 slides, comprehensive for detailed discussions
- Professional academic style with CleanEasy theme
- TikZ diagrams and visualizations included
