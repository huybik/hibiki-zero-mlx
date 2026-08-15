# Technical report

`report.md` is the editable source. Its figures are hand-authored SVGs under
`assets/`; every plotted value comes from the report text and linked benchmark
artifacts.

Build the submission PDF with the project Python:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python docs/technical/build_pdf.py
```

The builder uses Pandoc for the Markdown AST and ReportLab for layout. It writes:

```text
output/pdf/hibiki-zero-mlx-technical-report.pdf
```

Required local tools: `pandoc`, ReportLab, and headless Google Chrome (SVG
rasterization with system fonts for the PDF only). The Markdown retains vector
SVGs for browser/GitHub reading.
