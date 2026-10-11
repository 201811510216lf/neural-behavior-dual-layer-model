# Figure QA

- Automated strict check: all 12 PNG figures pass the 300 dpi requirement.
- Export coverage: every figure stem is present as PNG, TIFF, PDF, and SVG.
- Visual montage review: labels, annotations, legends, axes, and panel alignment are readable; no clipping or overlap that changes interpretation was observed.
- Grayscale review: annotated heatmaps retain sign and magnitude information without relying only on color.
- Source-data review: every figure has a corresponding CSV or compressed CSV source file.

The `meta NOT subset` messages emitted during vector export are font-subsetting warnings from the plotting stack; they do not alter plotted data or raster/vector availability.
