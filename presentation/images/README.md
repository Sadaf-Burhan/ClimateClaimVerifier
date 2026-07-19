# images/ — drop your image files here

## Climate Scanner logo (required for the title + demo slides)
Save the logo as **`climate-scanner-logo.png`** in this folder. It appears small in the
top-right of the **title slide** and large on the **"Let's see the demo!"** slide.
If the file is missing, both slides simply hide the logo (nothing breaks).

## Eraser diagram exports (optional — replace the D2 recreations)

The architecture and workflow slides prefer **your exact Eraser diagrams** and fall
back to the D2 recreations only if these files are missing.

To use your Eraser diagrams as-is:

1. In Eraser.io, open each diagram → **Export**.
2. Choose **SVG** (sharpest; scales perfectly in the scrollable pane). PNG at 2× also works.
3. Save them here with these exact names:
   - `architecture.svg`  → shown on the "architecture" slide
   - `workflow.svg`      → shown on the "workflow" slide
4. Hard-refresh the deck (Ctrl+F5). The slides pick them up automatically — no code change.

Both are shown in a **scrollable pane**, so any size/orientation is fine: the image
renders full-size and the panel scrolls. If a file isn't here yet, the slide shows the
D2 version from `../diagrams/` instead (so nothing is ever broken).
