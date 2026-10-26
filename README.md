# Experimental CID-to-Unicode PDF text recovery

This project is an **experimental recovery tool**, not a general-purpose PDF text extractor and not production-ready software.

It is intended for a specific failure case: a PDF contains text encoded with a CID-keyed CFF font, but its CID-to-Unicode (`ToUnicode`) mapping is missing or unusable. In that situation, ordinary PDF text extraction may return incorrect characters or no useful text. This script tries to rebuild a usable mapping by identifying the embedded glyphs and comparing their shapes with glyphs from reference fonts.

The results are heuristic. Similar glyph shapes, unusual font metrics, an incomplete embedded font, or incorrect CID/GID assumptions can still produce wrong text. Always review the output.

## What the script does

1. Finds embedded CID-keyed CFF fonts used by the PDF.
2. Reads the CFF charset and uses recognizable glyph names such as `cid00068` to build CID/GID relationships. If the charset names do not provide a recognizable CID pattern at all, the current implementation can fall back to the assumption `CID == GID`; the log reports this assumption. If the names look like a CID map but are partial or contain duplicate CIDs, the font is skipped rather than trusting an incomplete map.
3. Renders the embedded glyph outlines and compares them with Unicode-mapped glyphs in selected reference fonts using intersection-over-union (IoU) shape scores.
4. Chooses a Unicode candidate for each CID. When reference-font scores are within `CROSS_FONT_TIE_TOLERANCE`, the average score of the reference font is used to resolve the near-tie.
5. If the final mapping's average IoU is below `0.80`, attempts to rebuild the mapping using **Calibri regular** as the sole reference font. This fallback requires the configured Calibri regular font to be available locally. If it is unavailable, the script logs a warning and keeps the best mapping it already found.
6. Uses MuPDF's text trace and character bounding boxes to place recovered characters back into the document's text layout.

This is **not OCR**: it does not recognize text from page images. It relies on embedded vector font outlines and PDF text-position information.

## Requirements

- Python 3.10 or newer is recommended.
- Reference font files that you are legally allowed to use.
- The Python packages listed in `requirements.txt`.

Install dependencies from the project directory:

```bash
python -m pip install -r requirements.txt
```

## Configure the default input

Open `decode_cid_pdf_classes.py` and edit `Configuration.PDF_PATH` near the top of the file. It can point to one PDF file or a folder containing PDFs.

Example for a PDF beside the script:

```python
PDF_PATH = Path(__file__).with_name("input.pdf")
```

Example for a folder beside the script:

```python
PDF_PATH = Path(__file__).with_name("pdfs")
```

When `PDF_PATH` is a folder, the script processes PDFs directly inside it. Subfolders are only included when the command-line `--recursive` option is used.

## Run from the command line

Run the configured input (the value of `Configuration.PDF_PATH`):

```bash
python decode_cid_pdf_classes.py
```

Process one PDF:

```bash
python decode_cid_pdf_classes.py "path/to/input.pdf"
```

Process all PDFs directly inside a folder:

```bash
python decode_cid_pdf_classes.py "path/to/pdfs"
```

Include PDFs in subfolders:

```bash
python decode_cid_pdf_classes.py "path/to/pdfs" --recursive
```

Choose an output folder:

```bash
python decode_cid_pdf_classes.py "path/to/input.pdf" --output-dir "path/to/output"
```

Run `python decode_cid_pdf_classes.py --help` to see all command-line options.

## Reference fonts

The list of candidate reference fonts and their family/style labels is configured in `ReferenceFontLibrary.FONT_TABLE` in the Python script. The default font search directories are set in `Configuration.FONT_SEARCH_DIRS`.

On startup, the script copies configured font files that are missing from the local `reference_fonts/` folder when it can find them in the configured search directories. Existing local copies are retained. You can also place font files in `reference_fonts/` yourself, using the filenames declared in `FONT_TABLE`.

For reliable results:

- Check the console output for missing reference fonts and style-name mismatch warnings.
- Confirm that regular, italic, bold, and bold-italic entries point to the intended files.
- If the low-score fallback is needed, ensure the configured `Calibri` / `regular` font (`calibri.ttf` by default) is available.

**Font licensing:** installed fonts may have restrictions on copying, redistribution, or use. Check the applicable license before copying fonts to another machine or packaging them with this code. This project does not provide font files.

## Output

For each input PDF, the script writes a UTF-8 text file named:

```text
<input-name>_decoded.txt
```

By default, the text file is placed beside the source PDF. Use `--output-dir` to write it elsewhere.

For multi-page PDFs, the output includes page labels separated by blank lines, for example:

```text
Page 1 of 3

Recovered text from the first page.

Page 2 of 3

Recovered text from the second page.

Page 3 of 3

Recovered text from the third page.
```

A single-page PDF is written without a page label. Mapping scores and per-CID choices are printed for diagnostics and kept in memory during the run. For folder batches, rendered reference-font glyphs are shared in memory between PDFs to reduce repeated work. The script does not write mapping CSV files or JSON cache files.

## Known limitations

- It is designed for PDFs with CID-keyed CFF fonts and missing or unusable `ToUnicode` mappings; it is not intended to handle every PDF font format or encoding problem.
- Glyph-shape matching can confuse visually similar characters, especially where reference-font metrics differ from the embedded font.
- The reference font selected by the embedded font's name is only a candidate filter. A confident family-name match does not guarantee that the family or style is the original source font.
- A low average IoU triggers the Calibri regular fallback, but that fallback is not evidence that Calibri was the original font or that the resulting text is correct.
- Character order, spacing, text blocks, rotations, and bounding-box matching can still be imperfect.
- If no usable font mapping can be built, some text or entire fonts may be skipped.

## Experimental-use notice

Use this tool for investigation, recovery experiments, and manual review. Do not rely on its output without validation for legal, archival, accessibility, publishing, or other high-stakes purposes. Keep an unmodified copy of the original PDF and compare recovered text against known content whenever possible.
