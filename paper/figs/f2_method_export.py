"""Export the compiled TikZ method figure as a vector SVG.

Compile f2_method.tex with Tectonic first, then run this script with PyMuPDF.
Text and paths remain editable; the full diagram source is f2_method.tex.
"""
from pathlib import Path

import fitz


def main():
    source = Path(__file__).with_name("f2_method.pdf")
    with fitz.open(source) as document:
        assert len(document) == 1, "The method figure must fit one page"
        svg = document[0].get_svg_image(text_as_path=False)
        assert "<image" not in svg, "Raster content found in method figure export"
        source.with_suffix(".svg").write_text(svg)
        print(f"Vector SVG exported at {document[0].rect.width / 72:.3f} inches")


if __name__ == "__main__":
    main()
