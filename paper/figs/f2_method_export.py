"""Export the compiled TikZ method figure as an SVG.

Compile f2_method.tex with Tectonic first, then run this script with PyMuPDF.
Text and paths remain editable vectors; the one raster is the sample photo of panel (a),
embedded as JPEG in the PDF. The full diagram source is f2_method.tex.
"""
from pathlib import Path

import fitz


def main():
    source = Path(__file__).with_name("f2_method.pdf")
    with fitz.open(source) as document:
        assert len(document) == 1, "The method figure must fit one page"
        page = document[0]
        images = page.get_images(full=True)
        assert len(images) == 1 and images[0][8] == "DCTDecode", "Expected exactly one JPEG photo"
        svg = page.get_svg_image(text_as_path=False)
        assert svg.count("<image") == 1, "Expected exactly one raster element in the SVG"
        out = source.with_suffix(".svg")
        out.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
        print(f"SVG exported at {page.rect.width / 72:.3f} x {page.rect.height / 72:.3f} in, "
              f"{out.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
