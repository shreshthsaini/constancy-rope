#!/usr/bin/env python3
"""Package main.tex's literal dependency graph without modifying the paper.

Run with Python 3.9+ and --output /path/to/new_archive.zip. The archive is
deterministic for fixed input bytes. Existing output files are never replaced.
This is a compile-source bundle, not a figure or experiment regeneration bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile


ARCHIVE_ROOT = "constancy_rope_paper_src"
COMMAND = re.compile(
    r"\\(?P<name>InputIfFileExists|includegraphics|figorplaceholder|"
    r"bibliographystyle|bibliography|addbibresource|documentclass|"
    r"RequirePackage|usepackage|include|input)\*?\s*"
    r"(?:\[[^\]]*\]\s*)?\{(?P<target>[^{}]+)\}"
)
UNSUPPORTED = re.compile(
    r"\\(?:includeonly|graphicspath|import|subimport|inputminted|"
    r"lstinputlisting|verbatiminput|includepdf|attachfile)\b"
)
TEX_INPUTS = {"input", "include", "InputIfFileExists"}
GRAPHICS = {"includegraphics", "figorplaceholder"}
BUNDLE_TYPES = {
    "documentclass": ".cls", "usepackage": ".sty",
    "RequirePackage": ".sty", "bibliographystyle": ".bst",
}
README = """# Paper source

This archive contains the active manuscript, its generated table inputs,
bibliography and figure PDFs. It excludes inactive sections and internal notes.
Figures and tables are supplied as compile inputs; this is not an experiment
or figure regeneration bundle.

From this directory, verify the files and compile with Tectonic:

```sh
sha256sum -c SHA256SUMS
OMP_NUM_THREADS=8 tectonic -X compile main.tex
```

The result is main.pdf. Tectonic downloads missing standard TeX packages on its
first run; a populated cache permits offline compilation. No GPU or Python
packages are required to compile the manuscript. pdfLaTeX has not been tested.

dependency_graph.json lists the local dependencies and standard TeX resources.
SHA256SUMS covers every supplied file except itself. The optional Python 3.9+
packager can recreate the archive from this directory:

```sh
python3 package_source.py --output ../paper_source_copy.zip
```

The packager uses fixed ZIP metadata and sorted entries. It refuses to replace
an existing archive. Identical source bytes and compression implementation
produce identical archives.
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_comments(source: str) -> str:
    """Remove TeX comments while retaining escaped percent characters."""
    lines = []
    for line in source.splitlines(keepends=True):
        for i, char in enumerate(line):
            if char != "%":
                continue
            backslashes = 0
            j = i - 1
            while j >= 0 and line[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                line = line[:i] + "\n"
                break
        lines.append(line)
    return "".join(lines)


def discover(root: Path) -> tuple[dict[str, bytes], dict]:
    files: dict[str, bytes] = {}
    edges = []
    standard_resources = set()
    optional_missing = []

    def resolve(target: str, suffixes: tuple[str, ...], optional: bool = False):
        if any(char in target for char in "\\#{}\n\r"):
            raise ValueError(f"Nonliteral dependency path: {target!r}")
        path = Path(target)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Dependency must stay inside the paper: {target}")
        if any(part in {".agent", ".agents", "archive", "dist"} for part in path.parts):
            raise ValueError(f"Internal or inactive dependency: {target}")
        candidates = [path] if path.suffix else [Path(target + ext) for ext in suffixes]
        for candidate in candidates:
            full = (root / candidate).resolve()
            if not full.is_relative_to(root):
                raise ValueError(f"Dependency resolves outside the paper: {target}")
            if full.is_file():
                return candidate.as_posix()
        if optional:
            return None
        raise FileNotFoundError(f"Missing dependency: {target}")

    def visit(name: str):
        if name in files:
            return
        data = (root / name).read_bytes()
        files[name] = data
        if Path(name).suffix not in {".tex", ".sty", ".cls"}:
            return
        source = strip_comments(data.decode("utf-8"))
        unsupported = UNSUPPORTED.search(source)
        if unsupported:
            raise ValueError(f"Unsupported dependency command in {name}: {unsupported[0]}")
        if re.search(r"\\(?:input|include)\b(?!\s*\{)", source):
            raise ValueError(f"Unbraced input is not supported: {name}")
        for match in COMMAND.finditer(source):
            command, target = match.group("name", "target")
            target = target.strip()
            # The manuscript's wrapper is handled at each literal call site.
            if name == "macros.tex" and command == "includegraphics" and target == "#2":
                continue
            targets = target.split(",") if command in BUNDLE_TYPES or command == "bibliography" else [target]
            for item in targets:
                item = item.strip()
                if command in BUNDLE_TYPES:
                    suffix = BUNDLE_TYPES[command]
                    dependency = resolve(item, (suffix,), optional=True)
                    if dependency is None:
                        standard_resources.add(item if item.endswith(suffix) else item + suffix)
                        continue
                elif command in TEX_INPUTS:
                    dependency = resolve(item, (".tex", ""), optional=command == "InputIfFileExists")
                    if dependency is None:
                        optional_missing.append({"source": name, "target": item})
                        continue
                elif command in GRAPHICS:
                    dependency = resolve(item, (".pdf", ".png", ".jpg", ".jpeg"))
                else:
                    dependency = resolve(item, (".bib",))
                edges.append({"source": name, "command": command, "target": dependency})
                visit(dependency)

    visit("main.tex")
    graph = {
        "format_version": 1,
        "entrypoint": "main.tex",
        "dependency_file_count": len(files),
        "files": [{"path": name, "sha256": sha256(data), "bytes": len(data)}
                  for name, data in sorted(files.items())],
        "edges": sorted(edges, key=lambda edge: (edge["source"], edge["target"], edge["command"])),
        "standard_tex_resources": sorted(standard_resources),
        "optional_inputs_absent": sorted(optional_missing, key=lambda edge: (edge["source"], edge["target"])),
    }
    return files, graph


def package(root: Path, output: Path) -> dict:
    files, graph = discover(root)
    source_snapshot = dict(files)
    script = Path(__file__).resolve()
    files["package_source.py"] = script.read_bytes()
    files["README.md"] = README.encode("utf-8")
    files["dependency_graph.json"] = (json.dumps(graph, indent=2, sort_keys=True) + "\n").encode("utf-8")
    files["SHA256SUMS"] = "".join(
        f"{sha256(data)}  {name}\n" for name, data in sorted(files.items())
    ).encode("utf-8")
    changed = [name for name, data in source_snapshot.items() if (root / name).read_bytes() != data]
    if changed:
        raise RuntimeError(f"Source files changed during packaging: {changed}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves existing release and preview artifacts.
    with output.open("xb") as raw:
        with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in sorted(files.items()):
                info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return {
        "archive": str(output.resolve()),
        "sha256": sha256(output.read_bytes()),
        "bytes": output.stat().st_size,
        "dependency_file_count": len(source_snapshot),
        "packaged_file_count": len(files),
        "archive_root": ARCHIVE_ROOT,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New ZIP path; existing files are refused")
    parser.add_argument("--paper-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(package(args.paper_dir.resolve(), args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
