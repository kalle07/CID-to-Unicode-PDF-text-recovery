"""Experimental recovery for PDFs with CID-keyed CFF fonts and no ToUnicode map.

Use Configuration.PDF_PATH for the default PDF file or input folder, or pass a
PDF path/folder on the command line. Decoded text is written to *_decoded.txt.
Reference fonts are copied to ./reference_fonts when available. Mapping scores
are kept in memory; no mapping CSV or JSON cache files are written.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from PIL import Image
from fontTools.cffLib import CFFFontSet
from fontTools.pens.basePen import BasePen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTFont


class Configuration:
    """Central configuration for PDF decoding and font matching."""

    PDF_PATH = Path(__file__).with_name("Automobildesign und Technik_1.pdf")
    REFERENCE_FONT_DIR = Path(__file__).with_name("reference_fonts")

    IMAGE_SIZE = 128
    BASELINE_FRACTION = 0.76
    RASTER_SUPERSAMPLE = 1
    # Only near-equal cross-font scores use the reference font's average score.
    CROSS_FONT_TIE_TOLERANCE = 0.01
    FINAL_MAPPING_MIN_AVERAGE_IOU = 0.70
    FALLBACK_FONT_FAMILY = "Calibri"
    FALLBACK_FONT_STYLE = "regular"
    FONT_NAME_FUZZY_THRESHOLD = 0.80
    FONT_NAME_AMBIGUITY_MARGIN = 0.035

    FONT_SEARCH_DIRS: list[Path] = [
        Path(r"C:\Windows\Fonts"),
        Path("/Library/Fonts"),
        Path("/usr/share/fonts/truetype"),
    ]

    BUCKET_SIZE = 5.0
    LOCAL_MATCH_DISTANCE = 2.0
    GLOBAL_MATCH_DISTANCE = 5.0
    HORIZONTAL_LINE_ANGLE_TOLERANCE = 0.15
    SPACE_GAP_FRACTION = 0.28
    MIN_SPACE_GAP = 2.0


class GlyphOutlineRenderer:
    """Renders font glyph outlines into compact binary masks."""

    @dataclass(frozen=True)
    class RenderedGlyph:
        mask: int
        pixel_count: int
        bitmap: Image.Image | None

        @property
        def is_empty(self) -> bool:
            return self.pixel_count == 0

    class _OutlinePen(BasePen):
        """Collect sampled contour points from TrueType or CFF glyph outlines."""

        def __init__(self, glyph_set=None):
            super().__init__(glyph_set)
            self.contours: list[list[tuple[float, float]]] = []
            self.current: list[tuple[float, float]] = []
            self.position = (0.0, 0.0)

        def moveTo(self, point):
            if self.current:
                self.contours.append(self.current)
            self.current = [point]
            self.position = point

        def lineTo(self, point):
            self.current.append(point)
            self.position = point

        def curveTo(self, p1, p2, p3):
            self._add_cubic_curve(p1, p2, p3)
            self.position = p3

        def curveToOne(self, p1, p2, p3):
            self._add_cubic_curve(p1, p2, p3)
            self.position = p3

        def _add_cubic_curve(self, p1, p2, p3):
            x0, y0 = self.position
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3

            steps = 16
            for i in range(1, steps + 1):
                t = i / steps
                u = 1.0 - t
                x = (
                    u**3 * x0
                    + 3 * u**2 * t * x1
                    + 3 * u * t**2 * x2
                    + t**3 * x3
                )
                y = (
                    u**3 * y0
                    + 3 * u**2 * t * y1
                    + 3 * u * t**2 * y2
                    + t**3 * y3
                )
                self.current.append((x, y))

            self.position = p3

        def closePath(self):
            if self.current:
                if self.current[-1] != self.current[0]:
                    self.current.append(self.current[0])
                self.contours.append(self.current)
                self.current = []

        def endPath(self):
            if self.current:
                self.contours.append(self.current)
                self.current = []

        def finish(self):
            if self.current:
                self.contours.append(self.current)
                self.current = []
            return self.contours

    @staticmethod
    def _fill_nonzero_winding(
        image: np.ndarray,
        contours: list[list[tuple[float, float]]],
    ) -> None:
        """Rasterize closed contours using the non-zero winding fill rule.

        Fills `image` in-place. The input must be a 2-D uint8 array of
        shape `(height, width)` initialized to zero. Counters are preserved
        because the winding number is tracked per scan line.
        """
        height, width = image.shape
        row_events: list[list[tuple[float, int]]] = [[] for _ in range(height)]

        for contour in contours:
            points = [(float(x), float(y)) for x, y in contour]
            if len(points) < 3:
                continue
            if points[-1] != points[0]:
                points.append(points[0])

            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                dy = y2 - y1
                if abs(dy) < 1e-12:
                    continue

                first_row = max(0, math.ceil(min(y1, y2) - 0.5))
                stop_row = min(height, math.ceil(max(y1, y2) - 0.5))
                if first_row >= stop_row:
                    continue

                winding_delta = 1 if dy > 0 else -1
                for row in range(first_row, stop_row):
                    scan_y = row + 0.5
                    x = x1 + ((scan_y - y1) / dy) * (x2 - x1)
                    row_events[row].append((x, winding_delta))

        for y, events in enumerate(row_events):
            if len(events) < 2:
                continue

            events.sort(key=lambda event: event[0])
            winding = 0
            previous_x: float | None = None
            index = 0

            while index < len(events):
                x = events[index][0]

                if previous_x is not None and winding != 0 and x > previous_x:
                    x_start = max(0, math.ceil(previous_x - 0.5))
                    x_stop = min(width, math.ceil(x - 0.5))
                    if x_start < x_stop:
                        image[y, x_start:x_stop] = 255

                next_index = index
                delta_sum = 0
                while (
                    next_index < len(events)
                    and abs(events[next_index][0] - x) < 1e-7
                ):
                    delta_sum += events[next_index][1]
                    next_index += 1

                winding += delta_sum
                previous_x = x
                index = next_index

    @classmethod
    def render_glyph_bitmap(
        cls,
        contours: list[list[tuple[float, float]]],
        units_per_em: int,
        image_size: int = Configuration.IMAGE_SIZE,
        supersample: int = Configuration.RASTER_SUPERSAMPLE,
    ) -> Image.Image | None:
        """Render outlines with proper counters on a shared em scale.

        Contours are rasterized with the non-zero winding rule at an enlarged
        resolution, then downsampled with LANCZOS. Glyphs are centered
        horizontally; height and baseline are not independently normalized.
        """
        points = [p for contour in contours for p in contour]
        if len(points) < 2 or not units_per_em:
            return None

        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax - xmin <= 0 or ymax - ymin <= 0:
            return None

        scale = (image_size * Configuration.BASELINE_FRACTION) / float(units_per_em)
        x_center = (xmin + xmax) / 2.0
        x_offset = image_size / 2.0 - x_center * scale
        baseline = image_size * Configuration.BASELINE_FRACTION
        supersample = max(1, int(supersample))
        high_size = image_size * supersample

        transformed_contours: list[list[tuple[float, float]]] = []
        for contour in contours:
            transformed_contours.append([
                (
                    (x_offset + float(x) * scale) * supersample,
                    (baseline - float(y) * scale) * supersample,
                )
                for x, y in contour
            ])

        buf = np.zeros((high_size, high_size), dtype=np.uint8)
        cls._fill_nonzero_winding(buf, transformed_contours)
        if buf.max() == 0:
            return None

        high_res = Image.fromarray(buf, mode="L")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = high_res.resize((image_size, image_size), resampling)
        return image if image.getbbox() is not None else None

    @classmethod
    def to_mask(cls, image: Image.Image | None) -> tuple[int, int]:
        """Return a compact bit mask and its foreground-pixel count."""
        if image is None:
            return 0, 0

        dither_none = getattr(getattr(Image, "Dither", Image), "NONE", 0)
        binary = image.convert("1", dither=dither_none)
        mask = int.from_bytes(binary.tobytes(), byteorder="big")
        return mask, mask.bit_count()

    @classmethod
    def render_contours(
        cls,
        contours: list[list[tuple[float, float]]],
        units_per_em: int,
        image_size: int = Configuration.IMAGE_SIZE,
        supersample: int = Configuration.RASTER_SUPERSAMPLE,
    ) -> "GlyphOutlineRenderer.RenderedGlyph":
        image = cls.render_glyph_bitmap(
            contours,
            units_per_em,
            image_size,
            supersample,
        )
        mask, pixel_count = cls.to_mask(image)
        return cls.RenderedGlyph(mask=mask, pixel_count=pixel_count, bitmap=image)

    @staticmethod
    def overlap_score(
        mask_a: int,
        count_a: int,
        mask_b: int,
        count_b: int,
    ) -> float:
        """Compute IoU from pre-computed integer masks and pixel counts."""
        intersection = (mask_a & mask_b).bit_count()
        union = count_a + count_b - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def unicode_preference(codepoint: int) -> tuple[int, int]:
        """Prefer standard text characters when glyph outlines are visually tied."""
        if 0x20 <= codepoint <= 0x7E:
            return (0, codepoint)
        if 0xA0 <= codepoint <= 0xFF:
            return (1, codepoint)
        if 0x0100 <= codepoint <= 0x024F:
            return (2, codepoint)
        if 0x0370 <= codepoint <= 0x052F:
            return (3, codepoint)
        if 0x2000 <= codepoint <= 0x206F:
            return (4, codepoint)
        return (5, codepoint)


class ReferenceFontLibrary:
    """Manage local reference fonts, render glyphs, and score Unicode candidates."""

    FONT_TABLE: dict[str, dict[str, str]] = {
        "Arial": {
            "regular": "arial.ttf",
            "bold": "arialbd.ttf",
            "italic": "ariali.ttf",
            "bolditalic": "arialbi.ttf",
        },
        "Calibri": {
            "regular": "calibri.ttf",
            "bold": "calibrib.ttf",
            "italic": "calibrii.ttf",
            "bolditalic": "calibriz.ttf",
        },
        "Cambria": {
            "regular": "cambria.ttc",
            "bold": "cambriab.ttf",
            "italic": "cambriai.ttf",
            "bolditalic": "cambriaz.ttf",
        },
        "Candara": {
            "regular": "candara.ttf",
            "bold": "candarab.ttf",
            "italic": "candarai.ttf",
            "bolditalic": "candaraz.ttf",
        },
        "Consolas": {
            "regular": "consola.ttf",
            "bold": "consolab.ttf",
            "italic": "consolai.ttf",
            "bolditalic": "consolaz.ttf",
        },
        "Constantia": {
            "regular": "constan.ttf",
            "bold": "constanb.ttf",
            "italic": "constani.ttf",
            "bolditalic": "constanz.ttf",
        },
        "Corbel": {
            "regular": "corbel.ttf",
            "bold": "corbelb.ttf",
            "italic": "corbeli.ttf",
            "bolditalic": "corbelz.ttf",
        },
        "Courier New": {
            "regular": "cour.ttf",
            "bold": "courbd.ttf",
            "italic": "couri.ttf",
            "bolditalic": "courbi.ttf",
        },
        "Georgia": {
            "regular": "georgia.ttf",
            "bold": "georgiab.ttf",
            "italic": "georgiai.ttf",
            "bolditalic": "georgiaz.ttf",
        },
        "Segoe UI": {
            "regular": "segoeui.ttf",
            "bold": "segoeuib.ttf",
            "italic": "segoeuii.ttf",
            "bolditalic": "segoeuiz.ttf",
        },
        "Tahoma": {
            "regular": "tahoma.ttf",
            "bold": "tahomabd.ttf",
        },
        "Times New Roman": {
            "regular": "times.ttf",
            "bold": "timesbd.ttf",
            "italic": "timesi.ttf",
            "bolditalic": "timesbi.ttf",
        },
        "Trebuchet MS": {
            "regular": "trebuc.ttf",
            "bold": "trebucbd.ttf",
            "italic": "trebucit.ttf",
            "bolditalic": "trebucbi.ttf",
        },
        "Verdana": {
            "regular": "verdana.ttf",
            "bold": "verdanab.ttf",
            "italic": "verdanai.ttf",
            "bolditalic": "verdanaz.ttf",
        },
        "CorporateS": {
            "regular": "Corporate S Regular.otf",
            "italic": "CorporateS Regular Italic.otf",
            "bold": "Corporate S Extra Bold.otf",
        },
        "CorporateA": {
            "regular": "Corporate A Light Regular.otf",
            "italic": "Corporate A Light Italic.otf",
        },
    }

    def __init__(
        self,
        search_dirs: list[Path] | None = None,
        local_font_dir: Path | None = None,
    ):
        self.search_dirs = list(
            Configuration.FONT_SEARCH_DIRS if search_dirs is None else search_dirs
        )
        self.local_font_dir = Path(local_font_dir or Configuration.REFERENCE_FONT_DIR)
        self._rendered_cache: dict[str, tuple[list[dict[str, Any]], int]] = {}
        self._bootstrap_local_fonts_if_needed()
        self.font_records, self.missing_font_files = self._load_local_font_records()
        self.fonts_by_family: dict[str, list[dict[str, Any]]] = {
            family: [record for record in self.font_records if family in record["families"]]
            for family in self.FONT_TABLE
        }

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _configured_filenames() -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for styles in ReferenceFontLibrary.FONT_TABLE.values():
            for filename in styles.values():
                key = filename.casefold()
                if key not in seen:
                    names.append(filename)
                    seen.add(key)
        return names

    def _bootstrap_local_fonts_if_needed(self) -> None:
        """Copy any missing configured fonts; retain existing local copies."""
        self.local_font_dir.mkdir(parents=True, exist_ok=True)
        configured_names = self._configured_filenames()

        # A partially populated local folder must not prevent later additions
        # (for example, adding Corporate S Regular.otf to FONT_TABLE).
        local_by_name = {
            path.name.casefold(): path
            for path in self.local_font_dir.iterdir()
            if path.is_file() and path.stat().st_size > 0
        }
        missing_names = [
            name for name in configured_names
            if name.casefold() not in local_by_name
        ]
        if not missing_names:
            return

        wanted = {name.casefold() for name in missing_names}
        sources: dict[str, Path] = {}
        for directory in self.search_dirs:
            if not directory.is_dir():
                continue
            try:
                for candidate in directory.rglob("*"):
                    if candidate.is_file() and candidate.name.casefold() in wanted:
                        sources.setdefault(candidate.name.casefold(), candidate)
            except OSError as exc:
                print(f"  Could not inspect font directory {directory}: {exc}")

        copied = 0
        unavailable: list[str] = []
        for filename in missing_names:
            key = filename.casefold()
            source = sources.get(key)
            destination = self.local_font_dir / filename
            if source is None:
                unavailable.append(filename)
                continue
            try:
                shutil.copy2(source, destination)
                copied += 1
                local_by_name[key] = destination
            except OSError as exc:
                unavailable.append(filename)
                print(f"  Could not copy reference font {source}: {exc}")

        if copied:
            print(f"Copied {copied} missing reference font file(s) to {self.local_font_dir}")
        if unavailable:
            preview = ", ".join(unavailable[:10])
            if len(unavailable) > 10:
                preview += f", ... (+{len(unavailable) - 10} more)"
            print(f"  Not found in configured font search directories: {preview}")
        if not local_by_name:
            print(
                f"WARNING: No configured reference fonts were copied. "
                f"Check FONT_SEARCH_DIRS or place fonts in {self.local_font_dir}."
            )

    @classmethod
    def _filename_style_hint(cls, filename: str, family: str = "") -> str:
        """Infer regular/italic/bold/bolditalic from full words or common suffixes."""
        stem = Path(filename).stem
        normalized_stem = cls._normalize_font_name(stem)
        normalized_family = cls._normalize_font_name(family)

        # Splitting CamelCase and punctuation recognizes full style words without
        # treating arbitrary letters inside a family name as style abbreviations.
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem)
        tokens = {
            cls._normalize_font_name(token)
            for token in re.findall(r"[A-Za-z0-9]+", spaced)
        }
        italic_words = {"italic", "italics", "oblique", "obl", "slanted", "slant"}
        bold_words = {
            "bold", "heavy", "black", "extrabold", "ultrabold", "semibold",
            "demibold", "boldface", "demi", "bd",
        }
        regular_words = {"regular", "reg", "roman", "normal", "book"}

        has_italic = bool(tokens & italic_words)
        has_bold = bool(tokens & bold_words)
        if has_italic and has_bold:
            return "bolditalic"
        if has_italic:
            return "italic"
        if has_bold:
            return "bold"
        if tokens & regular_words:
            # "Regular Italic" is handled above as italic; here regular is the
            # upright/normal style rather than a synonym for a font weight.
            return "regular"

        # Common Windows font filenames encode style as a short suffix:
        # ariali, arialbd, arialbi, calibrib, calibriz, timesi, timesbd, etc.
        # Remove the configured family prefix when present, so the final "i"
        # in a family name such as Calibri is not mistaken for an italic suffix.
        residual = ""
        if normalized_family and normalized_stem.startswith(normalized_family):
            residual = normalized_stem[len(normalized_family):]
        if normalized_stem == normalized_family:
            residual = ""

        suffix_map = {
            "bolditalic": "bolditalic", "boldoblique": "bolditalic",
            "bi": "bolditalic", "ib": "bolditalic", "z": "bolditalic",
            "extrabold": "bold", "ultrabold": "bold", "semibold": "bold",
            "demibold": "bold", "bold": "bold", "heavy": "bold", "black": "bold",
            "bd": "bold", "sb": "bold", "db": "bold", "b": "bold",
            "italic": "italic", "oblique": "italic", "obl": "italic",
            "it": "italic", "i": "italic",
        }
        if residual in suffix_map:
            return suffix_map[residual]

        # Some font filenames do not include the family name (e.g. timesbi.ttf),
        # so also recognize well-known terminal suffixes in those cases.
        if normalized_stem != normalized_family:
            for suffix in sorted(suffix_map, key=len, reverse=True):
                if normalized_stem.endswith(suffix) and len(normalized_stem) > len(suffix):
                    return suffix_map[suffix]

        # Regular fonts often have no style suffix at all. A neutral filename,
        # or one containing only a weight such as "Light", is upright/non-bold.
        return "regular"

    @classmethod
    def _filename_matches_declared_style(
        cls,
        filename: str,
        family: str,
        declared_style: str,
    ) -> bool:
        """Check whether a configured style key agrees with its filename."""
        canonical_style = cls._normalize_font_name(declared_style)
        allowed_styles = {"regular", "reg", "roman", "normal", "book"}
        if canonical_style in allowed_styles:
            canonical_style = "regular"
        elif canonical_style in {"bolditalic", "boldoblique", "bi", "ib"}:
            canonical_style = "bolditalic"
        elif canonical_style in {"italic", "it", "i", "oblique", "obl"}:
            canonical_style = "italic"
        elif canonical_style in {"bold", "b", "bd", "heavy", "black", "extrabold", "semibold", "demibold"}:
            canonical_style = "bold"
        return cls._filename_style_hint(filename, family) == canonical_style

    def _load_local_font_records(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Index local fonts, validating configured style labels and deduplicating bytes."""
        records_by_hash: dict[str, dict[str, Any]] = {}
        missing: list[str] = []

        for family, styles in self.FONT_TABLE.items():
            for style, filename in styles.items():
                path = self.local_font_dir / filename
                if not path.is_file() or path.stat().st_size <= 0:
                    missing.append(filename)
                    continue
                try:
                    digest = self._sha256_file(path)
                except OSError as exc:
                    print(f"  Could not read reference font {path}: {exc}")
                    missing.append(filename)
                    continue

                record = records_by_hash.get(digest)
                if record is None:
                    record = {
                        "path": path,
                        "sha256": digest,
                        "families": set(),
                        "styles": set(),
                        "configured_entries": [],
                    }
                    records_by_hash[digest] = record

                matches = self._filename_matches_declared_style(filename, family, style)
                record["configured_entries"].append({
                    "family": family,
                    "declared_style": style,
                    "filename": filename,
                    "style_matches_filename": matches,
                })
                if matches:
                    record["families"].add(family)
                    record["styles"].add(style)
                else:
                    detected_style = self._filename_style_hint(filename, family)
                    print(
                        f"  WARNING: configured style mismatch: {family!r} / {style!r} "
                        f"points to {filename!r}, whose name suggests {detected_style!r}. "
                        "This entry will not be used for that family/style match."
                    )

        records = list(records_by_hash.values())
        if missing:
            unique_missing = list(dict.fromkeys(missing))
            print(
                f"Reference fonts available: {len(records)} unique file(s); "
                f"{len(unique_missing)} configured file(s) missing from {self.local_font_dir}."
            )
            preview = ", ".join(unique_missing[:10])
            if len(unique_missing) > 10:
                preview += f", ... (+{len(unique_missing) - 10} more)"
            print("  Missing: " + preview)
        else:
            print(f"Reference fonts available: {len(records)} unique file(s) in {self.local_font_dir}")
        return records, list(dict.fromkeys(missing))

    @classmethod
    def _normalize_font_name(cls, name: str) -> str:
        # Remove common six-uppercase-letter PDF subset prefixes such as ABCDEF+.
        name = re.sub(r"^[A-Z]{6}\+", "", str(name).strip())
        # Quotation marks, hyphens, punctuation, and whitespace are immaterial.
        return re.sub(r"[^a-z0-9]", "", name.casefold())

    @staticmethod
    def _fuzzy_family_score(normalized_name: str, normalized_family: str) -> float:
        """Compare a family name with any similarly sized substring of the PDF name."""
        if not normalized_name or not normalized_family:
            return 0.0
        if normalized_family in normalized_name:
            return 1.0

        family_length = len(normalized_family)
        min_length = max(3, int(family_length * 0.72))
        max_length = min(len(normalized_name), max(family_length + 3, int(family_length * 1.30)))
        best = 0.0
        for window_length in range(min_length, max_length + 1):
            for start in range(0, len(normalized_name) - window_length + 1):
                window = normalized_name[start : start + window_length]
                score = SequenceMatcher(None, normalized_family, window).ratio()
                if score > best:
                    best = score
        return best

    def select_reference_fonts(
        self,
        embedded_font_name: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Use a lenient family filter; if ambiguous/unmatched, return all local fonts."""
        normalized_name = self._normalize_font_name(embedded_font_name)
        if not self.font_records:
            return [], "no reference fonts available"

        exact_matches = [
            family
            for family in self.FONT_TABLE
            if self._normalize_font_name(family) in normalized_name
            and self._normalize_font_name(family)
        ]
        if exact_matches:
            # Prefer the longest family name when names overlap.
            family = max(exact_matches, key=lambda item: len(self._normalize_font_name(item)))
            selected = self.fonts_by_family.get(family, [])
            if selected:
                return selected, f"family match: {family}"

        fuzzy_scores = sorted(
            (
                (
                    self._fuzzy_family_score(normalized_name, self._normalize_font_name(family)),
                    family,
                )
                for family in self.FONT_TABLE
            ),
            reverse=True,
        )
        best_score, best_family = fuzzy_scores[0] if fuzzy_scores else (0.0, "")
        runner_up = fuzzy_scores[1][0] if len(fuzzy_scores) > 1 else 0.0
        if (
            best_score >= Configuration.FONT_NAME_FUZZY_THRESHOLD
            and best_score - runner_up >= Configuration.FONT_NAME_AMBIGUITY_MARGIN
        ):
            selected = self.fonts_by_family.get(best_family, [])
            if selected:
                return selected, f"fuzzy family match: {best_family} ({best_score:.2f})"

        return self.font_records, "no confident family match; checking all available fonts"

    def get_configured_font_record(
        self, family: str, style: str
    ) -> dict[str, Any] | None:
        """Return the validated local record for one configured family/style entry."""
        if style not in self.FONT_TABLE.get(family, {}):
            return None
        for record in self.font_records:
            for entry in record.get("configured_entries", []):
                if (
                    entry.get("family") == family
                    and entry.get("declared_style") == style
                    and entry.get("style_matches_filename") is True
                ):
                    return record
        return None

    @staticmethod
    def _render_glyph_chunk_worker(args: tuple) -> dict[str, tuple[int, int]]:
        """Render a chunk of a font's glyph outlines in a worker process."""
        glyph_names, font_path, image_size, supersample = args
        font = TTFont(font_path)
        try:
            glyph_set = font.getGlyphSet()
            units_per_em = int(font["head"].unitsPerEm)
            results: dict[str, tuple[int, int]] = {}
            for name in glyph_names:
                try:
                    recording = DecomposingRecordingPen(glyph_set)
                    glyph_set[name].draw(recording)
                    pen = GlyphOutlineRenderer._OutlinePen()
                    recording.replay(pen)
                    rendered = GlyphOutlineRenderer.render_contours(
                        pen.finish(), units_per_em, image_size, supersample
                    )
                    results[name] = (rendered.mask, rendered.pixel_count)
                except Exception:
                    results[name] = (0, 0)
            return results
        finally:
            font.close()

    def render_reference_glyphs(self, font_path: Path) -> tuple[list[dict[str, Any]], int]:
        """Render all distinct Unicode glyphs from one reference font."""
        cached_key = str(font_path.resolve()).casefold()
        if cached_key in self._rendered_cache:
            print(f"  Reusing rendered glyphs: {font_path.name}")
            return self._rendered_cache[cached_key]

        font = TTFont(str(font_path))
        try:
            cmap = font.getBestCmap() or {}
            units_per_em = int(font["head"].unitsPerEm)
            unique_glyph_names = sorted(set(cmap.values()))
        finally:
            font.close()

        if not unique_glyph_names:
            raise RuntimeError(f"No Unicode cmap entries in reference font: {font_path}")

        print(f"  Rendering {len(unique_glyph_names)} unique glyph(s) from {font_path.name}...")
        num_workers = min(os.cpu_count() or 1, 4)
        chunk_size = max(1, math.ceil(len(unique_glyph_names) / max(1, num_workers)))
        chunks = [
            (
                unique_glyph_names[i : i + chunk_size],
                str(font_path),
                Configuration.IMAGE_SIZE,
                Configuration.RASTER_SUPERSAMPLE,
            )
            for i in range(0, len(unique_glyph_names), chunk_size)
        ]

        masks_by_name: dict[str, tuple[int, int]] = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._render_glyph_chunk_worker, chunk) for chunk in chunks]
            for completed, future in enumerate(as_completed(futures), start=1):
                masks_by_name.update(future.result())
                if len(chunks) > 1:
                    print(f"    Render chunk {completed}/{len(chunks)} complete")

        font = TTFont(str(font_path))
        try:
            cmap = font.getBestCmap() or {}
        finally:
            font.close()

        candidates: list[dict[str, Any]] = []
        for codepoint, glyph_name in sorted(cmap.items()):
            mask, pixel_count = masks_by_name.get(glyph_name, (0, 0))
            if pixel_count:
                candidates.append(
                    {
                        "codepoint": int(codepoint),
                        "glyph_name": glyph_name,
                        "mask": mask,
                        "pixel_count": pixel_count,
                    }
                )

        if not candidates:
            raise RuntimeError(f"No usable glyph outlines were found in reference font: {font_path}")

        result = (candidates, units_per_em)
        self._rendered_cache[cached_key] = result
        print(f"  Usable Unicode glyphs: {len(candidates)}")
        return result

    def get_rendered_reference_glyphs(self, font_path: Path) -> tuple[list[dict[str, Any]], int]:
        return self.render_reference_glyphs(font_path)

    def score_candidate(
        self,
        pdf_glyph: GlyphOutlineRenderer.RenderedGlyph,
        reference_candidates: list[dict[str, Any]],
    ) -> tuple[int, str, float]:
        """Return the highest raw-IoU Unicode candidate; only exact ties use preference."""
        if pdf_glyph.is_empty:
            return 0x20, "space", 0.0
        if not reference_candidates:
            raise RuntimeError("The reference font has no usable glyph candidates")

        scored: list[tuple[float, int, str]] = []
        for candidate in reference_candidates:
            score = GlyphOutlineRenderer.overlap_score(
                pdf_glyph.mask,
                pdf_glyph.pixel_count,
                candidate["mask"],
                candidate["pixel_count"],
            )
            scored.append((score, int(candidate["codepoint"]), str(candidate["glyph_name"])))

        scored.sort(
            key=lambda item: (
                -item[0],
                GlyphOutlineRenderer.unicode_preference(item[1]),
                item[1],
                item[2],
            )
        )
        best_score, best_codepoint, best_glyph_name = scored[0]
        return best_codepoint, best_glyph_name, float(best_score)


class CidFontDecoder:
    """Extract embedded CID-keyed CFF fonts and build in-memory Unicode mappings."""

    CID_NAME_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"cid0*(\d+)", re.IGNORECASE),
        re.compile(r"g0*(\d+)", re.IGNORECASE),
        re.compile(r"glyph0*(\d+)", re.IGNORECASE),
        re.compile(r"(\d+)"),
    ]

    def __init__(self, reference_library: ReferenceFontLibrary):
        self.reference_library = reference_library
        # Kept in memory for diagnosis; nothing is written to CSV/JSON.
        self.last_comparison_results: dict[int, dict[str, Any]] = {}

    @staticmethod
    def extract_font(
        pdf_doc: pymupdf.Document,
        xref: int,
    ) -> tuple[str, Any, dict[int, int], dict[int, int], int]:
        """Extract a CID-keyed CFF font and validate its CID/GID name table."""
        info = pdf_doc.extract_font(xref)
        if not info or len(info) < 4 or not info[3]:
            raise RuntimeError(f"Cannot extract embedded font xref {xref}.")

        cff = CFFFontSet()
        try:
            cff.decompile(io.BytesIO(info[3]), None)
        except Exception as exc:
            raise RuntimeError(
                f"Font xref {xref} is not a readable CFF font; skipping it. ({exc})"
            ) from exc
        if not cff.topDictIndex:
            raise RuntimeError(f"No CFF top dictionary found in font xref {xref}.")

        top = cff.topDictIndex[0]
        if getattr(top, "ROS", None) is None:
            raise RuntimeError(
                f"Font xref {xref} is a simple (non-CID) CFF font; skipping it."
            )

        font_matrix = getattr(top, "FontMatrix", None) or [0.001, 0, 0, 0.001, 0, 0]
        try:
            cff_units_per_em = round(1.0 / abs(float(font_matrix[0])))
            if cff_units_per_em <= 0:
                raise ValueError("invalid FontMatrix scale")
        except (ValueError, TypeError, ZeroDivisionError, IndexError):
            cff_units_per_em = 1000

        charset = list(top.charset)
        if not charset:
            raise RuntimeError(f"Font xref {xref}: CFF charset is empty.")

        selected_maps: tuple[dict[int, int], dict[int, int], str] | None = None
        saw_cid_like_name = False
        for pattern in CidFontDecoder.CID_NAME_PATTERNS:
            candidate_gid_to_cid: dict[int, int] = {}
            candidate_cid_to_gid: dict[int, int] = {}
            valid = True
            for gid, glyph_name in enumerate(charset):
                name = str(glyph_name)
                if gid == 0 and name.casefold() == ".notdef":
                    continue
                match = pattern.fullmatch(name)
                if match is None:
                    valid = False
                    continue
                saw_cid_like_name = True
                cid = int(match.group(1))
                if cid in candidate_cid_to_gid:
                    valid = False
                    continue
                candidate_gid_to_cid[gid] = cid
                candidate_cid_to_gid[cid] = gid
            if valid and candidate_gid_to_cid:
                selected_maps = (
                    candidate_gid_to_cid,
                    candidate_cid_to_gid,
                    pattern.pattern,
                )
                break

        if selected_maps is None and saw_cid_like_name:
            raise RuntimeError(
                f"Font xref {xref} has partial or duplicate CID naming; refusing an "
                "incomplete CID/GID map."
            )

        if selected_maps is None:
            # If the charset does not encode CIDs in its names, retain the previous
            # Identity CID == GID assumption and make that assumption explicit.
            gid_to_cid = {gid: gid for gid in range(len(charset))}
            cid_to_gid = {cid: cid for cid in range(len(charset))}
            pattern_used = "identity fallback (CID == GID)"
            print(
                "  WARNING: CFF charset names do not provide a complete, unique CID map; "
                "assuming CID == GID."
            )
        else:
            gid_to_cid, cid_to_gid, pattern_used = selected_maps

        print(f"  CID naming rule: {pattern_used} ({len(cid_to_gid)} mapped glyphs)")
        return info[0], top, gid_to_cid, cid_to_gid, cff_units_per_em

    @staticmethod
    def _render_embedded_glyphs(
        top: Any,
        cid_to_gid: dict[int, int],
        cff_units_per_em: int,
    ) -> dict[int, dict[str, Any]]:
        """Render each embedded CID exactly once before testing reference fonts."""
        charstrings = top.CharStrings
        rendered_by_cid: dict[int, dict[str, Any]] = {}
        for cid in sorted(cid_to_gid):
            gid = cid_to_gid[cid]
            glyph_name = str(top.charset[gid])
            try:
                pen = GlyphOutlineRenderer._OutlinePen()
                charstrings[glyph_name].draw(pen)
                rendered = GlyphOutlineRenderer.render_contours(pen.finish(), cff_units_per_em)
            except Exception as exc:
                print(f"  CID {cid}: could not render outline ({exc}); treating it as empty.")
                rendered = GlyphOutlineRenderer.RenderedGlyph(0, 0, None)
            rendered_by_cid[cid] = {
                "gid": gid,
                "cff_glyph_name": glyph_name,
                "rendered": rendered,
            }
        return rendered_by_cid

    def _score_one_font(
        self,
        embedded_glyphs: dict[int, dict[str, Any]],
        reference_candidates: list[dict[str, Any]],
    ) -> dict[int, tuple[int, str, float]]:
        """Return each CID's best Unicode, glyph name, and score for one font."""
        cid_best: dict[int, tuple[int, str, float]] = {}
        for cid, entry in embedded_glyphs.items():
            rendered = entry["rendered"]
            if rendered.is_empty:
                # An empty outline may be represented as a space, but is not a match.
                cid_best[cid] = (0x20, "space", 0.0)
            else:
                cid_best[cid] = self.reference_library.score_candidate(
                    rendered, reference_candidates
                )
        return cid_best

    @staticmethod
    def _average_score(cid_best: dict[int, tuple[int, str, float]]) -> float:
        if not cid_best:
            return 0.0
        return sum(result[2] for result in cid_best.values()) / len(cid_best)

    def _choose_best_cross_font_results(
        self,
        font_results: list[dict[str, Any]],
        expected_cids: set[int],
    ) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
        """Select highest raw score; font average is used only inside the tie band."""
        final_mapping: dict[int, str] = {}
        winners: dict[int, dict[str, Any]] = {}

        for cid in sorted(expected_cids):
            candidates: list[dict[str, Any]] = []
            for result in font_results:
                best = result["cid_best"].get(cid)
                if best is None:
                    continue
                codepoint, glyph_name, score = best
                candidates.append(
                    {
                        "font": result["font"],
                        "font_average_score": result["average_score"],
                        "codepoint": codepoint,
                        "glyph_name": glyph_name,
                        "score": float(score),
                    }
                )

            if not candidates:
                raise RuntimeError(f"No reference-font candidate was scored for CID {cid}.")

            best_score = max(item["score"] for item in candidates)
            close = [
                item for item in candidates
                if item["score"] >= best_score - Configuration.CROSS_FONT_TIE_TOLERANCE
            ]
            close.sort(
                key=lambda item: (
                    -item["font_average_score"],
                    -item["score"],
                    GlyphOutlineRenderer.unicode_preference(item["codepoint"]),
                    item["codepoint"],
                    str(item["font"]["path"]).casefold(),
                )
            )
            winner = close[0]
            final_mapping[cid] = chr(winner["codepoint"])
            winners[cid] = winner

        return final_mapping, winners

    def build_mapping(
        self,
        pdf_doc: pymupdf.Document,
        xref: int,
        embedded_font_name: str,
    ) -> tuple[dict[int, int], dict[int, str]] | None:
        """Build all per-font/per-CID scores and assign every CID a winner."""
        try:
            (
                _embedded_name,
                top,
                gid_to_cid,
                cid_to_gid,
                cff_units_per_em,
            ) = self.extract_font(pdf_doc, xref)
        except RuntimeError as exc:
            message = str(exc)
            if (
                "Cannot extract" in message
                or "No CFF top dictionary" in message
                or "simple (non-CID) CFF font" in message
                or "not a readable CFF font" in message
                or "partial or duplicate CID naming" in message
            ):
                print(f"  Skipping xref {xref}: {message}")
                return None
            raise

        print(
            f"\nProcessing xref {xref}: {embedded_font_name} "
            f"({len(cid_to_gid)} CIDs)"
        )
        selected_fonts, selection_reason = self.reference_library.select_reference_fonts(
            embedded_font_name
        )
        print(f"  Reference-font selection: {selection_reason}")
        if selected_fonts:
            print("  Candidate reference fonts:")
            for candidate in selected_fonts:
                style_label = ", ".join(sorted(candidate.get("styles", set()))) or "style unclassified"
                family_label = ", ".join(sorted(candidate.get("families", set()))) or "unclassified family"
                print(f"    {candidate['path'].name} [{family_label}; {style_label}]")
        if not selected_fonts:
            print("  No usable reference fonts are available; cannot map this embedded font.")
            return None

        embedded_glyphs = self._render_embedded_glyphs(top, cid_to_gid, cff_units_per_em)
        font_results: list[dict[str, Any]] = []
        for font in selected_fonts:
            path = font["path"]
            try:
                reference_glyphs, _reference_units = (
                    self.reference_library.get_rendered_reference_glyphs(path)
                )
                cid_best = self._score_one_font(embedded_glyphs, reference_glyphs)
            except Exception as exc:
                print(f"  Could not score reference font {path.name}: {exc}")
                continue

            average_score = self._average_score(cid_best)
            family_names = ", ".join(sorted(font["families"])) or "unclassified"
            style_names = ", ".join(sorted(font.get("styles", set()))) or "unclassified"
            print(
                f"  {path.name}: average best-CID IoU={average_score:.4f} "
                f"(families: {family_names}; styles: {style_names})"
            )
            font_results.append(
                {
                    "font": font,
                    "average_score": average_score,
                    "cid_best": cid_best,
                }
            )

        if not font_results:
            print(f"  No reference font could be rendered for xref {xref}; skipping mapping.")
            return None

        expected_cids = set(cid_to_gid)
        final_mapping, winners = self._choose_best_cross_font_results(font_results, expected_cids)
        final_average = (
            sum(item["score"] for item in winners.values()) / len(winners)
            if winners else 0.0
        )
        fallback_applied = False
        original_final_average = final_average

        if final_average < Configuration.FINAL_MAPPING_MIN_AVERAGE_IOU:
            fallback_family = Configuration.FALLBACK_FONT_FAMILY
            fallback_style = Configuration.FALLBACK_FONT_STYLE
            fallback_font = self.reference_library.get_configured_font_record(
                fallback_family, fallback_style
            )
            if fallback_font is None:
                print(
                    f"  WARNING: Final mapping average IoU {final_average:.4f} is below "
                    f"{Configuration.FINAL_MAPPING_MIN_AVERAGE_IOU:.2f}, but "
                    f"{fallback_family} {fallback_style} is not available. "
                    "Keeping the highest-scoring mapping from the selected fonts."
                )
            else:
                try:
                    fallback_glyphs, _fallback_units = (
                        self.reference_library.get_rendered_reference_glyphs(
                            fallback_font["path"]
                        )
                    )
                    fallback_cid_best = self._score_one_font(
                        embedded_glyphs, fallback_glyphs
                    )
                    fallback_average = self._average_score(fallback_cid_best)
                    fallback_result = {
                        "font": fallback_font,
                        "average_score": fallback_average,
                        "cid_best": fallback_cid_best,
                    }
                    final_mapping, winners = self._choose_best_cross_font_results(
                        [fallback_result], expected_cids
                    )
                    final_average = (
                        sum(item["score"] for item in winners.values()) / len(winners)
                        if winners else 0.0
                    )
                    fallback_applied = True
                    print(
                        f"  Final mapping average IoU {original_final_average:.4f} is below "
                        f"{Configuration.FINAL_MAPPING_MIN_AVERAGE_IOU:.2f}; "
                        f"falling back to {fallback_family} {fallback_style}. "
                        f"Fallback average IoU={fallback_average:.4f}."
                    )
                except Exception as exc:
                    print(
                        f"  WARNING: Could not score {fallback_family} {fallback_style} "
                        f"for fallback ({exc}); keeping the highest-scoring mapping "
                        "from the selected fonts."
                    )

        self.last_comparison_results[xref] = {
            "embedded_font_name": embedded_font_name,
            "selection_reason": selection_reason,
            "font_results": font_results,
            "winners": winners,
            "original_final_average_iou": original_final_average,
            "final_average_iou": final_average,
            "fallback_applied": fallback_applied,
        }

        #print("  Final CID assignments:")
        for cid in sorted(winners):
            winner = winners[cid]
            character = final_mapping[cid]
            #print(
            #    f"    CID {cid:4} -> {character!r} "
            #    f"(U+{ord(character):04X}, score={winner['score']:.4f}, "
            #    f"font={winner['font']['path'].name})"
            #)
        print(f"  Final mapping average IoU: {final_average:.4f}")
        return gid_to_cid, final_mapping


class PdfTextDecoder:
    """Decodes PDF text traces and reconstructs readable page text."""

    def __init__(
        self,
        fonts_db: dict[int, tuple[dict[int, int], dict[int, str]]],
    ):
        self.fonts_db = fonts_db
        self.spatial_index: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.available_indices: set[int] = set()
        self.known_fonts: list[tuple[str, int]] = []

    @staticmethod
    def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        return cx, cy

    @staticmethod
    def _sanitize_char(char: str) -> str:
        """
        Keep printable text, normalize Unicode spaces, and discard controls.

        A normal ASCII space is retained. Tabs and line-control characters are
        discarded because line breaks are reconstructed from the PDF's line
        structure and horizontal gaps are handled by layout logic.
        """
        if not char:
            return ""
        if char == " ":
            return " "
        if char in "\t\r\n\v\f":
            return ""
        if char.isspace():
            return " "
        return char if char.isprintable() else ""

    def _build_spatial_index(self, decoded_chars: list[dict[str, Any]]) -> None:
        self.spatial_index.clear()
        self.available_indices = set(range(len(decoded_chars)))
        for idx, item in enumerate(decoded_chars):
            cx, cy = self._bbox_center(item["bbox"])
            bucket = (
                int(cx // Configuration.BUCKET_SIZE),
                int(cy // Configuration.BUCKET_SIZE),
            )
            self.spatial_index[bucket].append(idx)

    def _build_known_fonts(self, page: pymupdf.Page) -> None:
        self.known_fonts = []
        for font_info in page.get_fonts(full=True):
            if len(font_info) < 5:
                continue
            xref = int(font_info[0])
            full_font_name = font_info[3] or ""
            if xref in self.fonts_db and full_font_name:
                self.known_fonts.append((full_font_name, xref))

    def _resolve_span_xref(self, span_font_name: str) -> int | None:
        if not span_font_name:
            return None
        for full_name, xref in self.known_fonts:
            if span_font_name in full_name or full_name in span_font_name:
                return xref
        return None

    def decode_trace_characters(
        self,
        page: pymupdf.Page,
        page_number: int,
    ) -> list[dict[str, Any]]:
        """
        Get Unicode for target-font glyphs using MuPDF's rendered character traces.
        Resolves span font names to xrefs, then looks up mappings by xref.
        """
        self._build_known_fonts(page)

        decoded: list[dict[str, Any]] = []
        span_count = 0
        char_count = 0
        matched_count = 0
        direct_cid_hits = 0
        gid_fallback_hits = 0
        unmapped_trace_ids = 0
        unresolved_spans = 0

        for span in page.get_texttrace():
            span_count += 1
            trace_font_name = span.get("font", "")
            if not trace_font_name:
                continue

            resolved_xref = self._resolve_span_xref(trace_font_name)
            if resolved_xref is None:
                unresolved_spans += 1
                continue

            gid_to_cid, cid_mapping = self.fonts_db[resolved_xref]

            for char_entry in span.get("chars", []):
                char_count += 1
                if len(char_entry) < 4:
                    continue

                raw_id = int(char_entry[1])
                bbox = char_entry[3]

                # Preserve the lookup order from the previously working version:
                # try the trace identifier as a CID first, then as a GID.
                character = cid_mapping.get(raw_id)
                if character is not None:
                    direct_cid_hits += 1
                else:
                    cid = gid_to_cid.get(raw_id)
                    character = cid_mapping.get(cid) if cid is not None else None
                    if character is not None:
                        gid_fallback_hits += 1

                if character is not None:
                    matched_count += 1
                    decoded.append(
                        {
                            "xref": resolved_xref,
                            "character": character,
                            "bbox": bbox,
                        }
                    )
                else:
                    unmapped_trace_ids += 1

        print(
            f"[DEBUG PAGE {page_number}] Summary: spans={span_count}, "
            f"chars_checked={char_count}, matched={matched_count}, "
            f"direct_CID_hits={direct_cid_hits}, GID_fallback_hits={gid_fallback_hits}, "
            f"unmapped_trace_ids={unmapped_trace_ids}, "
            f"unresolved_spans={unresolved_spans}"
        )
        return decoded

    def _find_best_match_in_candidates(
        self,
        cx: float,
        cy: float,
        candidates: list[int],
        decoded_chars: list[dict[str, Any]],
        expected_xref: int | None = None,
    ) -> tuple[int | None, float]:
        best_idx = None
        best_dist_sq = float("inf")
        for idx in candidates:
            if idx not in self.available_indices:
                continue
            if expected_xref is not None and decoded_chars[idx].get("xref") != expected_xref:
                continue
            other_bbox = decoded_chars[idx]["bbox"]
            ox, oy = self._bbox_center(other_bbox)
            dist_sq = (cx - ox) ** 2 + (cy - oy) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = idx
        return best_idx, math.sqrt(best_dist_sq) if best_idx is not None else float("inf")

    def _candidate_indices_near(self, cx: float, cy: float) -> list[int]:
        bucket_x = int(cx // Configuration.BUCKET_SIZE)
        bucket_y = int(cy // Configuration.BUCKET_SIZE)
        candidates: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(
                    self.spatial_index.get((bucket_x + dx, bucket_y + dy), [])
                )
        return list(dict.fromkeys(candidates))

    def _match_decoded_chars_to_raw_chars(
        self,
        raw: dict[str, Any],
        decoded_chars: list[dict[str, Any]],
        page_number: int,
    ) -> dict[int, str]:
        replacements: dict[int, str] = {}
        exact_matches = 0
        local_search_successes = 0
        global_fallback_attempts = 0
        global_fallback_successes = 0
        still_unmatched_count = 0

        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    xref = self._resolve_span_xref(span.get("font", ""))
                    if xref is None:
                        continue

                    for char_obj in span.get("chars", []):
                        box = char_obj.get("bbox", (0, 0, 0, 0))
                        cx, cy = self._bbox_center(box)
                        matched_trace_index = None
                        candidates = self._candidate_indices_near(cx, cy)
                        if candidates:
                            best_idx, best_dist = self._find_best_match_in_candidates(
                                cx, cy, candidates, decoded_chars, expected_xref=xref
                            )
                            if best_idx is not None and best_dist < Configuration.LOCAL_MATCH_DISTANCE:
                                matched_trace_index = best_idx
                                exact_matches += 1
                                local_search_successes += 1

                        # Search globally whenever the local candidate set has no acceptable match,
                        # not only when the nearby buckets are completely empty.
                        if matched_trace_index is None:
                            global_fallback_attempts += 1
                            all_available = list(self.available_indices)
                            if all_available:
                                best_idx, best_dist = self._find_best_match_in_candidates(
                                    cx, cy, all_available, decoded_chars, expected_xref=xref
                                )
                                if best_idx is not None and best_dist < Configuration.GLOBAL_MATCH_DISTANCE:
                                    matched_trace_index = best_idx
                                    global_fallback_successes += 1

                        if matched_trace_index is not None:
                            self.available_indices.discard(matched_trace_index)
                            replacements[id(char_obj)] = self._sanitize_char(
                                decoded_chars[matched_trace_index]["character"]
                            )
                        else:
                            original = char_obj.get("c", "")
                            if original.isspace():
                                replacements[id(char_obj)] = self._sanitize_char(original)
                            else:
                                replacements[id(char_obj)] = ""
                                still_unmatched_count += 1
                                #if still_unmatched_count <= 5:
                                    #print(
                                    #    f"[DEBUG PAGE {page_number}] unmatched raw char: "
                                    #    f"c={original!r} bbox={box} "
                                    #    f"font={span.get('font', '')} xref={xref}"
                                    #)


        if still_unmatched_count > 0:
            print(
                f"WARNING Page {page_number}: "
                f"{still_unmatched_count} characters had no trace match and were skipped, Most likely control characters."
            )

        print(
            f"[BBOX PAGE {page_number}] exact={exact_matches}, "
            f"local_successes={local_search_successes}, "
            f"global_attempts={global_fallback_attempts}, "
            f"global_successes={global_fallback_successes}"
        )
        return replacements

    def _assemble_line(
        self,
        line: dict[str, Any],
        replacements: dict[int, str],
    ) -> str:
        line_chars: list[tuple[str, tuple[float, float, float, float], float]] = []
        for span in line.get("spans", []):
            xref = self._resolve_span_xref(span.get("font", ""))
            is_target = xref is not None
            for char_obj in span.get("chars", []):
                if is_target:
                    text = replacements.get(id(char_obj), "")
                else:
                    text = self._sanitize_char(char_obj.get("c", ""))

                bbox = char_obj.get("bbox", (0, 0, 0, 0))
                line_chars.append((text, bbox, float(span.get("size", 10.0))))

        built: list[str] = []
        previous_bbox: tuple[float, float, float, float] | None = None
        previous_size = 10.0
        horizontal = abs(float(line.get("dir", (1, 0))[1])) < Configuration.HORIZONTAL_LINE_ANGLE_TOLERANCE

        for text, bbox, size in line_chars:
            x0, y0, x1, y1 = [float(v) for v in bbox[:4]]

            if previous_bbox is not None and horizontal:
                gap = x0 - previous_bbox[2]
                threshold = max(
                    Configuration.MIN_SPACE_GAP,
                    min(size, previous_size) * Configuration.SPACE_GAP_FRACTION,
                )
                built_text = "".join(built)
                if (
                    gap > threshold
                    and built_text
                    and not built_text[-1].isspace()
                    and text
                    and not text[0].isspace()
                ):
                    built.append(" ")

            if text:
                built.append(text)

            previous_bbox = (x0, y0, x1, y1)
            previous_size = size

        line_text = re.sub(r" {2,}", " ", "".join(built)).strip(" ")
        return line_text

    def assemble_page_text(
        self,
        page: pymupdf.Page,
        decoded_trace_chars: list[dict[str, Any]],
        page_number: int,
    ) -> str:
        """
        Reconstruct text by overlaying decoded characters onto the rawdict structure.
        Uses spatial hashing and nearest-neighbor matching to handle float drift.
        """
        raw = page.get_text("rawdict", sort=False)
        self._build_known_fonts(page)
        self._build_spatial_index(decoded_trace_chars)
        replacements = self._match_decoded_chars_to_raw_chars(
            raw,
            decoded_trace_chars,
            page_number,
        )

        output_blocks: list[str] = []
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_lines = [
                self._assemble_line(line, replacements)
                for line in block.get("lines", [])
            ]
            block_text = "\n".join(block_lines).strip("\n")
            if block_text:
                output_blocks.append(block_text)

        return "\n\n".join(output_blocks)


class PdfDecoder:
    """Coordinate font discovery, in-memory mapping, page decoding, and text output."""

    def __init__(
        self,
        pdf_path: Path,
        output_dir: Path | None = None,
        reference_library: ReferenceFontLibrary | None = None,
    ):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir) if output_dir is not None else self.pdf_path.parent
        self.reference_library = reference_library or ReferenceFontLibrary()
        self.cid_decoder = CidFontDecoder(self.reference_library)
        self.txt_path = self.output_dir / f"{self.pdf_path.stem}_decoded.txt"
        self.fonts_db: dict[int, tuple[dict[int, int], dict[int, str]]] = {}

    def _discover_font_xrefs(self, pdf: pymupdf.Document) -> dict[int, str]:
        xref_to_font_info: dict[int, str] = {}
        for page in pdf:
            for font_info in page.get_fonts(full=True):
                if len(font_info) < 5:
                    continue
                xref = int(font_info[0])
                basefont = font_info[3] or "Unknown"
                if xref not in xref_to_font_info:
                    print(
                        f"  Discovered font: xref={xref}, basefont='{basefont}', "
                        f"type='{font_info[2]}'"
                    )
                    xref_to_font_info[xref] = basefont
        return xref_to_font_info

    def build_font_mappings(self, pdf: pymupdf.Document) -> dict[int, str]:
        xref_to_font_info = self._discover_font_xrefs(pdf)
        print(
            f"\nFound {len(xref_to_font_info)} unique font xrefs. "
            "Scanning for CID-keyed CFF fonts..."
        )

        for xref, full_font_name in xref_to_font_info.items():
            built = self.cid_decoder.build_mapping(pdf, xref, full_font_name)
            if built is None:
                continue
            gid_to_cid, cid_mapping = built
            self.fonts_db[xref] = (gid_to_cid, cid_mapping)

        return xref_to_font_info

    def decode_pages(self, pdf: pymupdf.Document) -> tuple[list[str], int]:
        if not self.fonts_db:
            raise RuntimeError("No usable CID-keyed CFF font mappings were built.")

        print(
            f"\nSuccessfully built mappings for {len(self.fonts_db)} font(s). "
            "Decoding pages..."
        )

        text_decoder = PdfTextDecoder(self.fonts_db)
        decoded_pages: list[str] = []
        total_decoded_chars = 0

        for page_index in range(len(pdf)):
            page = pdf[page_index]
            page_number = page_index + 1
            decoded_trace_chars = text_decoder.decode_trace_characters(page, page_number)
            decoded_page = text_decoder.assemble_page_text(page, decoded_trace_chars, page_number)
            decoded_pages.append(decoded_page)
            total_decoded_chars += len(decoded_trace_chars)
            print(
                f"Decoded page {page_number}/{len(pdf)} "
                f"({len(decoded_trace_chars)} mapped glyphs)"
            )

        if total_decoded_chars == 0:
            raise RuntimeError("No glyphs from the mapped fonts were found in the text trace.")

        return decoded_pages, total_decoded_chars

    def run(self) -> None:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        pdf = pymupdf.open(self.pdf_path)
        try:
            self.build_font_mappings(pdf)
            decoded_pages, total_decoded_chars = self.decode_pages(pdf)
            if len(decoded_pages) > 1:
                page_sections = [
                    f"\n\nPage {page_number} of {len(decoded_pages)}\n\n{page_text}"
                    for page_number, page_text in enumerate(decoded_pages, start=1)
                ]
                output_text = "\n\n".join(page_sections)
            else:
                output_text = decoded_pages[0] if decoded_pages else ""
            self.txt_path.write_text(output_text, encoding="utf-8")
            print(
                f"\nFinished. Decoded {total_decoded_chars} characters across "
                f"{len(self.fonts_db)} font(s)."
            )
            print(f"Output: {self.txt_path}")
            print("Mapping scores were kept in memory; no CSV or JSON files were written.")
        finally:
            pdf.close()

def _collect_pdf_paths(input_path: Path, recursive: bool = False) -> list[Path]:
    """Resolve a single PDF file or a directory containing PDF files."""
    input_path = Path(input_path).expanduser()
    if input_path.is_file():
        if input_path.suffix.casefold() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path.resolve()]
    if input_path.is_dir():
        candidates = input_path.rglob("*") if recursive else input_path.iterdir()
        pdf_paths = sorted(
            (path.resolve() for path in candidates
             if path.is_file() and path.suffix.casefold() == ".pdf"),
            key=lambda path: str(path).casefold(),
        )
        if not pdf_paths:
            depth_text = " recursively" if recursive else " in this folder"
            raise FileNotFoundError(f"No PDF files found{depth_text}: {input_path}")
        return pdf_paths
    raise FileNotFoundError(f"Input PDF or folder not found: {input_path}")

def main(argv: list[str] | None = None) -> int:
    """Run one PDF or every PDF in a folder. Defaults to Configuration.PDF_PATH."""
    parser = argparse.ArgumentParser(
        description=(
            "Experimental recovery of text from PDFs that use CID-keyed CFF fonts "
            "without a usable ToUnicode map."
        )
    )
    parser.add_argument(
        "input", nargs="?", type=Path, default=None,
        help=(
            "PDF file or folder containing PDFs. If omitted, use Configuration.PDF_PATH "
            "from this script."
        ),
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="When input is a folder, also process PDFs in subfolders.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Directory for decoded text files (default: beside each input PDF).",
    )
    args = parser.parse_args(argv)

    input_path = args.input if args.input is not None else Configuration.PDF_PATH
    pdf_paths = _collect_pdf_paths(input_path, recursive=args.recursive)
    # Share the rendered-reference-font cache across PDFs in a folder batch.
    reference_library = ReferenceFontLibrary()
    for index, pdf_path in enumerate(pdf_paths, start=1):
        if len(pdf_paths) > 1:
            print(f"\n===== PDF {index} of {len(pdf_paths)}: {pdf_path.name} =====")
        PdfDecoder(
            pdf_path,
            output_dir=args.output_dir,
            reference_library=reference_library,
        ).run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise
