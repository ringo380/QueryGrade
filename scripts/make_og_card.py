#!/usr/bin/env python3
"""Regenerate the og:image social card at analyzer/static/analyzer/img/og-card.png.

Run this after changing the logo, the brand colors, or the tagline:

    python scripts/make_og_card.py
    python manage.py collectstatic --noinput   # refresh the whitenoise manifest

Requires Google Chrome (headless) for rendering. Standard library only
otherwise, so it runs without installing anything.

Why 1200x630: Facebook, LinkedIn, and X all accept that size without
re-cropping. Why PNG: all three reject SVG and render no preview image at
all, which is the bug this asset exists to fix (issue #107).

The logo is read from logo-mark.svg rather than redrawn here, so the card
cannot drift from the real mark. Fonts are base64-embedded because a file://
render cannot load them cross-origin.

analyzer/test_seo.py asserts the committed PNG is 1200x630 and matches the
og:image:width/height meta tags, so a regenerated card that changes size
fails the suite rather than silently shipping.
"""

import base64
import pathlib
import re
import struct
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
STATIC = REPO / "analyzer/static/analyzer"
OUT = STATIC / "img/og-card.png"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

WIDTH, HEIGHT = 1200, 630

# Brand palette, matching logo-mark.svg and the Tailwind classes in base.html.
INDIGO = "#4f46e5"
EMERALD = "#10b981"
SLATE_900 = "#0f172a"
SLATE_300 = "#cbd5e1"
SLATE_400 = "#94a3b8"
SLATE_700 = "#334155"

# Keep in sync with the og:description block in base.html. Line breaks are
# explicit: at 36px the longest line below is ~756px, well inside the 1040px
# content box, so no line reflows onto an orphan word.
TAGLINE_LINES = [
    "Analyze and grade your SQL queries,",
    "flag performance issues, and get",
    "index recommendations.",
]


def b64(path):
    return base64.b64encode(path.read_bytes()).decode()


def logo_svg():
    """Inline logo-mark.svg so the card tracks the real logo."""
    svg = (STATIC / "img/logo-mark.svg").read_text()
    # drop the XML declaration if one is ever added; keep the <svg> element
    match = re.search(r"<svg.*</svg>", svg, flags=re.S)
    if not match:
        sys.exit("could not parse logo-mark.svg")
    return match.group(0)


def build_html():
    font_600 = b64(STATIC / "fonts/jetbrains-mono-600.woff2")
    font_500 = b64(STATIC / "fonts/jetbrains-mono-500.woff2")
    tagline = "<br>\n    ".join(TAGLINE_LINES)
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  @font-face {{
    font-family: 'JetBrains Mono'; font-weight: 600; font-style: normal;
    src: url(data:font/woff2;base64,{font_600}) format('woff2');
  }}
  @font-face {{
    font-family: 'JetBrains Mono'; font-weight: 500; font-style: normal;
    src: url(data:font/woff2;base64,{font_500}) format('woff2');
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {WIDTH}px; height: {HEIGHT}px; }}
  body {{
    background: {SLATE_900};
    font-family: 'JetBrains Mono', monospace;
    /* 80px keeps every element inside the safe area platforms may crop */
    padding: 80px;
    display: flex; flex-direction: column; justify-content: center;
    position: relative; overflow: hidden;
  }}
  .accent {{
    position: absolute; top: 0; left: 0; width: 100%; height: 10px;
    background: linear-gradient(90deg, {INDIGO} 0%, {EMERALD} 100%);
  }}
  .brand {{ display: flex; align-items: center; gap: 28px; margin-bottom: 40px; }}
  .brand svg {{ width: 104px; height: 104px; flex: none; }}
  h1 {{ font-size: 82px; font-weight: 600; color: #ffffff; letter-spacing: -2px; }}
  .tagline {{
    font-size: 36px; font-weight: 500; color: {SLATE_300};
    line-height: 1.5; max-width: 1040px;
  }}
  .footer {{
    margin-top: 52px; display: flex; align-items: center; gap: 20px;
    font-size: 28px; font-weight: 500;
  }}
  .domain {{ color: {EMERALD}; }}
  .rule {{ width: 3px; height: 30px; background: {SLATE_700}; }}
  .kicker {{ color: {SLATE_400}; }}
</style></head>
<body>
  <div class="accent"></div>
  <div class="brand">
    {logo_svg()}
    <h1>QueryGrade</h1>
  </div>
  <div class="tagline">
    {tagline}
  </div>
  <div class="footer">
    <span class="domain">querygrade.com</span>
    <span class="rule"></span>
    <span class="kicker">SQL query analysis &amp; grading</span>
  </div>
</body></html>
"""


def render(html):
    if not pathlib.Path(CHROME).exists():
        sys.exit(f"Google Chrome not found at {CHROME}")
    with tempfile.TemporaryDirectory() as tmp:
        page = pathlib.Path(tmp) / "og-card.html"
        page.write_text(html)
        result = subprocess.run(
            [
                CHROME,
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                f"--screenshot={OUT}",
                f"--window-size={WIDTH},{HEIGHT}",
                f"file://{page}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode != 0:
        sys.exit(f"chrome failed ({result.returncode}):\n{result.stderr[-2000:]}")


def verify():
    """Assert the rendered file is what the meta tags and tests promise."""
    with open(OUT, "rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        sys.exit("rendered file is not a PNG")
    width, height = struct.unpack(">II", header[16:24])
    if (width, height) != (WIDTH, HEIGHT):
        sys.exit(f"expected {WIDTH}x{HEIGHT}, rendered {width}x{height}")
    kb = OUT.stat().st_size / 1024
    if kb >= 1024:
        sys.exit(f"card is {kb:.0f}KB; platforms cap around 1MB")
    print(f"og-card.png  {width}x{height}  {kb:.0f}KB  -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    render(build_html())
    verify()
