"""Build logs.html: every experiment write-up in docs/experiments/, in full, with its images.

    python3 tools/build_logs.py

Reads docs/experiments/README.md (index table) and exp*.md, copies the images
the pages use to assets/logs/, and writes logs.html at the repository root.
Standard library only.
"""

import html
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "docs" / "experiments"
IMG_OUT = ROOT / "assets" / "logs"


def inline(text):
    """Escape, then apply code, images, links, bold and italics."""
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append("<code>" + html.escape(part[1:-1]) + "</code>")
            continue
        s = html.escape(part, quote=False)
        s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: image(m.group(2), m.group(1), inline_img=True), s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: link(m.group(1), m.group(2)), s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", s)
        out.append(s)
    return "".join(out)


def link(text, href):
    m = re.match(r"exp(\d\d)_[\w]+\.md(#.*)?$", href)
    if m:
        return f'<a href="#exp-{m.group(1)}">{text}</a>'
    if href.startswith("http"):
        return f'<a href="{html.escape(href)}" rel="noopener">{text}</a>'
    return text  # repository files outside the site


USED = set()


def image(src, alt, inline_img=False):
    name = Path(src).name
    if (EXP / "img" / name).exists():
        USED.add(name)
        src = f"assets/logs/{name}"
    return f'<img src="{html.escape(src)}" alt="{html.escape(alt)}" loading="lazy">'


def table(rows):
    def cells(line):
        line = line.strip().strip("|")
        return [c.strip() for c in line.split("|")]
    head = cells(rows[0])
    body = [cells(r) for r in rows[2:]]
    t = ['<div class="wrap"><table>', "<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead>", "<tbody>"]
    for r in body:
        t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    t.append("</tbody></table></div>")
    return "\n".join(t)


ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")


def lists(lines):
    """Nested lists by indentation. Indented lines that are not items continue the last item."""
    root = {"kind": None, "items": [], "indent": -1}
    stack = [root]
    for line in lines:
        m = ITEM.match(line)
        if m:
            ind, mark, text = len(m.group(1)), m.group(2), m.group(3)
            kind = "ol" if mark[0].isdigit() else "ul"
            while len(stack) > 1 and ind < stack[-1]["indent"]:
                stack.pop()
            top = stack[-1]
            if ind > top["indent"] and top is not root and top["items"]:
                child = {"kind": kind, "items": [], "indent": ind}
                top["items"][-1]["sub"].append(child)
                stack.append(child)
                top = child
            elif top is root:
                child = {"kind": kind, "items": [], "indent": ind}
                root["items"].append(child)
                stack.append(child)
                top = child
            top["items"].append({"text": text, "sub": []})
        else:
            stack[-1]["items"][-1]["text"] += " " + line.strip()

    def render(node):
        out = [f"<{node['kind']}>"]
        for it in node["items"]:
            out.append("<li>" + inline(it["text"]) + "".join(render(c) for c in it["sub"]) + "</li>")
        out.append(f"</{node['kind']}>")
        return "".join(out)
    return "\n".join(render(n) for n in root["items"])


def markdown(text):
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        h = re.match(r"^(#{1,4})\s+(.*)$", line)
        if h:
            level = len(h.group(1)) + 1  # page title is h1, write-up title h2
            out.append(f"<h{level}>{inline(h.group(2))}</h{level}>")
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(table(rows))
            continue
        if line.startswith(">"):
            q = []
            while i < len(lines) and lines[i].startswith(">"):
                q.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(q)) + "</blockquote>")
            continue
        if ITEM.match(line):
            block = []
            while i < len(lines) and lines[i].strip() and (ITEM.match(lines[i]) or lines[i].startswith(" ")):
                block.append(lines[i])
                i += 1
            out.append(lists(block))
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line)
        if m:
            cap = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith("*") and not lines[j].startswith("**"):
                c = []
                while j < len(lines) and lines[j].strip():
                    c.append(lines[j].strip())
                    j += 1
                cap = "<figcaption>" + inline(" ".join(c).strip("*")) + "</figcaption>"
                i = j
            else:
                i += 1
            out.append("<figure>" + image(m.group(2), m.group(1)) + cap + "</figure>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||>|!\[)", lines[i]) and not ITEM.match(lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append("<p>" + inline(" ".join(para)) + "</p>")
    return "\n".join(out)


def index_rows():
    rows = []
    for line in (EXP / "README.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| \[EXP-(\d\d)\]\([^)]+\) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| (.+) \|$", line)
        if m:
            rows.append(m.groups())
    return rows


CSS = """
  :root {
    --page: #eef0f3; --sheet: #ffffff; --ink: #1a1d23; --ink-2: #3b4250; --muted: #6a7382;
    --faint: #9aa2ae; --rule: #dde1e7; --rule-strong: #1a1d23; --tint: #f5f6f8; --done: #2350c9;
    --sheet-pad: 48px; --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    color-scheme: light;
  }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body {
    margin: 0; background: var(--page); color: var(--ink);
    font-family: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont,
      "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", sans-serif;
    font-size: 15px; line-height: 1.7; overflow-wrap: anywhere;
  }
  a { color: var(--done); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .sheet { max-width: 920px; margin: 28px auto 56px; background: var(--sheet); border: 1px solid var(--rule); padding: var(--sheet-pad); }
  @media (max-width: 720px) { :root { --sheet-pad: 20px; } .sheet { margin: 0; border: 0; } }
  .topnav {
    position: sticky; top: 0; z-index: 20; display: flex; align-items: center; gap: 22px;
    margin: calc(var(--sheet-pad) * -1) calc(var(--sheet-pad) * -1) 28px; padding: 0 var(--sheet-pad);
    background: var(--sheet); border-bottom: 1px solid var(--rule); font-size: 14px;
    overflow-x: auto; white-space: nowrap;
  }
  .topnav .brand { font-weight: 800; letter-spacing: -0.02em; margin-right: auto; padding: 14px 0; }
  .topnav a { color: var(--muted); padding: 14px 0 12px; border-bottom: 2px solid transparent; }
  .topnav a:hover { color: var(--ink); text-decoration: none; }
  .topnav a.on { color: var(--ink); font-weight: 650; border-bottom-color: var(--ink); }
  .doc-head { border-bottom: 2px solid var(--rule-strong); padding-bottom: 16px; }
  .doc-title { font-size: 30px; line-height: 1.25; font-weight: 800; letter-spacing: -0.025em; margin: 0; }
  @media (max-width: 720px) { .doc-title { font-size: 25px; } }
  .jump { display: flex; flex-wrap: wrap; gap: 6px 16px; margin-top: 14px; font-size: 13px; font-family: var(--mono); }
  h2.sec { font-size: 20px; font-weight: 750; margin: 40px 0 14px; padding-bottom: 8px; border-bottom: 1px solid var(--rule); }
  article { margin-top: 56px; padding-top: 22px; border-top: 2px solid var(--rule-strong); scroll-margin-top: 56px; }
  article h2 { font-size: 21px; font-weight: 780; letter-spacing: -0.02em; margin: 0 0 6px; }
  article h3 { font-size: 16.5px; font-weight: 720; margin: 26px 0 8px; }
  article h4, article h5 { font-size: 14.5px; font-weight: 700; margin: 20px 0 8px; }
  .meta { font-size: 12.5px; color: var(--muted); font-family: var(--mono); margin: 0 0 14px; }
  p { margin: 0 0 12px; color: var(--ink-2); }
  ul, ol { margin: 0 0 12px; padding-left: 1.3em; color: var(--ink-2); }
  li { margin: 3px 0; }
  li > ul, li > ol { margin: 4px 0 4px; }
  code { font-family: var(--mono); font-size: 0.86em; background: var(--tint); padding: 1px 5px; border-radius: 3px; }
  blockquote { margin: 0 0 14px; padding: 8px 14px; border-left: 3px solid var(--rule); background: var(--tint); color: var(--ink-2); font-size: 14px; }
  .wrap { overflow-x: auto; margin: 0 0 14px; }
  table { border-collapse: collapse; font-size: 13.5px; min-width: 100%; }
  th, td { text-align: left; vertical-align: top; padding: 6px 10px; border-bottom: 1px solid var(--rule); }
  th { font-weight: 650; border-bottom: 1.5px solid var(--ink); white-space: nowrap; }
  td { color: var(--ink-2); font-variant-numeric: tabular-nums; }
  figure { margin: 16px 0 18px; }
  figure img, p img { display: block; max-width: 100%; height: auto; border: 1px solid var(--rule); }
  figcaption { font-size: 13px; color: var(--muted); margin-top: 6px; }
  .idx td:first-child { white-space: nowrap; font-family: var(--mono); font-size: 12.5px; }
  .idx td:nth-child(2), .idx td:nth-child(3) { white-space: nowrap; }
  footer { margin-top: 56px; padding-top: 16px; border-top: 1px solid var(--rule); font-size: 12.5px; color: var(--muted); }
"""


def main():
    pages = sorted(EXP.glob("exp[0-9][0-9]_*.md"))
    rows = index_rows()
    idx = ['<div class="wrap"><table class="idx"><thead><tr><th>ID</th><th>Date</th><th>Cost</th><th>Topic</th><th>Result in one line</th></tr></thead><tbody>']
    for n, date, topic, cost, result in rows:
        idx.append(f'<tr><td><a href="#exp-{n}">EXP-{n}</a></td><td>{date.strip()}</td><td>{inline(cost.strip())}</td>'
                   f'<td>{inline(topic.strip())}</td><td>{inline(result.strip())}</td></tr>')
    idx.append("</tbody></table></div>")

    arts = []
    for p in pages:
        n = p.name[3:5]
        body = markdown(p.read_text(encoding="utf-8"))
        # the line after the title (date, scripts, cost) reads as metadata
        body = re.sub(r"(</h2>)\n<p>(Date: .*?)</p>", r'\1\n<p class="meta">\2</p>', body, count=1)
        arts.append(f'<article id="exp-{n}">\n{body}\n</article>')

    jump = "".join(f'<a href="#exp-{p.name[3:5]}">EXP-{p.name[3:5]}</a>' for p in pages)
    page = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="robots" content="noindex, nofollow">
<meta name="color-scheme" content="light only">
<title>GraspGrade 실험 기록</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
<style>{CSS}</style>
</head>
<body>
<main class="sheet">
  <nav class="topnav" aria-label="페이지">
    <span class="brand">GraspGrade</span>
    <a href="index.html#summary">요약</a>
    <a href="index.html#story">핵심 실험</a>
    <a href="index.html#detail">실험 상세</a>
    <a href="demo.html">데모</a>
    <a href="logs.html" class="on">기록</a>
  </nav>
  <header class="doc-head">
    <h1 class="doc-title">실험 기록 전체</h1>
    <nav class="jump" aria-label="실험 바로가기">{jump}</nav>
  </header>
  <h2 class="sec">목록</h2>
  {"".join(idx)}
{chr(10).join(arts)}
  <footer>영상 출처: OpenGraph Ego-Tactile (CC-BY-4.0), HOT3D (Project Aria)</footer>
</main>
</body>
</html>
"""
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    for name in sorted(USED):
        src, dst = EXP / "img" / name, IMG_OUT / name
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dst)
    (ROOT / "logs.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"logs.html: {len(pages)} pages, {len(rows)} index rows, {len(USED)} images")


if __name__ == "__main__":
    main()
