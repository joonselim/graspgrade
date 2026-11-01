"""Build en.html, the English report page, from index.html.

The summary and key experiments pages come from tools/en/summary.html and
tools/en/story.html. The schedule data and the chart data come from
tools/en/phases.js and tools/en/charts.js. The detail page is replaced by a
link to logs.html, which already holds the English write-ups. Everything else,
the styles and the page code, is taken from index.html, so a layout change
there reaches the English page on the next build.

When index.html changes its words, change the matching English file in
tools/en/ and run this again:

    python3 -I tools/build_en.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EN = ROOT / "tools" / "en"


def between(s, start, end, new):
    a = s.index(start)
    b = s.index(end, a)
    return s[:a] + new + s[b:]


def rep(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:70])
    return s.replace(a, b)


def main():
    s = (ROOT / "index.html").read_text(encoding="utf-8")
    stamp = s[s.index("마지막 갱신 ") + len("마지막 갱신 "):]
    stamp = stamp[:stamp.index("<")].strip()

    # pages
    s = between(s, '  <div class="page" data-page="summary">', '  <div class="page" data-page="story" hidden>',
                (EN / "summary.html").read_text(encoding="utf-8"))
    s = between(s, '  <div class="page" data-page="story" hidden>', '  <footer>',
                (EN / "story.html").read_text(encoding="utf-8"))

    # head, top bar, footer
    s = rep(s, '<html lang="ko">', '<html lang="en">')
    s = rep(s, '<title>GraspGrade 진행</title>', '<title>GraspGrade progress</title>')
    s = between(s, '  <nav class="topnav"', '  <div class="page" data-page="summary">', '''  <nav class="topnav" aria-label="Pages">
    <span class="brand">GraspGrade</span>
    <a href="#summary" data-nav="summary">Summary</a>
    <a href="#story" data-nav="story">Key experiments</a>
    <a href="logs.html">Experiment log</a>
    <a href="demo.html">Demo</a>
    <a class="lang" href="index.html" hreflang="ko" lang="ko">KO</a>
  </nav>

''')
    s = between(s, '  <footer>', '</main>', f'''  <footer>
    Video sources: OpenGraph Ego-Tactile (CC-BY-4.0)<br>
    Last updated {stamp}
    <p class="demo-link"><a href="assets/video/graspgrade_demo.mp4">Watch the 3 minute video →</a></p>
    <p class="demo-link"><a href="logs.html">Full experiment log →</a></p>
  </footer>

''')

    # page code: data and words
    s = between(s, '  var PHASES = [', '  var START = ', (EN / "phases.js").read_text(encoding="utf-8"))
    s = between(s, '  var N1 = [', '  function vizTip() {', (EN / "charts.js").read_text(encoding="utf-8"))
    # the MediaPipe chart is on the detail page only
    s = between(s, '  var MP = [', '  function renderChart() {', '  var MP = [];\n\n')
    for a, b in [
        ('content: "표 " counter(tbl) ". ";', 'content: "Table " counter(tbl) ". ";'),
        ('content: "그림 " counter(fig) ". ";', 'content: "Figure " counter(fig) ". ";'),
        ('c.appendChild(el("small", null, k === 0 || parts[1] === "01" ? +parts[0] + "월" : "D" + (k + 1)));',
         'c.appendChild(el("small", null, k === 0 || parts[1] === "01" ? ({ 10: "Oct", 11: "Nov" })[+parts[0]] : "D" + (k + 1)));'),
        ('row.setAttribute("aria-label", (i + 1) + "단계 " + p.title);',
         'row.setAttribute("aria-label", "Phase " + (i + 1) + " " + p.title);'),
        ('top.appendChild(el("span", null, "단계 " + (i + 1)));', 'top.appendChild(el("span", null, "Phase " + (i + 1)));'),
        ('c.appendChild(el("div", "ph-h", "요구사항"));', 'c.appendChild(el("div", "ph-h", "Requirements"));'),
        ('c.appendChild(el("div", "ph-h", "체크리스트"));', 'c.appendChild(el("div", "ph-h", "Checklist"));'),
        ('document.title = { detail: "GraspGrade 실험 상세", story: "GraspGrade 핵심 실험" }[page] || "GraspGrade 진행";',
         'document.title = { detail: "GraspGrade experiment detail", story: "GraspGrade key experiments" }[page] || "GraspGrade progress";'),
        # experiment references open the English write-ups in logs.html
        ("document.querySelectorAll('td.first > a[href^=\"#exp-\"]')", "document.querySelectorAll('td.first > a[href^=\"#exp-\"]')"),
        ('a.href = "#" + m[0].toLowerCase();', 'a.href = "logs.html#" + m[0].toLowerCase();'),
        ('var go = el("a", "go", "보기 →");', 'var go = el("a", "go", "Open →");'),
        ('box.setAttribute("aria-label", "가까운 실험");', 'box.setAttribute("aria-label", "Nearby experiments");'),
        # charts
        ('<p class="viz-cap">평가 1: 영상만으로 찾은 접촉 시작과 끝 (검증 22개, 장갑 기준)</p>',
         '<p class="viz-cap">Test 1: contact start and end found from video alone (22 held-out, against the glove)</p>'),
        ('<p class="viz-sub">막대가 길수록 장갑과 잘 맞음. 1이 완벽.</p>',
         '<p class="viz-sub">Longer bars match the glove better. 1 is perfect.</p>'),
        ('<span><i style="--c:var(--v-hi)"></i>가장 높은 방법</span><span><i style="--c:var(--v-base)"></i>나머지 방법</span>',
         '<span><i style="--c:var(--v-hi)"></i>Best method</span><span><i style="--c:var(--v-base)"></i>Other methods</span>'),
        ('"±5프레임은 0.17초, ±15프레임은 0.5초 안에서 맞춘 경계의 비율(F1). IoU는 접촉이라고 본 프레임이 장갑과 겹치는 정도. 자세한 수치는 EXP-37."',
         '"±5 frames is 0.17 s and ±15 frames is 0.5 s, the window in which a boundary counts as matched (F1). IoU is how much the frames called contact overlap the glove. Full numbers in EXP-37."'),
        ('<p class="viz-cap">평가 2: 로봇 손이 따라할 수 있는 클립을 골라내는가 (검증, AUC)</p>',
         '<p class="viz-cap">Test 2: does it pick out clips a robot hand can copy (held-out, AUC)</p>'),
        ('<p class="viz-sub">점이 오른쪽일수록 잘 골라냄. 0.5는 동전 던지기, 가로선은 95% 구간.</p>',
         '<p class="viz-sub">Dots further right pick better. 0.5 is a coin flip, the lines are 95% intervals.</p>'),
        ('<span><i style="--c:var(--v-base)"></i>Muse Spark에게 한 번에 묻기</span><span><i style="--c:var(--v-hi)"></i>체크리스트로 쪼개 묻기</span><span><i style="--c:var(--v-ref)"></i>물체 이름 기준선</span>',
         '<span><i style="--c:var(--v-base)"></i>Ask Muse Spark in one go</span><span><i style="--c:var(--v-hi)"></i>Ask through the checklist</span><span><i style="--c:var(--v-ref)"></i>Object name baseline</span>'),
        ('"AUC는 따라할 수 있는 클립과 없는 클립을 하나씩 뽑았을 때 앞쪽에 더 높은 점수를 줄 확률. 자세한 수치는 EXP-40."',
         '"AUC is the chance that, for one copyable and one not copyable clip, the copyable one gets the higher score. Full numbers in EXP-40."'),
        ('<p class="viz-cap">질문 12개와 학습한 가중치 (점수 학습용 400개)</p>',
         '<p class="viz-cap">The 12 questions and their learned weights (400 score training clips)</p>'),
        ('<p class="viz-sub">가중치가 클수록 그 항목이 "상"일 때 점수가 많이 오름. 음수는 다른 항목을 함께 볼 때 그 항목이 "상"이면 점수를 내리는 쪽으로 학습된 것.</p>',
         '<p class="viz-sub">A larger weight raises the score more when the item is "high". A negative weight means that, with the other items in the model, a "high" on that item lowers the score.</p>'),
        ('<span><i style="--c:var(--v-pos)"></i>점수를 올림</span><span><i style="--c:var(--v-neg)"></i>점수를 내림</span>',
         '<span><i style="--c:var(--v-pos)"></i>Raises the score</span><span><i style="--c:var(--v-neg)"></i>Lowers the score</span>'),
        ('["출처", "항목, 무엇을 보나, \\"상\\"(로봇 손에 쉬움) 기준", "가중치"]',
         '["Source", "Item, what it looks at, rule for \\"high\\" (easy for a robot hand)", "Weight"]'),
        ('"</span><span class=\\"hi-rule\\">상: " + r.r', '"</span><span class=\\"hi-rule\\">High: " + r.r'),
        ('"<br>상: " + r.r + "<br>가중치 "', '"<br>High: " + r.r + "<br>Weight "'),
    ]:
        s = rep(s, a, b)

    # experiment ids link to the English write-ups
    s = s.replace('href="#exp-', 'href="logs.html#exp-')
    s = rep(s, "document.querySelectorAll('td.first > a[href^=\"#exp-\"]')", "document.querySelectorAll('td.first > a[href*=\"#exp-\"]')")
    s = rep(s, "document.querySelectorAll('a[href^=\"#exp-\"]')", "document.querySelectorAll('a[href*=\"#exp-\"]')")

    left = [line for line in s.splitlines() if any("가" <= ch <= "힣" for ch in line)]
    assert not left, "Korean text left in en.html:\n" + "\n".join(x.strip()[:120] for x in left[:20])
    (ROOT / "en.html").write_text(s, encoding="utf-8", newline="\n")
    print("en.html written")


if __name__ == "__main__":
    main()
