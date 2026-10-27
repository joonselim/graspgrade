// GraspGrade viewer: reads data/number2.json (clip scores) and data/number1.json (contact timing).
const state = { clips: [], presets: {}, preset: null, episodes: [] };

const gradeName = v => (v >= 1 ? "high" : v >= 0.5 ? "mid" : "low");

async function load(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(path + " " + r.status);
  return r.json();
}

function setTab(name) {
  document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.getElementById("tab-clips").hidden = name !== "clips";
  document.getElementById("tab-contact").hidden = name !== "contact";
  document.getElementById("presets").style.visibility = name === "clips" ? "visible" : "hidden";
}

function renderPresets() {
  const box = document.getElementById("presets");
  box.innerHTML = '<span class="hint">Preset</span>';
  const names = ["All", ...Object.keys(state.presets)];
  names.forEach(n => {
    const b = document.createElement("button");
    b.textContent = n === "All" ? "All" : `${n} (${state.presets[n]}+)`;
    b.classList.toggle("active", (state.preset || "All") === n);
    b.onclick = () => { state.preset = n === "All" ? null : n; renderPresets(); renderClipList(); };
    box.appendChild(b);
  });
}

function visibleClips() {
  const min = state.preset ? state.presets[state.preset] : -1;
  return state.clips.filter(c => c.score >= min).sort((a, b) => b.score - a.score);
}

function renderClipList() {
  const list = document.getElementById("clip-list");
  const clips = visibleClips();
  document.getElementById("count").textContent = `${clips.length} clips`;
  list.innerHTML = "";
  clips.forEach((c, i) => {
    const li = document.createElement("li");
    li.className = "item";
    li.innerHTML = `<span>${c.object.replace(/_/g, " ")}</span><span class="pill">${c.score.toFixed(0)}</span>` +
      `<span class="sub">${c.hand} hand, ${c.person}</span>`;
    li.onclick = () => showClip(c, li);
    list.appendChild(li);
    if (i === 0) showClip(c, li);
  });
}

function showClip(c, li) {
  document.querySelectorAll("#clip-list li").forEach(x => x.classList.remove("active"));
  li.classList.add("active");
  const v = document.getElementById("clip-video");
  v.src = c.video;
  v.play().catch(() => {});
  document.getElementById("clip-score").textContent = c.score.toFixed(0);
  document.getElementById("clip-truth").innerHTML =
    `Ground truth: <b style="color:var(${c.copyable_truth ? "--good" : "--bad"})">${c.copyable_truth ? "copyable" : "not copyable"}</b>`;
  const bar = document.getElementById("clip-contact");
  bar.style.left = `${(100 * c.contact[0]) / c.duration}%`;
  bar.style.width = `${(100 * (c.contact[1] - c.contact[0])) / c.duration}%`;
  const cards = document.getElementById("grade-cards");
  cards.innerHTML = "";
  c.grades.forEach(g => {
    const n = gradeName(g.value);
    const li2 = document.createElement("li");
    li2.innerHTML = `<span>${g.name}<br><span class="weight">weight ${g.weight >= 0 ? "+" : ""}${g.weight}</span></span>` +
      `<span class="grade ${n}">${n}</span>`;
    cards.appendChild(li2);
  });
  document.getElementById("clip-reason").textContent = c.muse_reason;
  trackCursor(v, document.getElementById("clip-cursor"), c.duration);
}

function trackCursor(video, cursor, duration) {
  video.ontimeupdate = () => { cursor.style.left = `${(100 * video.currentTime) / duration}%`; };
}

function renderEpisodes() {
  const list = document.getElementById("episode-list");
  list.innerHTML = "";
  state.episodes.forEach((e, i) => {
    const li = document.createElement("li");
    li.className = "item";
    li.innerHTML = `<span>Episode ${e.episode}</span><span class="sub">${Math.round(e.duration)} s</span>`;
    li.onclick = () => showEpisode(e, li);
    list.appendChild(li);
    if (i === 0) showEpisode(e, li);
  });
}

function showEpisode(e, li) {
  document.querySelectorAll("#episode-list li").forEach(x => x.classList.remove("active"));
  li.classList.add("active");
  const v = document.getElementById("ep-video");
  v.src = e.video;
  const tl = document.getElementById("ep-timeline");
  tl.innerHTML = "";
  ["left", "right"].forEach(hand => {
    [["glove", "truth", "glove (ground truth)"], ["arm_B", "ours", "arm B, SAM rule"]].forEach(([key, cls, label]) => {
      const t = document.createElement("div");
      t.className = "track";
      t.innerHTML = `<span class="label">${hand}: ${label}</span>`;
      (e.hands[hand][key] || []).forEach(([a, b]) => {
        const s = document.createElement("div");
        s.className = `seg ${cls}`;
        s.style.left = `${(100 * a) / e.duration}%`;
        s.style.width = `${(100 * (b - a)) / e.duration}%`;
        t.appendChild(s);
      });
      t.onclick = ev => {
        const r = t.getBoundingClientRect();
        v.currentTime = ((ev.clientX - r.left) / r.width) * e.duration;
      };
      tl.appendChild(t);
    });
  });
  const cur = document.createElement("div");
  cur.className = "cursor";
  tl.appendChild(cur);
  trackCursor(v, cur, e.duration);
}

async function main() {
  document.querySelectorAll(".tabs button").forEach(b => (b.onclick = () => setTab(b.dataset.tab)));
  try {
    const n2 = await load("data/number2.json");
    state.clips = n2.clips;
    state.presets = n2.presets;
    document.getElementById("clip-note").textContent = n2.note;
    renderPresets();
    renderClipList();
  } catch (err) {
    document.getElementById("clip-note").textContent = "Could not load data/number2.json: " + err.message;
  }
  try {
    const n1 = await load("data/number1.json");
    state.episodes = n1.episodes;
    document.getElementById("ep-note").textContent = n1.note + " Click a track to jump in the video.";
    renderEpisodes();
  } catch (err) {
    document.getElementById("ep-note").textContent = "Could not load data/number1.json: " + err.message;
  }
}

main();
