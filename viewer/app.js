// GraspGrade viewer: reads data/number2.json (clip scores) and data/number1.json (contact timing).
// Direct links for recording: #clip=ID, #episode=36&hand=right, #compare=ID1,ID2.
const state = { clips: [], presets: {}, preset: null, episodes: [], arms: {}, n1note: "", hand: null };

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
  document.getElementById("tab-compare").hidden = name !== "compare";
  document.getElementById("presets").style.visibility = name === "clips" ? "visible" : "hidden";
  document.getElementById("clips-summary").hidden = name !== "clips";
  document.getElementById("contact-summary").hidden = name !== "contact";
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
    li.dataset.id = c.id;
    li.innerHTML = `<span>${c.object.replace(/_/g, " ")}</span><span class="pill">${c.score.toFixed(0)}</span>` +
      `<span class="sub">${c.hand} hand, ${c.person}, ${c.copyable_truth ? "copyable" : "not copyable"}</span>`;
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
    `Ground truth: <b style="color:var(${c.copyable_truth ? "--good" : "--bad"})">${c.copyable_truth ? "copyable" : "not copyable"}</b>` +
    (c.failed_condition ? ` (fails: ${c.failed_condition})` : "");
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
  document.getElementById("clip-alone").textContent = c.muse_alone
    ? `${Math.round(100 * c.muse_alone.probability)}% copyable. ${c.muse_alone.reason}` : "";
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
    li.dataset.episode = e.episode;
    const f = k => e.scores[k].boundary_f1_15.toFixed(2);
    li.innerHTML = `<span>Episode ${e.episode}</span><span class="pill">${f("arm_A_shift")} / ${f("arm_C_shift")}</span>` +
      `<span class="sub">${Math.round(e.duration)} s</span>`;
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
  const tracks = [["glove", "truth", "glove (ground truth)"],
    ...Object.entries(state.arms).map(([key, label]) => [key, key, label])];
  (state.hand ? [state.hand] : ["left", "right"]).forEach(hand => {
    tracks.forEach(([key, cls, label]) => {
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
  document.getElementById("ep-note").textContent = `Episode ${e.episode}, F1 ±5 / F1 ±15 / IoU: ` +
    Object.entries(state.arms).map(([k, label]) => {
      const s = e.scores[k];
      return `${label.split(",")[0]} ${s.boundary_f1_5.toFixed(2)} / ${s.boundary_f1_15.toFixed(2)} / ${s.contact_iou.toFixed(2)}`;
    }).join("; ") + ". " + state.n1note + " Click a track to jump in the video.";
}

function showCompare(ids) {
  const box = document.getElementById("compare-grid");
  box.innerHTML = "";
  ids.map(id => state.clips.find(c => c.id === id)).filter(Boolean).forEach(c => {
    const card = document.createElement("section");
    card.className = "stage";
    card.innerHTML = `<video src="${c.video}" controls muted playsinline loop></video>` +
      `<div class="compare-head"><span>${c.object.replace(/_/g, " ")}, ${c.hand} hand, ${c.person}</span>` +
      `<span class="score">${c.score.toFixed(0)}</span></div>` +
      `<div class="truth">Ground truth: <b style="color:var(${c.copyable_truth ? "--good" : "--bad"})">` +
      `${c.copyable_truth ? "copyable" : "not copyable"}</b>${c.failed_condition ? ` (fails: ${c.failed_condition})` : ""}</div>` +
      `<ul class="compare-grades">${c.grades.map(g => `<li><span>${g.name}</span>` +
        `<span class="grade ${gradeName(g.value)}">${gradeName(g.value)}</span></li>`).join("")}</ul>`;
    box.appendChild(card);
  });
  box.querySelectorAll("video").forEach(v => v.play().catch(() => {}));
}

// Opens the view named in the hash. Without a hash the page starts on the first tab as before.
function openFromHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.has("clip")) {
    setTab("clips");
    if (state.preset) { state.preset = null; renderPresets(); renderClipList(); }
    const li = document.querySelector(`#clip-list li[data-id="${p.get("clip")}"]`);
    if (li) { li.click(); li.scrollIntoView({ block: "nearest" }); }
  } else if (p.has("episode")) {
    setTab("contact");
    state.hand = ["left", "right"].includes(p.get("hand")) ? p.get("hand") : null;
    const li = document.querySelector(`#episode-list li[data-episode="${p.get("episode")}"]`);
    if (li) { li.click(); li.scrollIntoView({ block: "nearest" }); }
  } else if (p.has("compare")) {
    setTab("compare");
    showCompare(p.get("compare").split(","));
  }
}

async function main() {
  document.querySelectorAll(".tabs button").forEach(b => (b.onclick = () => setTab(b.dataset.tab)));
  try {
    const n2 = await load("data/number2.json");
    state.clips = n2.clips;
    state.presets = n2.presets;
    document.getElementById("clip-note").textContent = n2.note;
    document.getElementById("clips-summary").textContent = n2.summary;
    renderPresets();
    renderClipList();
  } catch (err) {
    document.getElementById("clip-note").textContent = "Could not load data/number2.json: " + err.message;
  }
  try {
    const n1 = await load("data/number1.json");
    state.episodes = n1.episodes;
    state.arms = n1.arms;
    state.n1note = n1.note;
    document.getElementById("contact-summary").textContent = n1.summary;
    renderEpisodes();
  } catch (err) {
    document.getElementById("ep-note").textContent = "Could not load data/number1.json: " + err.message;
  }
  openFromHash();
  window.addEventListener("hashchange", openFromHash);
}

main();
