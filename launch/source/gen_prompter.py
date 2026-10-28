import json, sys
sys.path.insert(0, ".")
from script_data import SECTIONS

# ---- markdown script ----
md = ["# Demo video script: Yelp complaint tagger", "",
      "About 7 minutes at a relaxed pace. Lines starting with **[TYPE]** are typed into the demo live. *Cues* say what is on screen.",
      "Disclaimer: this is a learning project, not production software. The model is licensed CC BY-NC 4.0 (non-commercial).", ""]
for s in SECTIONS:
    md += [f"## {s['time']} · {s['title']}", f"*{s['cue']}*", ""]
    md += [f"- **[TYPE]** {l[7:]}" if l.startswith("[TYPE]") else f"- {l}" for l in s["lines"]] + [""]
open("demo-script.md", "w").write("\n".join(md))

html = r"""<title>Complaint Tagger Teleprompter</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=IBM+Plex+Mono:wght@500;600&display=swap">
<style>
/* Layout: full-height reading stage with a fixed reading line; control bar pinned to the bottom. Deliberately dark-only, like studio prompters. */
:root{--bg:#121417;--fg:#F2F1EC;--dim:#7D828C;--cue:#E0A22B;--type:#7FC8F8;--line:#C8323C;--bar:#1B1E23;--edge:#2C3038;
--read:'Atkinson Hyperlegible',Verdana,sans-serif;--mono:'IBM Plex Mono',Menlo,monospace;color-scheme:dark}
html,body{height:100%}
body{background:var(--bg);color:var(--fg);font-family:var(--read);overflow:hidden}
.app{height:100%;display:flex;flex-direction:column}
.stage{flex:1;overflow-y:auto;scroll-behavior:auto;position:relative;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.stage.mirror .script{transform:scaleX(-1)}
.script{max-width:1100px;margin:0 auto;padding-block:42vh 70vh}
.sec{padding-block:0 1.4em}
.sec h2{font-family:var(--mono);font-size:max(14px,.32em);font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--dim);margin:0 0 .4em}
.cue{font-family:var(--mono);font-size:max(14px,.34em);color:var(--cue);margin:0 0 .7em}
.ln{margin:0 0 .55em;line-height:1.35;text-wrap:pretty}
.ln.type{color:var(--type);font-family:var(--mono);font-size:.82em}
.ln.type::before{content:"TYPE  ";color:var(--dim)}
.hide-cues .cue,.hide-cues .sec h2{display:none}
.marker{position:fixed;left:0;right:0;top:40vh;height:0;border-top:2px solid var(--line);opacity:.55;pointer-events:none;z-index:2}
.marker::before{content:"";position:absolute;left:6px;top:-9px;border:8px solid transparent;border-left:12px solid var(--line)}
.count{position:fixed;inset:0;display:grid;place-items:center;font-family:var(--mono);font-size:22vh;color:var(--cue);background:rgba(18,20,23,.75);z-index:5}
.bar{background:var(--bar);border-top:1px solid var(--edge);display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;
padding-block:10px calc(10px + env(safe-area-inset-bottom,0px));padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));font-family:var(--mono);font-size:13px;transition:opacity .3s}
.playing .bar{opacity:.35}.playing .bar:hover,.playing .bar:focus-within{opacity:1}
button,select{font:inherit;color:var(--fg);background:#262A31;border:1px solid var(--edge);border-radius:6px;padding:7px 12px;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--cue);outline-offset:2px}
button.primary{background:var(--line);border-color:var(--line);font-weight:600;min-width:92px}
button[aria-pressed="true"]{border-color:var(--cue);color:var(--cue)}
label{display:flex;align-items:center;gap:8px;color:var(--dim)}
input[type=range]{width:110px;accent-color:var(--cue)}
.stat{color:var(--dim);font-variant-numeric:tabular-nums;margin-left:auto}
.keys{color:var(--dim);width:100%;font-size:12px}
@media (max-width:640px){.keys{display:none}.stat{margin-left:0}}
@media (prefers-reduced-motion:reduce){.bar{transition:none}}
</style>
<div class="app" id="app">
  <div class="stage" id="stage"><div class="script" id="script"></div></div>
  <div class="marker" aria-hidden="true"></div>
  <div class="bar" role="toolbar" aria-label="Teleprompter controls">
    <button class="primary" id="play">▶ Start</button>
    <button id="restart">⟲ Top</button>
    <select id="jump" aria-label="Jump to section"></select>
    <label for="speed">Speed <input type="range" id="speed" min="10" max="160" step="5" value="45"><span id="speedv">45</span></label>
    <label for="size">Size <input type="range" id="size" min="28" max="110" step="2" value="56"><span id="sizev">56</span></label>
    <button id="mirror" aria-pressed="false">Mirror</button>
    <button id="cues" aria-pressed="true">Cues</button>
    <span class="stat" id="stat">0:00 · 0%</span>
    <span class="keys">Space start/pause · ↑↓ speed · +/− size · M mirror · C cues · click the text to pause · scroll or drag to reposition</span>
  </div>
</div>
<div class="count" id="count" hidden></div>
<script>
const SECTIONS = __DATA__;
const $ = id => document.getElementById(id);
const stage = $("stage"), script = $("script"), app = $("app");
const store = { get(k, d) { try { const v = localStorage.getItem("tp-" + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
                set(k, v) { try { localStorage.setItem("tp-" + k, JSON.stringify(v)); } catch {} } };

SECTIONS.forEach(s => {
  const sec = document.createElement("section"); sec.className = "sec"; sec.id = "s-" + s.id;
  const h = document.createElement("h2"); h.textContent = s.time + " · " + s.title; sec.append(h);
  const c = document.createElement("p"); c.className = "cue"; c.textContent = "[" + s.cue + "]"; sec.append(c);
  s.lines.forEach(l => { const p = document.createElement("p"); const t = l.startsWith("[TYPE]");
    p.className = "ln" + (t ? " type" : ""); p.textContent = t ? l.slice(7) : l; sec.append(p); });
  script.append(sec);
  const o = document.createElement("option"); o.value = s.id; o.textContent = s.time + "  " + s.title; $("jump").append(o);
});

let playing = false, last = 0, pos = 0, elapsed = 0, wake = null;
const speedEl = $("speed"), sizeEl = $("size");
function applySize() { script.style.fontSize = sizeEl.value + "px"; $("sizev").textContent = sizeEl.value; store.set("size", +sizeEl.value); }
function applySpeed() { $("speedv").textContent = speedEl.value; store.set("speed", +speedEl.value); }
speedEl.value = store.get("speed", 45); sizeEl.value = store.get("size", 56); applySize(); applySpeed();
function setToggle(btn, on, cls, key) { btn.setAttribute("aria-pressed", on); cls(on); store.set(key, on); }
const mirrorOn = on => stage.classList.toggle("mirror", on), cuesOn = on => script.classList.toggle("hide-cues", !on);
setToggle($("mirror"), store.get("mirror", false), mirrorOn, "mirror");
setToggle($("cues"), store.get("cues", true), cuesOn, "cues");

function fmt(s) { s = Math.floor(s); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); }
function stat() { const max = stage.scrollHeight - stage.clientHeight; $("stat").textContent = fmt(elapsed) + " · " + Math.round(max > 0 ? 100 * stage.scrollTop / max : 0) + "%"; }
function frame(t) {
  if (!playing) return;
  const dt = last ? (t - last) / 1000 : 0; last = t; elapsed += dt;
  pos += speedEl.value * dt; stage.scrollTop = pos;
  if (stage.scrollTop + stage.clientHeight >= stage.scrollHeight - 1) pause();
  stat(); requestAnimationFrame(frame);
}
async function lockScreen() { try { wake = await navigator.wakeLock?.request("screen"); } catch { wake = null; } }
function start() {
  const c = $("count"); let n = 3; c.hidden = false; c.textContent = n;
  const iv = setInterval(() => { n--; if (n > 0) { c.textContent = n; return; } clearInterval(iv); c.hidden = true;
    playing = true; last = 0; pos = stage.scrollTop; app.classList.add("playing"); $("play").textContent = "❚❚ Pause";
    lockScreen(); requestAnimationFrame(frame); }, 700);
}
function pause() { playing = false; app.classList.remove("playing"); $("play").textContent = "▶ Resume"; try { wake?.release(); } catch {} stat(); }
$("play").onclick = () => playing ? pause() : start();
$("restart").onclick = () => { pause(); stage.scrollTop = 0; pos = 0; elapsed = 0; $("play").textContent = "▶ Start"; stat(); };
$("jump").onchange = e => { const el = $("s-" + e.target.value); stage.scrollTop = el.offsetTop - stage.clientHeight * 0.4 + 10; pos = stage.scrollTop; stat(); };
speedEl.oninput = applySpeed; sizeEl.oninput = applySize;
$("mirror").onclick = e => setToggle(e.currentTarget, e.currentTarget.getAttribute("aria-pressed") !== "true", mirrorOn, "mirror");
$("cues").onclick = e => setToggle(e.currentTarget, e.currentTarget.getAttribute("aria-pressed") !== "true", cuesOn, "cues");
script.onclick = () => { if (playing) pause(); };
stage.addEventListener("scroll", () => { if (!playing) { pos = stage.scrollTop; stat(); } });
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); playing ? pause() : start(); }
  else if (e.key === "ArrowUp") { speedEl.value = +speedEl.value + 5; applySpeed(); e.preventDefault(); }
  else if (e.key === "ArrowDown") { speedEl.value = +speedEl.value - 5; applySpeed(); e.preventDefault(); }
  else if (e.key === "+" || e.key === "=") { sizeEl.value = +sizeEl.value + 4; applySize(); }
  else if (e.key === "-") { sizeEl.value = +sizeEl.value - 4; applySize(); }
  else if (e.key.toLowerCase() === "m") $("mirror").click();
  else if (e.key.toLowerCase() === "c") $("cues").click();
});
document.addEventListener("visibilitychange", () => { if (document.hidden && playing) pause(); });
stat();
</script>
"""
open("teleprompter.html", "w").write(html.replace("__DATA__", json.dumps(SECTIONS)))
print("ok")
