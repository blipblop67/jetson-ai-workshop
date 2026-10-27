// AI Workflow Studio - front end
let CAT = null;                 // catalogue from the server
let editor = null;              // Drawflow instance
let selected = null;            // selected Drawflow node id
let running = false, lastAlert = 0, pollTimer = null, liveView = null, deployed = false, deployInfo = null;
const $ = (s) => document.querySelector(s);

// ------------------------------------------------------------------ helpers
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind; t.textContent = msg;
  $("#toasts").appendChild(t);
  while ($("#toasts").children.length > 3) $("#toasts").firstChild.remove();     // never cover the screen
  setTimeout(() => t.remove(), kind === "alert" ? 6000 : 3500);
}
function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    [0, 0.25].forEach((d) => {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = "square"; o.frequency.value = 880; g.gain.value = 0.08;
      o.connect(g); g.connect(ctx.destination); o.start(ctx.currentTime + d); o.stop(ctx.currentTime + d + 0.15);
    });
  } catch (e) {}
}
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r;
}
const def = (type) => CAT.blocks[type];
// Drawflow throws if a node id no longer exists (e.g. status from a previous project) - so look up safely
function nodeOf(id) { const d = editor.drawflow.drawflow.Home.data; return d[id] ? editor.getNodeFromId(id) : null; }
const color = (type) => CAT.colors[def(type).cat];
function defaults(type) { const o = {}; def(type).params.forEach((p) => (o[p.key] = JSON.parse(JSON.stringify(p.default)))); return o; }

function summary(type, p) {
  switch (type) {
    case "camera": return p.csi ? "CSI camera" : `USB camera ${p.source} · ${p.resolution}`;
    case "video": return p.file || "choose a video";
    case "detect": return `${(p.classes || []).length ? p.classes.join(", ") : "all objects"} · conf ${p.conf}${p.track ? " · IDs" : ""}`;
    case "classify": return p.model || "choose your model";
    case "keep": case "blur": return (p.classes || []).join(", ") || "all";
    case "zone": return `${p.name} · ${p.points.length} points`;
    case "line": return `${p.direction} at ${p.position}`;
    case "count_if": return `${(p.object || []).join("/") || "objects"} ${p.op} ${p.value}`;
    case "label_if": return `label = ${p.label || "?"}`;
    case "cooldown": return `once every ${p.seconds}s`;
    case "filter": return p.type;
    case "view": return p.title;
    case "alert": return p.message;
    case "snapshot": return `max 1 / ${p.every}s`;
    case "log": return p.mode === "every N seconds" ? `every ${p.seconds}s → ${p.file}` : `on trigger → ${p.file}`;
    case "webhook": return `${p.url} · ${p.mode || "on trigger"}`;
    case "telegram": return p.chat ? `chat ${p.chat}` : "set token + chat";
    case "ipcam": return p.url;
    case "pose": return `people + body points${p.track ? " · IDs" : ""}`;
    case "color": return (p.colors || []).join(", ") || "choose colours";
    case "motion": return `${p.sensitivity} sensitivity`;
    case "measure": return `OK ${p.min_mm}–${p.max_mm} mm · view ${p.view_mm} mm`;
    case "pose_if": return `${p.pose} ≥ ${p.value}`;
    case "dwell": return `longer than ${p.limit}s`;
    case "occupancy": return `capacity ${p.capacity} · ${p.when}`;
    case "speed": return `${p.metres} m · limit ${p.limit} km/h`;
    case "schedule": return `${p.start} → ${p.end}`;
    case "heatmap": return `fade ${p.fade}`;
    case "record": return `${p.before}s before + ${p.seconds}s after`;
    case "sheets": return p.url ? (p.mode === "every N seconds" ? `every ${p.every}s` : "on trigger") : "paste web app URL";
    case "mqtt": return `${p.topic} @ ${p.host}`;
  }
  return "";
}
function nodeHTML(type, id) {
  const d = def(type);
  return `<div class="nb-head" style="--c:${color(type)}">${d.icon} ${d.label}</div>
          <div class="nb-body"><div class="nb-sum" id="sum-${id}"></div><div class="nb-info" id="info-${id}"></div></div>`;
}
function refreshNode(id) {
  const n = editor.getNodeFromId(id); const el = document.getElementById("sum-" + id);
  if (el) el.textContent = summary(n.name, n.data);
}

// ------------------------------------------------------------------ build palette
function buildPalette(filter = "") {
  const list = $("#paletteList"); list.innerHTML = "";
  CAT.order.forEach((cat) => {
    const types = CAT.block_order.filter((t) => CAT.blocks[t].cat === cat &&
      (CAT.blocks[t].label + " " + CAT.blocks[t].help).toLowerCase().includes(filter.toLowerCase()));
    if (!types.length) return;
    list.insertAdjacentHTML("beforeend", `<div class="cat"><i style="background:${CAT.colors[cat]}"></i>${cat}</div>`);
    types.forEach((t) => {
      const b = CAT.blocks[t], el = document.createElement("div");
      el.className = "pblock"; el.draggable = true; el.style.setProperty("--c", CAT.colors[cat]);
      el.title = b.help; el.innerHTML = `<span class="ic">${b.icon}</span>${b.label}`;
      el.addEventListener("dragstart", (e) => e.dataTransfer.setData("type", t));
      el.addEventListener("dblclick", () => addBlock(t, 120 + Math.random() * 200, 100 + Math.random() * 200));
      list.appendChild(el);
    });
  });
}

// ------------------------------------------------------------------ canvas
function addBlock(type, x, y, params) {
  const d = def(type);
  const id = editor.addNode(type, d.inputs === 0 ? 0 : 1, 1, x, y, "", { ...defaults(type), ...(params || {}) }, "");
  const el = document.querySelector(`#node-${id} .drawflow_content_node`);
  el.innerHTML = nodeHTML(type, id);
  refreshNode(id); updateEmpty();
  return id;
}
function updateEmpty() {
  const n = Object.keys(editor.export().drawflow.Home.data).length;
  $("#emptyHint").style.display = n ? "none" : "flex";
}
function exportWF() {
  const data = editor.export().drawflow.Home.data, nodes = [], edges = [];
  Object.values(data).forEach((n) => {
    nodes.push({ id: "n" + n.id, type: n.name, params: n.data, x: Math.round(n.pos_x), y: Math.round(n.pos_y) });
    (n.outputs.output_1 ? n.outputs.output_1.connections : []).forEach((c) => edges.push(["n" + n.id, "n" + c.node]));
  });
  return { name: $("#projName").value || "My AI Project", nodes, edges };
}
function importWF(wf) {
  editor.clear(); clearProblems();
  $("#projName").value = wf.name || "My AI Project";
  const map = {};
  (wf.nodes || []).forEach((n) => { if (CAT.blocks[n.type]) map[n.id] = addBlock(n.type, n.x || 0, n.y || 0, n.params); });
  (wf.edges || []).forEach(([a, b]) => { if (map[a] && map[b]) editor.addConnection(map[a], map[b], "output_1", "input_1"); });
  selectNode(null); updateEmpty(); setTimeout(fitView, 50);
}
function fitView() {            // zoom/pan so every block is visible
  const nodes = Object.values(editor.export().drawflow.Home.data);
  if (!nodes.length) { editor.zoom_reset(); return; }
  const minX = Math.min(...nodes.map((n) => n.pos_x)), minY = Math.min(...nodes.map((n) => n.pos_y));
  const maxX = Math.max(...nodes.map((n) => n.pos_x)) + 220, maxY = Math.max(...nodes.map((n) => n.pos_y)) + 110;
  const box = $("#drawflow").getBoundingClientRect(), pad = 40;
  const z = Math.max(0.4, Math.min(1, (box.width - 2 * pad) / (maxX - minX), (box.height - 2 * pad) / (maxY - minY)));
  editor.zoom = z;
  // Drawflow scales around the canvas centre, so compensate when translating
  const cx = box.width / 2, cy = box.height / 2;
  editor.canvas_x = pad - minX * z - cx * (1 - z) + ((box.width - 2 * pad) - (maxX - minX) * z) / 2;
  editor.canvas_y = pad - minY * z - cy * (1 - z) + ((box.height - 2 * pad) - (maxY - minY) * z) / 2;
  editor.precanvas.style.transform = `translate(${editor.canvas_x}px, ${editor.canvas_y}px) scale(${z})`;
}

// ------------------------------------------------------------------ settings panel
function selectNode(id) {
  selected = id; const box = $("#settings");
  if (!id) { box.className = "empty"; box.innerHTML = "Click a block to change its settings."; return; }
  const n = editor.getNodeFromId(id), d = def(n.name);
  box.className = ""; box.innerHTML = `<div class="s-title">${d.icon} ${d.label}</div><div class="s-help">${d.help}</div>`;
  d.params.forEach((p) => box.appendChild(field(id, p, n.data[p.key])));
  const del = document.createElement("button"); del.className = "del"; del.textContent = "Delete block";
  del.onclick = () => { editor.removeNodeId("node-" + id); };
  box.appendChild(del);
  switchTab("settings");
}
function setParam(id, key, val) {
  const n = editor.getNodeFromId(id); n.data[key] = val;
  editor.updateNodeDataFromId(id, n.data); refreshNode(id);
}
function field(id, p, val) {
  const f = document.createElement("div"); f.className = "field";
  if (p.kind === "bool") {
    f.classList.add("check");
    f.innerHTML = `<label><input type="checkbox" ${val ? "checked" : ""}> ${p.label}</label>`;
    f.querySelector("input").onchange = (e) => setParam(id, p.key, e.target.checked);
    return f;
  }
  f.innerHTML = `<label>${p.label}</label>`;
  if (p.kind === "text") {
    const i = document.createElement("input"); i.value = val ?? "";
    i.oninput = () => setParam(id, p.key, i.value); f.appendChild(i);
  } else if (p.kind === "number") {
    const i = document.createElement("input"); i.type = "number"; i.value = val;
    ["min", "max", "step"].forEach((k) => p[k] !== undefined && (i[k] = p[k]));
    i.oninput = () => setParam(id, p.key, parseFloat(i.value)); f.appendChild(i);
  } else if (p.kind === "select") {
    const s = document.createElement("select");
    let opts = p.dynamic ? (CAT.models[p.dynamic] || []) : p.options;
    if (p.dynamic && val && !opts.includes(val)) opts = [val, ...opts];
    if (!opts.length) s.innerHTML = `<option value="">(none found)</option>`;
    opts.forEach((o) => s.insertAdjacentHTML("beforeend", `<option ${o === val ? "selected" : ""}>${o}</option>`));
    s.onchange = () => setParam(id, p.key, s.value); f.appendChild(s);
    if (p.dynamic && !val && opts.length) setParam(id, p.key, opts[0]);
  } else if (p.kind === "classes") {
    const list = [...(val || [])];
    const wrap = document.createElement("div");
    wrap.innerHTML = `<input list="names" placeholder="type e.g. person, then Enter"><div class="chips"></div>`;
    const inp = wrap.querySelector("input"), chips = wrap.querySelector(".chips");
    const draw = () => {
      chips.innerHTML = list.length ? "" : `<span style="color:var(--muted);font-size:12px">none</span>`;
      list.forEach((c, i) => {
        const ch = document.createElement("span"); ch.className = "chip"; ch.innerHTML = `${c}<button>✕</button>`;
        ch.querySelector("button").onclick = () => { list.splice(i, 1); setParam(id, p.key, [...list]); draw(); };
        chips.appendChild(ch);
      });
    };
    const add = () => {
      const v = inp.value.trim().toLowerCase();
      if (v && !list.includes(v)) {
        list.push(v); setParam(id, p.key, [...list]);
        if (!CAT.coco.includes(v) && !CAT.extra_names.includes(v))
          toast(`"${v}" is not one of the 80 standard objects. That's fine if your own custom model detects it.`);
      }
      inp.value = ""; draw();
    };
    inp.onkeydown = (e) => { if (e.key === "Enter") add(); };
    inp.onchange = add;
    draw(); f.appendChild(wrap);
  } else if (p.kind === "multi") {
    const list = [...(val || [])], wrap = document.createElement("div"); wrap.className = "chips";
    p.options.forEach((o) => {
      const b = document.createElement("button"); b.className = "pick" + (list.includes(o) ? " on" : "");
      b.innerHTML = `<i style="background:${o}"></i>${o}`;
      b.onclick = () => {
        const i = list.indexOf(o); i >= 0 ? list.splice(i, 1) : list.push(o);
        b.classList.toggle("on", i < 0); setParam(id, p.key, p.options.filter((x) => list.includes(x)));
      };
      wrap.appendChild(b);
    });
    f.appendChild(wrap);
  } else if (p.kind === "zone") {
    const box = document.createElement("div"); box.className = "zonebox";
    const pts = val || [];
    box.innerHTML = `<svg viewBox="0 0 160 90"><rect width="160" height="90" fill="#000"/>
      <polygon points="${pts.map(([x, y]) => `${x * 160},${y * 90}`).join(" ")}" fill="rgba(118,185,0,.3)" stroke="#76b900"/></svg>`;
    const b = document.createElement("button"); b.textContent = "✏️ Draw zone on the camera image"; b.style.marginTop = "6px";
    b.onclick = () => drawZone(id, p.key);
    f.appendChild(box); f.appendChild(b);
  }
  return f;
}

// ------------------------------------------------------------------ zone drawing
async function drawZone(id, key) {
  openModal("Draw the zone: click the corners, then Save", `<div class="empty">Getting a camera image…</div>`);
  const r = await api("/api/frame", exportWF());
  if (!r.ok) { const e = await r.json(); $("#mBody").innerHTML = `<div class="empty">${e.error}</div>`; return; }
  const url = URL.createObjectURL(await r.blob());
  const pts = [];
  $("#mBody").innerHTML = `<div class="zdraw"><img id="zimg" src="${url}"><svg id="zsvg" viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div>
    <div class="mbtns"><span class="grow">Click 3 or more points around the area. Objects whose feet are inside count as "in the zone".</span>
    <button id="zUndo">Undo</button><button id="zClear">Clear</button><button id="zSave" class="run">Save zone</button></div>`;
  const svg = $("#zsvg");
  const draw = () => {
    svg.innerHTML = `<polygon points="${pts.map((p) => p.join(",")).join(" ")}" fill="rgba(118,185,0,.3)" stroke="#76b900" stroke-width="0.004"/>` +
      pts.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="0.008" fill="#f2994a"/>`).join("");
  };
  svg.onclick = (e) => { const b = svg.getBoundingClientRect(); pts.push([+((e.clientX - b.left) / b.width).toFixed(3), +((e.clientY - b.top) / b.height).toFixed(3)]); draw(); };
  $("#zUndo").onclick = () => { pts.pop(); draw(); };
  $("#zClear").onclick = () => { pts.length = 0; draw(); };
  $("#zSave").onclick = () => {
    if (pts.length < 3) return toast("Click at least 3 points", "bad");
    setParam(id, key, pts); closeModal(); selectNode(id); toast("Zone saved");
  };
}

// ------------------------------------------------------------------ modal
function openModal(title, html) { $("#mTitle").textContent = title; $("#mBody").innerHTML = html; $("#modal").classList.remove("hidden"); }
function closeModal() { $("#modal").classList.add("hidden"); }
async function chooseProject(kind) {
  const d = await (await api("/api/projects")).json();
  const items = kind === "template" ? d.templates : d.projects;
  if (!items.length) return toast(kind === "template" ? "No templates found" : "No saved projects yet. Press Save first.", "bad");
  const card = (t) => `<div class="titem" data-f="${t.file}"><b>${t.name}</b><span>${t.description || ""}</span>
      <div class="tblocks">${(t.blocks || []).map((b) => CAT.blocks[b] ? CAT.blocks[b].icon : "").join(" ")}</div></div>`;
  let html;
  if (kind === "template") {          // grouped by task
    const groups = [...CAT.tasks, { id: "", label: "Other" }];
    html = groups.map((g) => {
      const list = items.filter((t) => (t.task || "") === g.id || (!g.id && !CAT.tasks.some((x) => x.id === t.task)));
      return list.length ? `<div class="tgroup">${g.label}</div><div class="tlist">${list.map(card).join("")}</div>` : "";
    }).join("");
  } else html = `<div class="tlist">${items.map(card).join("")}</div>`;
  openModal(kind === "template" ? "Start from a template" : "Open a saved project", html);
  document.querySelectorAll(".titem").forEach((el) => el.onclick = async () => {
    const wf = await (await api(`/api/project/${kind}/${el.dataset.f}`)).json();
    importWF(wf); closeModal(); toast(`Opened "${wf.name}"`);
  });
}

// ------------------------------------------------------------------ problems
function clearProblems() { $("#problems").innerHTML = ""; document.querySelectorAll(".has-error").forEach((e) => e.classList.remove("has-error")); }
function showProblems(list) {
  clearProblems();
  list.forEach((p) => {
    let name = "";
    if (p.node) {
      const id = p.node.slice(1), el = document.getElementById("node-" + id);
      const nn = nodeOf(id);
      if (el && nn) { el.classList.add("has-error"); name = def(nn.name).label + " "; }
    }
    $("#problems").insertAdjacentHTML("beforeend", `<div>⚠️ ${name}${p.msg}</div>`);
  });
}

// ------------------------------------------------------------------ run / status
async function check() {
  const d = await (await api("/api/validate", exportWF())).json();
  if (d.problems.length) showProblems(d.problems); else { clearProblems(); toast("✔ Workflow looks good"); }
}
async function runStop() {
  if (running) {
    if (deployed && !window.confirm("This stops the deployed project on the Jetson. Stop it?")) return;
    await api("/api/stop", {}); return poll();
  }
  const d = await (await api("/api/run", exportWF())).json();
  if (!d.ok) return showProblems(d.problems);
  clearProblems(); lastAlert = 0; liveView = null; toast("Starting… (loading AI models the first time takes a few seconds)");
  switchTab("live"); poll();
}
async function poll() {
  clearTimeout(pollTimer);
  let s;
  try { s = await (await api("/api/status?alert=" + lastAlert)).json(); } catch (e) { pollTimer = setTimeout(poll, 2000); return; }
  running = s.running;
  const btn = $("#btnRun"); btn.textContent = running ? "■ Stop" : "▶ Run"; btn.classList.toggle("stop", running);
  const pill = $("#statusPill");
  deployed = s.deployed && running;
  pill.className = "pill " + (s.error ? "error" : deployed ? "deployed" : running ? "running" : "idle");
  pill.textContent = s.error ? "Error" : deployed ? `Deployed · ${s.fps} FPS` : running ? `Running · ${s.fps} FPS` : "Stopped";
  updateDeployBar(s);
  $("#liveDot").className = running ? "on" : "";
  if (s.error && !pill.dataset.err) { toast(s.error, "bad"); pill.dataset.err = 1; }
  if (!s.error) delete pill.dataset.err;
  // block infos on the canvas
  Object.entries(s.blocks).forEach(([nid, info]) => { const el = document.getElementById("info-" + nid.slice(1)); if (el) el.textContent = info; });
  if (!running) document.querySelectorAll(".nb-info").forEach((e) => (e.textContent = ""));
  // views
  const sel = $("#viewSelect");
  if (s.views.length) {
    if ([...sel.options].map((o) => o.value).join() !== s.views.join()) {
      sel.innerHTML = s.views.map((v) => { const n = nodeOf(v.slice(1)); return `<option value="${v}">${n ? n.data.title : v}</option>`; }).join("");
    }
    sel.style.display = s.views.length > 1 ? "block" : "none";
    const want = s.views.includes(sel.value) ? sel.value : s.views[0];
    if (running && liveView !== want) { liveView = want; $("#liveImg").src = "/stream/" + want + "?t=" + Date.now(); }
  } else sel.style.display = "none";
  if (!running && liveView) { liveView = null; $("#liveImg").removeAttribute("src"); }
  $("#liveIdle").style.display = running && s.views.length ? "none" : "flex";
  // metrics
  const m = [["FPS", s.fps], ["Frames", s.frames], ["Running (s)", s.uptime], ...Object.entries(s.metrics)];
  $("#metrics").innerHTML = m.map(([l, n]) => `<div class="metric"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
  $("#blockStats").innerHTML = Object.entries(s.blocks).map(([nid, info]) => {
    const n = nodeOf(nid.slice(1)); return n ? `<div><span>${def(n.name).icon} ${def(n.name).label}</span><span>${info}</span></div>` : "";
  }).join("");
  // events
  $("#events").innerHTML = s.events.slice().reverse().map((e) => `<div class="ev ${e.kind.replace(/ /g, "_")}"><span>${e.t}</span><span class="k">${e.kind}</span><span>${e.detail}</span></div>`).join("") || `<div class="empty">No events yet.</div>`;
  // alerts
  s.alerts.forEach((a) => { lastAlert = Math.max(lastAlert, a.seq); toast("🚨 " + a.msg + (a.reason ? " - " + a.reason : ""), "alert"); if (a.sound) beep(); });
  pollTimer = setTimeout(poll, running ? 1000 : 2500);
}
async function loadSnaps() {
  const clips = await (await api("/api/clips")).json();
  $("#clips").innerHTML = clips.length ? "🎬 Video clips: " + clips.map((f) => `<a href="/clips/${f}">${f.replace("clip_", "").replace(".mp4", "")}</a>`).join("") : "";
  const files = await (await api("/api/snapshots")).json();
  $("#snaps").innerHTML = files.map((f) => `<a href="/snapshots/${f}" target="_blank"><img src="/snapshots/${f}" loading="lazy"><div>${f.replace("snap_", "").replace(".jpg", "")}</div></a>`).join("") || `<div class="empty" style="grid-column:1/-1">No snapshots yet. Add a 📸 Save Snapshot block after a condition.</div>`;
}
async function loadLogs() {
  const files = await (await api("/api/logs")).json();
  $("#logLinks").innerHTML = files.length ? "Download: " + files.map((f) => `<a href="/logs/${f}">${f}</a>`).join("") : "";
}
function switchTab(t) {
  document.querySelectorAll("#side nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.id === "tab-" + t));
  if (t === "snaps") loadSnaps();
  if (t === "events") loadLogs();
}

// ------------------------------------------------------------------ deploy
function qrSVG(text) {
  const q = qrcode(0, "M"); q.addData(text); q.make();
  return q.createSvgTag({ cellSize: 6, margin: 2, scalable: true });
}
async function deploy() {
  if (running && !deployed && !window.confirm("Stop the test run and deploy this project instead?")) return;
  if (deployed && !window.confirm("Replace the project that is deployed now?")) return;
  const d = await (await api("/api/deploy", exportWF())).json();
  if (!d.ok) return showProblems(d.problems);
  clearProblems(); deployInfo = d; lastAlert = 0; liveView = null;
  showDeploy(d, true); poll();
}
function showDeploy(d, fresh) {
  openModal(fresh ? "🚀 Deployed!" : "🚀 Deployed project", `<div class="dep">
    <div class="qr">${qrSVG(d.view_url)}</div>
    <div>
      <div>Open the live page on your <b>laptop or phone</b> (same Wi-Fi), or scan the QR code:</div>
      <div class="url">${d.view_url}</div>
      <ul>
        <li>It keeps running on the Jetson, even if you close this editor.</li>
        <li>The live page shows the video, the numbers, events and alerts.</li>
        <li>${d.boot ? "It starts again by itself after the Jetson restarts." : "It resumes whenever the studio starts. To also start after a reboot, the trainer runs <code>bash workflow_studio/install_service.sh</code> once."}</li>
      </ul>
      <div class="mbtns"><a href="/view" target="_blank"><button class="run">Open live page</button></a>
        <button id="depStop">■ Stop deployment</button><span class="grow"></span></div>
    </div></div>`);
  $("#depStop").onclick = async () => { await api("/api/stop", {}); closeModal(); toast("Deployment stopped"); poll(); };
}
function updateDeployBar(s) {
  const bar = $("#deployBar");
  if (!deployed) { bar.style.display = "none"; return; }
  const url = deployInfo ? deployInfo.view_url : location.origin + "/view";
  bar.style.display = "flex";
  bar.innerHTML = `🚀 <b>${s.name}</b> is deployed · <a href="/view" target="_blank">${url}</a> <button id="barQR">QR / details</button>`;
  $("#barQR").onclick = async () => showDeploy(deployInfo || await (await api("/api/info")).json(), false);
}

// ------------------------------------------------------------------ start up
async function init() {
  CAT = await (await api("/api/catalog")).json();
  document.body.insertAdjacentHTML("beforeend", `<datalist id="names">${[...CAT.coco, ...CAT.extra_names].map((c) => `<option value="${c}">`).join("")}</datalist>`);
  editor = new Drawflow($("#drawflow"));
  editor.reroute = false; editor.start();
  buildPalette();
  $("#search").oninput = (e) => buildPalette(e.target.value);

  const wrap = $("#drawflow");
  wrap.addEventListener("dragover", (e) => e.preventDefault());
  wrap.addEventListener("drop", (e) => {
    e.preventDefault(); const t = e.dataTransfer.getData("type"); if (!t) return;
    const pc = editor.precanvas, k = pc.clientWidth / (pc.clientWidth * editor.zoom);
    const x = e.clientX * k - pc.getBoundingClientRect().x * k, y = e.clientY * k - pc.getBoundingClientRect().y * k;
    const id = addBlock(t, x - 100, y - 30); editor.selectNode && selectNode(id);
  });
  editor.on("nodeSelected", (id) => selectNode(id));
  editor.on("nodeUnselected", () => selectNode(null));
  editor.on("nodeRemoved", () => { selectNode(null); updateEmpty(); });
  editor.on("connectionCreated", (c) => {     // ONE input per block: keep only the newest connection
    const node = editor.getNodeFromId(c.input_id);
    node.inputs.input_1.connections.filter((x) => x.node !== String(c.output_id)).forEach((x) =>
      editor.removeSingleConnection(x.node, c.input_id, x.input, "input_1"));
    clearProblems();
  });

  $("#btnNew").onclick = () => { if (confirmLoss()) { importWF({ name: "My AI Project", nodes: [], edges: [] }); } };
  $("#btnTemplates").onclick = () => chooseProject("template");
  $("#emptyTemplates").onclick = (e) => { e.preventDefault(); chooseProject("template"); };
  $("#btnOpen").onclick = () => chooseProject("project");
  $("#btnSave").onclick = async () => { const d = await (await api("/api/project", exportWF())).json(); toast(`Saved as ${d.file}`); };
  $("#btnCheck").onclick = check;
  $("#btnRun").onclick = runStop;
  $("#btnDeploy").onclick = deploy;
  $("#mClose").onclick = closeModal;
  $("#viewSelect").onchange = () => { liveView = null; poll(); };
  document.querySelectorAll("#side nav button").forEach((b) => (b.onclick = () => switchTab(b.dataset.tab)));
  setInterval(() => { if ($("#tab-snaps").classList.contains("active")) loadSnaps(); }, 5000);
  updateEmpty(); poll();
  // reopening the editor while a project is deployed: show that project on the canvas
  const s = await (await api("/api/status")).json();
  if (s.deployed) {
    const r = await api("/api/deployed");
    if (r.ok) { importWF(await r.json()); toast(`"${s.name}" is deployed and running`); }
    deployInfo = await (await api("/api/info")).json();
  }
}
function confirmLoss() { return !Object.keys(editor.export().drawflow.Home.data).length || window.confirm("Clear the canvas? Unsaved changes will be lost."); }
init();
