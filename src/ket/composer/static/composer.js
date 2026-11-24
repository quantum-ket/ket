// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

// Front end of ket.composer.Composer (an anywidget module).
// The Python side validates the circuit, builds the gate and runs KBW;
// this side only edits the `ops` list and shows what Python sends back.

const SVG_NS = "http://www.w3.org/2000/svg";
const COL_W = 56;
const ROW_H = 52;
const TOP = 26;
const MARG_W = 92;
const BOX = 40;
const MIN_COLS = 10;
const MAX_QUBITS = 32;

const DEFAULT_PARAMS = {
  theta: "pi/2",
  phi: "0",
  lambda: "pi",
};
const PARAM_SYMBOL = { theta: "θ", phi: "φ", lambda: "λ" };
// Label of a gate angle in the editor: "angle", or "angle θ" when the gate
// has several (U3).
const angleLabel = (info, p) => (info.params.length > 1 ? `angle ${PARAM_SYMBOL[p] ?? p}` : "angle");

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(child));
  }
  return node;
}

function s(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(child));
  }
  return node;
}

// Show angles the way qulib.draw does: pi -> π, theta -> θ... Every Greek
// letter name works, with an index as a subscript: theta1, theta_1 -> θ₁.
// lambda_ (the parameter Python reads for the keyword lambda) shows as λ.
const GREEK = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", zeta: "ζ",
  eta: "η", theta: "θ", iota: "ι", kappa: "κ", lambda: "λ", mu: "μ",
  nu: "ν", xi: "ξ", omicron: "ο", pi: "π", rho: "ρ", sigma: "σ", tau: "τ",
  upsilon: "υ", phi: "φ", chi: "χ", psi: "ψ", omega: "ω",
  Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π",
  Sigma: "Σ", Upsilon: "Υ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
};
const GREEK_NAME = new RegExp(`(?<![\\w])(${Object.keys(GREEK).join("|")})(?:_?(\\d+))?_?(?![\\w])`, "g");
const SUBSCRIPT = "₀₁₂₃₄₅₆₇₈₉";
function pretty(source) {
  return source
    .replace(GREEK_NAME, (_, name, index) => GREEK[name] + (index ? [...index].map((d) => SUBSCRIPT[d]).join("") : ""))
    .replace(/\s*\*\s*/g, "")
    .replace(/\s+/g, "");
}

// Gate colors come from qulib.DRAW_STYLE, through the kc-gate-<name> classes.
const gateClass = (name) => `kc-gate-${name}`;

// Replace the children of a node, skipping null, undefined and false, which
// replaceChildren would otherwise show as text ("null").
function put(node, ...children) {
  node.replaceChildren(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
}

function clone(x) {
  return JSON.parse(JSON.stringify(x));
}

function fmt(x, digits = 3) {
  const v = Math.abs(x) < 0.5 * 10 ** -digits ? 0 : x;
  return v.toFixed(digits);
}

// Signed, as +1.000 and −1.000; zero has no sign.
function fmtSigned(x, digits = 3) {
  const text = fmt(x, digits);
  if (Number(text) > 0) return `+${text}`;
  if (Number(text) < 0) return `−${text.slice(1)}`;
  return text;
}

function phaseColor(ph) {
  // Phase 0 is blue; the hue turns with the phase.
  const hue = ((((ph * 180) / Math.PI + 215) % 360) + 360) % 360;
  return `hsl(${hue}, 70%, 50%)`;
}

// Energy of a basis state, as a fraction of the range [lo, hi] of the
// state: lowest is blue (cold), highest is red (hot).
function energyColor(energy, lo, hi) {
  const t = hi - lo > 1e-12 ? (energy - lo) / (hi - lo) : 0;
  return `hsl(${230 - 225 * t}, 70%, 50%)`;
}

// Copied gates: {ops (columns from 0), minQ, maxQ, width, params}.
// anywidget imports this module once per widget, so a module variable is not
// shared: the clipboard lives on globalThis, for every composer of the page,
// and in localStorage, for composers in other notebooks or tabs. Writing it
// fires CLIPBOARD_EVENT, so that the other composers show their Paste button.
const CLIPBOARD_KEY = "ket-composer-clipboard";
const CLIPBOARD_EVENT = "ket-composer-clipboard";

function getClipboard() {
  try {
    const raw = globalThis.localStorage?.getItem(CLIPBOARD_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    // No storage (private window, blocked site data) or bad data.
  }
  return globalThis.__ketComposerClipboard || null;
}

function setClipboard(data) {
  globalThis.__ketComposerClipboard = data;
  try {
    globalThis.localStorage?.setItem(CLIPBOARD_KEY, JSON.stringify(data));
  } catch {
    // The page-wide copy still works.
  }
  globalThis.dispatchEvent?.(new CustomEvent(CLIPBOARD_EVENT));
}

function render({ model, el }) {
  const state = {
    selection: new Set(), // keyOf() of the selected ops
    drag: null,
    history: [],
    future: [],
    basisFormat: "bits", // How the state shows each register: bits, unsigned or signed
  };

  const catalog = () => {
    const out = {};
    for (const g of model.get("catalog")) out[g.name] = g;
    return out;
  };
  const ops = () => model.get("ops");
  const nq = () => model.get("num_qubits");

  // Qubit registers, as the parameters of the composed gate: a circuit qubit
  // is [register name, index in the register].
  function qubitName(qubit) {
    let start = 0;
    for (const [name, size] of model.get("registers") || []) {
      if (qubit < start + size) return [name, qubit - start];
      start += size;
    }
    return ["q", qubit];
  }
  const qubitText = (qubit) => {
    const [name, index] = qubitName(qubit);
    return `${name}[${index}]`;
  };
  // Width of the wire labels: grows with the register names.
  let labelW = 52;

  // ---------- model updates ----------
  function commit(newOps, extra = {}) {
    state.history.push({ ops: clone(ops()), num_qubits: nq(), registers: clone(model.get("registers")) });
    if (state.history.length > 200) state.history.shift();
    state.future = [];
    model.set("ops", newOps);
    for (const [k, v] of Object.entries(extra)) model.set(k, v);
    model.save_changes();
  }

  function restore(from, to) {
    const snap = from.pop();
    if (!snap) return;
    to.push({ ops: clone(ops()), num_qubits: nq(), registers: clone(model.get("registers")) });
    model.set("registers", snap.registers);
    model.set("num_qubits", snap.num_qubits);
    model.set("ops", snap.ops);
    model.save_changes();
  }
  const undo = () => restore(state.history, state.future);
  const redo = () => restore(state.future, state.history);

  // ---------- selection ----------
  // An op is identified by its column and first qubit (unique in a column).
  const keyOf = (op) => `${op.col}:${op.qubits[0]}`;
  const selectedIndices = () =>
    ops()
      .map((op, i) => (state.selection.has(keyOf(op)) ? i : -1))
      .filter((i) => i >= 0);
  // Indices plus every op that shares a group with one of them.
  function withGroups(indices) {
    const all = ops();
    const groups = new Set(indices.map((i) => all[i].group).filter(Boolean));
    const out = new Set(indices);
    all.forEach((op, i) => {
      if (op.group && groups.has(op.group)) out.add(i);
    });
    return [...out];
  }
  function select(indices, add = false) {
    const all = ops();
    if (!add) state.selection = new Set();
    for (const i of indices) state.selection.add(keyOf(all[i]));
  }
  function selectOps(list) {
    state.selection = new Set(list.map(keyOf));
  }
  function toggle(indices) {
    const all = ops();
    const keys = indices.map((i) => keyOf(all[i]));
    const on = keys.every((k) => state.selection.has(k));
    for (const k of keys) on ? state.selection.delete(k) : state.selection.add(k);
  }
  function newGroupId(list) {
    let max = 0;
    for (const op of list) {
      const m = /^g(\d+)$/.exec(op.group || "");
      if (m) max = Math.max(max, Number(m[1]));
    }
    return `g${max + 1}`;
  }

  function numCols() {
    let max = -1;
    for (const op of ops()) max = Math.max(max, op.col);
    return Math.max(MIN_COLS, max + 2);
  }

  // Qubits a gate would take when dropped at `row` (null if it does not fit).
  function targetQubits(gateName, row, moving) {
    const info = catalog()[gateName];
    const n = nq();
    if (moving) {
      const offset = row - moving.qubits[0];
      let qs = moving.qubits.map((q) => q + offset);
      const lo = Math.min(...qs);
      const hi = Math.max(...qs);
      if (lo < 0) qs = qs.map((q) => q - lo);
      if (hi >= n) qs = qs.map((q) => q - (hi - n + 1));
      return qs.every((q) => q >= 0 && q < n) ? qs : null;
    }
    if (info.num_qubits > n) return null;
    const qs = [];
    let start = Math.min(row, n - info.num_qubits);
    for (let i = 0; i < info.num_qubits; i++) qs.push(start + i);
    return qs;
  }

  // A new gate from the palette; it replaces what it overlaps.
  function place(gateName, col, row) {
    const qubits = targetQubits(gateName, row, null);
    if (!qubits) return;
    const info = catalog()[gateName];
    const op = {
      gate: gateName,
      col,
      qubits,
      params: info.params.map((p) => DEFAULT_PARAMS[p] ?? "0"),
    };
    const kept = ops().filter((o) => !(o.col === col && o.qubits.some((q) => qubits.includes(q))));
    kept.push(op);
    selectOps([op]);
    commit(kept);
  }

  // Shift of a moved selection, kept inside the circuit: [dCol, dQubit].
  function clampShift(moving, dc, dq) {
    const qs = moving.flatMap((op) => op.qubits);
    const minQ = Math.min(...qs);
    const maxQ = Math.max(...qs);
    const minCol = Math.min(...moving.map((op) => op.col));
    dq = Math.max(-minQ, Math.min(nq() - 1 - maxQ, dq));
    dc = Math.max(-minCol, dc);
    return [dc, dq];
  }

  // Move ops together; like a dropped gate, they replace what they overlap.
  function moveOps(indices, dc, dq) {
    const all = ops();
    const moving = indices.map((i) => all[i]);
    [dc, dq] = clampShift(moving, dc, dq);
    if (dc === 0 && dq === 0) return;
    const moved = moving.map((op) => ({ ...clone(op), col: op.col + dc, qubits: op.qubits.map((q) => q + dq) }));
    const taken = new Set(moved.flatMap((op) => op.qubits.map((q) => `${op.col}:${q}`)));
    const keep = new Set(indices);
    const kept = all.filter((op, i) => !keep.has(i) && !op.qubits.some((q) => taken.has(`${op.col}:${q}`)));
    selectOps(moved);
    commit([...kept, ...moved]);
  }

  function removeOps(indices) {
    if (!indices.length) return;
    const drop = new Set(indices);
    state.selection = new Set();
    commit(ops().filter((_, i) => !drop.has(i)));
  }

  function groupSelection() {
    const indices = selectedIndices();
    if (indices.length < 2) return;
    const id = newGroupId(ops());
    const next = clone(ops());
    for (const i of indices) next[i].group = id;
    commit(next);
  }

  function ungroupSelection() {
    const indices = selectedIndices();
    if (!indices.some((i) => ops()[i].group)) return;
    const next = clone(ops());
    for (const i of indices) delete next[i].group;
    commit(next);
  }

  function copySelection() {
    const all = ops();
    const chosen = selectedIndices().map((i) => all[i]);
    if (!chosen.length) return false;
    const minCol = Math.min(...chosen.map((op) => op.col));
    const qs = chosen.flatMap((op) => op.qubits);
    // Slider values of the free parameters, for a composer that lacks them.
    const params = model.get("params") || {};
    const names = new Set((model.get("param_names") || []).filter((name) => name in params));
    setClipboard({
      ops: chosen.map((op) => ({ ...clone(op), col: op.col - minCol })),
      minQ: Math.min(...qs),
      maxQ: Math.max(...qs),
      width: Math.max(...chosen.map((op) => op.col)) - minCol + 1,
      params: Object.fromEntries([...names].map((name) => [name, params[name]])),
    });
    return true;
  }

  // Paste after the selection (or after the last used column), on the same
  // qubits when they fit, shifting the later columns to make room. The
  // pasted gates are selected, ready to be dragged into place.
  function paste() {
    const CLIPBOARD = getClipboard();
    if (!CLIPBOARD) return;
    const n = nq();
    const span = CLIPBOARD.maxQ - CLIPBOARD.minQ;
    if (span >= n) return;
    const all = ops();
    const sel = selectedIndices();
    const cols = (sel.length ? sel.map((i) => all[i]) : all).map((op) => op.col);
    const at = cols.length ? Math.max(...cols) + 1 : 0;
    const base = Math.min(CLIPBOARD.minQ, n - 1 - span);
    const next = clone(all).map((op) => (op.col >= at ? { ...op, col: op.col + CLIPBOARD.width } : op));
    // Copied groups become new groups.
    const ids = new Map();
    const pasted = CLIPBOARD.ops.map((op) => {
      const copy = { ...clone(op), col: op.col + at, qubits: op.qubits.map((q) => q - CLIPBOARD.minQ + base) };
      if (op.group) {
        if (!ids.has(op.group)) ids.set(op.group, newGroupId([...next, ...[...ids.values()].map((g) => ({ group: g }))]));
        copy.group = ids.get(op.group);
      }
      return copy;
    });
    selectOps(pasted);
    // Keep the slider values of the source composer, but not over ours.
    const params = model.get("params") || {};
    const missing = Object.entries(CLIPBOARD.params || {}).filter(([name]) => !(name in params));
    commit([...next, ...pasted], missing.length ? { params: { ...params, ...Object.fromEntries(missing) } } : {});
  }

  function updateOp(index, changes) {
    const next = clone(ops());
    Object.assign(next[index], changes);
    commit(next);
  }

  function setQubits(n) {
    n = Math.max(1, Math.min(MAX_QUBITS, n));
    if (n === nq()) return;
    const next = ops().filter((op) => op.qubits.every((q) => q < n));
    commit(next, { num_qubits: n });
  }

  // Insert an empty column before `col`, shifting the later columns right.
  function insertColumn(col) {
    const next = clone(ops()).map((op) =>
      op.col >= col ? { ...op, col: op.col + 1 } : op,
    );
    selectOps(selectedIndices().map((i) => next[i]));
    const simCol = model.get("sim_column");
    commit(next, simCol >= col ? { sim_column: simCol + 1 } : {});
  }

  function compact() {
    const cols = [...new Set(ops().map((op) => op.col))].sort((a, b) => a - b);
    const map = new Map(cols.map((c, i) => [c, i]));
    const next = clone(ops()).map((op) => ({ ...op, col: map.get(op.col) }));
    selectOps(selectedIndices().map((i) => next[i]));
    commit(next);
  }

  // ---------- layout ----------
  const root = h("div", { class: "kc-root", tabindex: "0" });
  el.append(root);

  const toolbar = h("div", { class: "kc-toolbar" });
  const palette = h("div", { class: "kc-palette" });
  const canvasWrap = h("div", { class: "kc-canvas" });
  const errorBox = h("div", { class: "kc-error" });
  const side = h("div", { class: "kc-side" });
  const inspector = h("div", { class: "kc-panel kc-inspector" });
  const paramsPanel = h("div", { class: "kc-panel kc-params" });
  const statePanel = h("div", { class: "kc-panel kc-state" });
  const blochPanel = h("div", { class: "kc-panel kc-bloch" });

  // Circuit and gate editing on top; below, the simulation: the state, with
  // a Bloch sphere per qubit on its right.
  root.append(
    toolbar,
    palette,
    h("div", { class: "kc-main" }, h("div", { class: "kc-left" }, canvasWrap, errorBox), side),
    h("div", { class: "kc-results" }, statePanel, blochPanel),
  );
  side.append(inspector, paramsPanel);

  // ---------- toolbar ----------
  function renderToolbar() {
    put(toolbar, 
      h("span", { class: "kc-title" }, "Ket Composer"),
      h(
        "span",
        { class: "kc-group" },
        "Qubits",
        h("button", { class: "kc-btn kc-small", title: "Remove the last qubit", onclick: () => setQubits(nq() - 1) }, "−"),
        h("span", { class: "kc-count" }, String(nq())),
        h("button", { class: "kc-btn kc-small", title: "Add a qubit", onclick: () => setQubits(nq() + 1) }, "+"),
      ),
      h(
        "label",
        { class: "kc-group", title: "Run the circuit on KBW after every edit" },
        h("input", {
          type: "checkbox",
          checked: model.get("simulate") ? "checked" : null,
          onchange: (e) => {
            model.set("simulate", e.target.checked);
            model.save_changes();
          },
        }),
        "Simulate",
      ),
      h("span", { class: "kc-spacer" }),
      h("button", { class: "kc-btn", title: "Undo (Ctrl+Z)", disabled: state.history.length ? null : "disabled", onclick: undo }, "↶ Undo"),
      h("button", { class: "kc-btn", title: "Redo (Ctrl+Y)", disabled: state.future.length ? null : "disabled", onclick: redo }, "↷ Redo"),
      h("button", { class: "kc-btn", title: "Copy the circuit as Ket code", onclick: copyCode }, "Copy code"),
      h("button", { class: "kc-btn", title: "Remove empty columns", onclick: compact }, "Compact"),
      h(
        "button",
        {
          class: "kc-btn kc-danger",
          title: "Remove every gate",
          onclick: () => {
            state.selection = new Set();
            commit([], { sim_column: -1 });
          },
        },
        "Clear",
      ),
      fileGroup(),
    );
  }

  // The kernel reads and writes the file (relative to its working
  // directory), so it works the same in every notebook front end.
  function fileGroup() {
    const input = h("input", {
      class: "kc-input kc-file",
      value: model.get("file"),
      spellcheck: "false",
      title: "JSON file, relative to the kernel's working directory",
      // Kept in the local model only, so a re-render keeps what was typed.
      oninput: (e) => model.set("file", e.target.value),
      onkeydown: (e) => {
        if (e.key === "Enter") send("save");
      },
    });
    const send = (action) => {
      const path = input.value.trim();
      if (!path) return;
      model.send({ action, path });
    };
    const status = model.get("status");
    return h(
      "span",
      { class: "kc-group" },
      input,
      h("button", { class: "kc-btn", title: "Save the circuit to the file", onclick: () => send("save") }, "Save"),
      h("button", { class: "kc-btn", title: "Load the circuit from the file", onclick: () => send("load") }, "Load"),
      status ? h("span", { class: `kc-status${status.startsWith("Could not") ? " kc-status-error" : ""}`, title: status }, status) : null,
    );
  }

  // ---------- palette ----------
  function renderPalette() {
    const groups = {};
    for (const g of model.get("catalog")) (groups[g.group] ??= []).push(g);
    put(palette, 
      ...Object.entries(groups).map(([group, gates]) =>
        h(
          "div",
          { class: "kc-pgroup" },
          h("div", { class: "kc-pgroup-title" }, group),
          h(
            "div",
            { class: "kc-pgroup-items" },
            gates.map((g) =>
              h(
                "div",
                {
                  class: `kc-tile ${gateClass(g.name)}`,
                  title: `${g.description}\nDrag onto a wire`,
                  onpointerdown: (e) => startDrag(e, g.name, null),
                },
                g.name === "SWAP" ? "SWAP" : g.label,
              ),
            ),
          ),
        ),
      ),
    );
  }

  // ---------- circuit ----------
  let svg = null;
  let highlight = null;
  let marquee = null;

  function gateBox(op, info, qubit, x, y, selected) {
    const cls = `kc-box${selected ? " kc-selected" : ""}`;
    const g = s("g", { class: "kc-op" });
    const hasParams = op.params && op.params.length;
    const w = BOX + (hasParams ? 8 : 0);
    g.append(
      s("rect", { class: cls, x: x - w / 2, y: y - BOX / 2, width: w, height: BOX }),
      s("text", { class: "kc-box-label", x, y: hasParams ? y - 5 : y + 1 }, info.label),
    );
    if (hasParams) {
      let txt = op.params.map(pretty).join(",");
      if (txt.length > 8) txt = txt.slice(0, 7) + "…";
      g.append(s("text", { class: "kc-box-param", x, y: y + 10 }, txt));
    }
    return g;
  }

  function renderCircuit() {
    const n = nq();
    const cols = numCols();
    const simulate = model.get("simulate");
    const result = model.get("sim_result") || {};
    const simCol = model.get("sim_column");
    labelW = Math.max(
      52,
      ...(model.get("registers") || []).map(([name, size]) => 22 + 10.5 * name.length + 7.5 * String(size - 1).length),
    );
    const width = labelW + cols * COL_W + MARG_W;
    const height = TOP + n * ROW_H + 6;
    const cat = catalog();
    // Forget selected ops that no longer exist (undo, load...).
    const existing = new Set(ops().map(keyOf));
    for (const k of [...state.selection]) if (!existing.has(k)) state.selection.delete(k);
    const selSet = new Set(selectedIndices());

    svg = s("svg", { class: "kc-svg", width, height, viewBox: `0 0 ${width} ${height}` });

    // Column headers.
    for (let c = 0; c < cols; c++) {
      const x = labelW + c * COL_W;
      const active = simCol === c;
      const head = s(
        "g",
        { class: `kc-colhead${active ? " kc-active" : ""}` },
        s("rect", { x: x + 4, y: 2, width: COL_W - 8, height: TOP - 6, rx: 4 }),
        s("text", { x: x + COL_W / 2, y: TOP / 2 + 1 }, String(c)),
        s("title", {}, active ? "Show the final state" : `Show the state after column ${c}`),
      );
      head.addEventListener("click", () => {
        model.set("sim_column", active ? -1 : c);
        model.save_changes();
      });
      svg.append(head);
    }

    // "+" handles on the column boundaries, up to the last used column:
    // insert an empty column there. A guide line shows where on hover.
    let lastCol = -1;
    for (const op of ops()) lastCol = Math.max(lastCol, op.col);
    const handles = s("g", { class: "kc-inserts" });
    for (let c = 0; c <= lastCol; c++) {
      const x = labelW + c * COL_W;
      const cy = TOP / 2;
      const handle = s(
        "g",
        { class: "kc-insert" },
        s("line", { class: "kc-insert-guide", x1: x, y1: TOP, x2: x, y2: TOP + n * ROW_H }),
        s("circle", { cx: x, cy, r: 7 }),
        s("path", { d: `M${x - 3.5},${cy}H${x + 3.5}M${x},${cy - 3.5}V${cy + 3.5}` }),
        s("title", {}, c === 0 ? "Insert a column at the beginning" : `Insert a column between ${c - 1} and ${c}`),
      );
      handle.addEventListener("click", (e) => {
        e.stopPropagation();
        insertColumn(c);
      });
      handles.append(handle);
    }

    // Wires.
    for (let q = 0; q < n; q++) {
      const y = TOP + q * ROW_H + ROW_H / 2;
      svg.append(
        s("text", { class: "kc-qlabel", x: labelW - 12, y: y + 1 }, qubitName(q)[0], s("tspan", { class: "kc-qsub", dy: 5 }, String(qubitName(q)[1]))),
        s("line", { class: "kc-wire", x1: labelW, y1: y, x2: labelW + cols * COL_W, y2: y }),
      );
    }

    // A faint divider between qubit registers.
    let start = 0;
    for (const [, size] of (model.get("registers") || []).slice(0, -1)) {
      start += size;
      const y = TOP + start * ROW_H;
      svg.append(s("line", { class: "kc-regsep", x1: 4, y1: y, x2: labelW + cols * COL_W, y2: y }));
    }

    // Groups: a dashed box around the cells of their gates.
    const groups = new Map();
    ops().forEach((op, i) => {
      if (!op.group) return;
      if (!groups.has(op.group)) groups.set(op.group, []);
      groups.get(op.group).push([op, i]);
    });
    for (const [id, members] of groups) {
      const colsOf = members.map(([op]) => op.col);
      const qs = members.flatMap(([op]) => op.qubits);
      const c0 = Math.min(...colsOf);
      const q0 = Math.min(...qs);
      const on = members.some(([, i]) => selSet.has(i));
      svg.append(
        s(
          "rect",
          {
            class: `kc-groupbox${on ? " kc-groupbox-on" : ""}`,
            x: labelW + c0 * COL_W + 3,
            y: TOP + q0 * ROW_H + 3,
            width: (Math.max(...colsOf) - c0 + 1) * COL_W - 6,
            height: (Math.max(...qs) - q0 + 1) * ROW_H - 6,
            rx: 8,
          },
          s("title", {}, `Group ${id}`),
        ),
      );
    }

    // Ops, by column.
    const byCol = new Map();
    ops().forEach((op, i) => {
      if (!byCol.has(op.col)) byCol.set(op.col, []);
      byCol.get(op.col).push([op, i]);
    });
    const rowY = (q) => TOP + q * ROW_H + ROW_H / 2;

    for (const [col, list] of byCol) {
      const x = labelW + col * COL_W + COL_W / 2;
      const controls = list.filter(([op]) => op.gate === "CTRL" || op.gate === "ACTRL");
      const gates = list.filter(([op]) => op.gate !== "CTRL" && op.gate !== "ACTRL");
      // Like qulib.draw, controls and connectors take the color of the controlled gate.
      // Controlled gates other than X, Y and Z take draw's default gate color.
      const ctrlColor = (name) => gateClass(["X", "Y", "Z", "CNOT"].includes(name) ? name : "CTRL");
      const colorOf = !gates.length ? "kc-inactive" : controls.length ? ctrlColor(gates[0][0].gate) : gateClass(gates[0][0].gate);
      const layer = s("g", { class: colorOf });

      // Vertical connector through every qubit touched in this column.
      if (controls.length) {
        const qs = list.flatMap(([op]) => op.qubits);
        layer.append(s("line", { class: "kc-connector", x1: x, y1: rowY(Math.min(...qs)), x2: x, y2: rowY(Math.max(...qs)) }));
      }
      for (const [op] of gates) {
        if (op.qubits.length > 1) {
          layer.append(s("line", { class: "kc-connector", x1: x, y1: rowY(Math.min(...op.qubits)), x2: x, y2: rowY(Math.max(...op.qubits)) }));
        }
      }

      for (const [op, index] of list) {
        const info = cat[op.gate];
        if (!info) continue;
        const selected = selSet.has(index);
        const isCtrl = op.gate === "CTRL" || op.gate === "ACTRL";
        const opColor = isCtrl ? "" : " " + (controls.length ? ctrlColor(op.gate) : gateClass(op.gate));
        const g = s("g", { class: `kc-op-wrap${opColor}`, "data-index": index });
        if (isCtrl) {
          const y = rowY(op.qubits[0]);
          g.append(
            s("rect", { class: "kc-hit", x: x - 14, y: y - 14, width: 28, height: 28 }),
            s("circle", { class: `kc-ctrl ${op.gate === "ACTRL" ? "kc-actrl" : ""}${selected ? " kc-selected" : ""}`, cx: x, cy: y, r: 5.5 }),
          );
        } else if (op.gate === "X" && controls.length) {
          // Controlled X target, drawn as ⊕ like qulib.draw.
          const y = rowY(op.qubits[0]);
          const r = 13;
          g.classList.add(gateClass("CX"));
          g.append(
            s("circle", { class: `kc-target${selected ? " kc-selected" : ""}`, cx: x, cy: y, r }),
            s("path", { class: "kc-target-plus", d: `M${x - r + 4},${y}H${x + r - 4}M${x},${y - r + 4}V${y + r - 4}` }),
          );
        } else if (op.gate === "CNOT") {
          // Control dot on the first qubit, ⊕ on the second.
          const yc = rowY(op.qubits[0]);
          const yt = rowY(op.qubits[1]);
          const r = 13;
          g.append(
            s("rect", { class: "kc-hit", x: x - 14, y: yc - 14, width: 28, height: 28 }),
            s("circle", { class: `kc-ctrl${selected ? " kc-selected" : ""}`, cx: x, cy: yc, r: 5.5 }),
            s("circle", { class: `kc-target${selected ? " kc-selected" : ""}`, cx: x, cy: yt, r }),
            s("path", { class: "kc-target-plus", d: `M${x - r + 4},${yt}H${x + r - 4}M${x},${yt - r + 4}V${yt + r - 4}` }),
          );
        } else if (op.gate === "SWAP") {
          for (const q of op.qubits) {
            const y = rowY(q);
            const d = 8;
            g.append(
              s("rect", { class: "kc-hit", x: x - 14, y: y - 14, width: 28, height: 28 }),
              s("path", { class: `kc-swap${selected ? " kc-selected" : ""}`, d: `M${x - d},${y - d}L${x + d},${y + d}M${x + d},${y - d}L${x - d},${y + d}` }),
            );
          }
        } else {
          for (const q of op.qubits) g.append(gateBox(op, info, q, x, rowY(q), selected));
        }
        g.append(s("title", {}, `${info.description}${op.params?.length ? "\n" + info.params.map((p, i) => `${angleLabel(info, p)} = ${op.params[i]}`).join("\n") : ""}\nClick to select, Shift+click to add, drag to move, drag off to remove`));
        g.addEventListener("pointerdown", (e) => startDrag(e, op.gate, index));
        layer.append(g);
      }
      svg.append(layer);
    }

    // Columns not included in the shown state.
    if (simulate && simCol >= 0) {
      const x = labelW + (simCol + 1) * COL_W;
      svg.append(s("rect", { class: "kc-dim", x, y: TOP - 2, width: (cols - simCol - 1) * COL_W, height: n * ROW_H + 4 }));
    }

    // Probability of |1> per qubit.
    if (simulate && result.marginals) {
      const x0 = labelW + cols * COL_W + 10;
      result.marginals.forEach((p, q) => {
        if (q >= n) return;
        const y = rowY(q);
        svg.append(
          s("rect", { class: "kc-marg-bg", x: x0, y: y - 8, width: 40, height: 16, rx: 3 }),
          s("rect", { class: "kc-marg", x: x0, y: y - 8, width: 40 * p, height: 16, rx: 3 }),
          s("text", { class: "kc-marg-text", x: x0 + 46, y: y + 1 }, `${Math.round(p * 100)}%`),
          s("title", {}, `P(${qubitText(q)} = 1) = ${fmt(p, 4)}`),
        );
      });
    }

    svg.append(handles);
    highlight = s("g", { class: "kc-highlight" });
    marquee = s("rect", { class: "kc-marquee", width: 0, height: 0 });
    svg.append(marquee);
    svg.append(highlight);
    put(canvasWrap, svg);
  }

  // ---------- dragging ----------
  function cellAt(clientX, clientY) {
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const col = Math.floor((x - labelW) / COL_W);
    const row = Math.floor((y - TOP) / ROW_H);
    if (col < 0 || col >= numCols() || row < 0 || row >= nq()) return null;
    return { col, row };
  }

  const cellRect = (col, q) => ({ x: labelW + col * COL_W + 3, y: TOP + q * ROW_H + 3, width: COL_W - 6, height: ROW_H - 6, rx: 6 });

  // Where the dragged gates would land, given the cell under the pointer.
  function showHighlight(cell) {
    if (!highlight) return;
    put(highlight);
    const d = state.drag;
    if (!cell || !d) return;
    if (d.indices) {
      const all = ops();
      const moving = d.indices.map((i) => all[i]);
      const anchor = all[d.index];
      const [dc, dq] = clampShift(moving, cell.col - anchor.col, cell.row - anchor.qubits[0]);
      for (const op of moving) for (const q of op.qubits) highlight.append(s("rect", cellRect(op.col + dc, q + dq)));
      return;
    }
    const qs = targetQubits(d.gate, cell.row, null);
    if (qs) for (const q of qs) highlight.append(s("rect", cellRect(cell.col, q)));
  }

  // Drag a palette gate (index null) or the selection that op `index` is in.
  function startDrag(e, gate, index) {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    root.focus({ preventScroll: true });
    const info = catalog()[gate];
    const additive = e.shiftKey || e.ctrlKey || e.metaKey;
    let indices = null;
    if (index !== null) {
      const own = withGroups([index]);
      if (!additive && !state.selection.has(keyOf(ops()[index]))) select(own);
      indices = withGroups(selectedIndices());
      if (!indices.includes(index)) indices = own;
    }
    state.drag = { gate, index, indices, x0: e.clientX, y0: e.clientY, moved: false, ghost: null };

    const move = (ev) => {
      const d = state.drag;
      if (!d) return;
      if (!d.moved && Math.hypot(ev.clientX - d.x0, ev.clientY - d.y0) < 4) return;
      if (!d.moved) {
        d.moved = true;
        const many = d.indices && d.indices.length > 1;
        d.ghost = h(
          "div",
          { class: `kc-ghost kc-tile ${many ? "kc-ghost-many" : gateClass(info.name)}` },
          many ? `${d.indices.length} gates` : info.name === "SWAP" ? "SWAP" : info.label,
        );
        document.body.append(d.ghost);
        root.classList.add("kc-dragging");
      }
      d.ghost.style.left = `${ev.clientX}px`;
      d.ghost.style.top = `${ev.clientY}px`;
      const cell = cellAt(ev.clientX, ev.clientY);
      d.ghost.classList.toggle("kc-remove", !cell && d.index !== null);
      showHighlight(cell);
    };

    const up = (ev) => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", up);
      document.removeEventListener("pointercancel", up);
      const d = state.drag;
      state.drag = null;
      root.classList.remove("kc-dragging");
      if (d?.ghost) d.ghost.remove();
      showHighlight(null);
      if (!d) return;
      if (!d.moved) {
        // A click: select the gate (and its group); Shift/Ctrl toggles it.
        if (d.index !== null) {
          const own = withGroups([d.index]);
          if (additive) toggle(own);
          else select(own);
          renderAll();
        }
        return;
      }
      const cell = ev.type === "pointerup" ? cellAt(ev.clientX, ev.clientY) : null;
      if (d.index === null) {
        if (cell) place(d.gate, cell.col, cell.row);
      } else if (cell) {
        const anchor = ops()[d.index];
        moveOps(d.indices, cell.col - anchor.col, cell.row - anchor.qubits[0]);
      } else if (ev.type === "pointerup") {
        removeOps(d.indices);
      }
    };

    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", up);
    document.addEventListener("pointercancel", up);
  }

  // Drag on the empty canvas: select every gate whose cells the box touches.
  function startMarquee(e) {
    if (e.button !== 0 || !svg) return;
    e.preventDefault();
    root.focus({ preventScroll: true });
    const additive = e.shiftKey || e.ctrlKey || e.metaKey;
    const origin = svg.getBoundingClientRect();
    const x0 = e.clientX - origin.left;
    const y0 = e.clientY - origin.top;
    let box = null;

    const move = (ev) => {
      const x1 = ev.clientX - origin.left;
      const y1 = ev.clientY - origin.top;
      if (!box && Math.hypot(x1 - x0, y1 - y0) < 4) return;
      box = { x: Math.min(x0, x1), y: Math.min(y0, y1), width: Math.abs(x1 - x0), height: Math.abs(y1 - y0) };
      for (const [k, v] of Object.entries(box)) marquee.setAttribute(k, v);
    };
    const up = () => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", up);
      document.removeEventListener("pointercancel", up);
      if (!box) {
        if (!additive && state.selection.size) {
          state.selection = new Set();
          renderAll();
        }
        return;
      }
      const hit = [];
      ops().forEach((op, i) => {
        const x = labelW + op.col * COL_W;
        const y = TOP + Math.min(...op.qubits) * ROW_H;
        const bottom = TOP + (Math.max(...op.qubits) + 1) * ROW_H;
        if (x < box.x + box.width && x + COL_W > box.x && y < box.y + box.height && bottom > box.y) hit.push(i);
      });
      select(withGroups(hit), additive);
      renderAll();
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", up);
    document.addEventListener("pointercancel", up);
  }

  // ---------- inspector ----------
  const btn = (label, title, onclick, extra = "") => h("button", { class: `kc-btn kc-small-btn ${extra}`, title, onclick }, label);
  const MOD = /Mac|iPhone|iPad/.test(navigator.platform || "") ? "⌘" : "Ctrl+";

  // Copy, paste, group and remove, for one or more selected gates.
  function selectionActions(indices) {
    const all = ops();
    const grouped = indices.some((i) => all[i].group);
    const oneGroup = grouped && new Set(indices.map((i) => all[i].group)).size === 1;
    return h(
      "div",
      { class: "kc-actions" },
      indices.length > 1 && !oneGroup ? btn("Group", `Group the selected gates (${MOD}G)`, groupSelection) : null,
      grouped ? btn("Ungroup", `Ungroup (${MOD}Shift+G)`, ungroupSelection) : null,
      btn("Copy", `Copy (${MOD}C)`, () => { copySelection(); renderInspector(); }),
      getClipboard() ? btn("Paste", `Paste after the selection (${MOD}V)`, paste) : null,
      btn(indices.length > 1 ? "Remove" : "Remove gate", "Remove (Delete)", () => removeOps(indices), "kc-danger"),
    );
  }

  function renderInspector() {
    const indices = selectedIndices();
    if (!indices.length) {
      put(inspector, 
        h("div", { class: "kc-panel-title" }, "Gate"),
        h("div", { class: "kc-hint" }, "Drag gates from the palette onto the wires. Click a gate to edit it. ● and ○ control every gate in their column."),
        h("div", { class: "kc-hint" }, `Drag on an empty area, or Shift+click, to select several gates; then group (${MOD}G), copy (${MOD}C) and paste (${MOD}V) them.`),
        getClipboard() ? h("div", { class: "kc-actions" }, btn("Paste", `Paste after the last column (${MOD}V)`, paste)) : null,
      );
      return;
    }
    if (indices.length > 1) {
      const all = ops();
      const ids = new Set(indices.map((i) => all[i].group));
      const group = ids.size === 1 ? [...ids][0] : null;
      const cols = indices.map((i) => all[i].col);
      put(inspector, 
        h(
          "div",
          { class: "kc-panel-title kc-gap" },
          `${indices.length} gates`,
          h("span", { class: "kc-muted" }, group ? `group ${group}` : `columns ${Math.min(...cols)}–${Math.max(...cols)}`),
        ),
        h("div", { class: "kc-hint" }, "Drag any of them to move them together."),
        selectionActions(indices),
      );
      return;
    }
    const index = indices[0];
    const op = ops()[index];
    const info = catalog()[op.gate];
    const n = nq();
    const rows = [];

    info.params.forEach((p, i) => {
      rows.push(
        h(
          "label",
          { class: "kc-field" },
          h("span", { class: "kc-field-name" }, angleLabel(info, p)),
          h("input", {
            class: "kc-input",
            value: op.params[i],
            spellcheck: "false",
            title: "Angle: a number, pi/4, 2*theta… Other names (alpha, β, theta1…) become gate parameters.",
            onchange: (e) => {
              const params = [...op.params];
              params[i] = e.target.value.trim() || "0";
              updateOp(index, { params });
            },
          }),
        ),
      );
    });

    op.qubits.forEach((q, i) => {
      const select = h(
        "select",
        {
          class: "kc-input",
          onchange: (e) => {
            const qubits = [...op.qubits];
            const v = Number(e.target.value);
            const other = qubits.indexOf(v);
            if (other >= 0) qubits[other] = qubits[i];
            qubits[i] = v;
            const clash = ops().some(
              (o, j) => j !== index && o.col === op.col && o.qubits.some((x) => qubits.includes(x)),
            );
            if (clash) {
              renderInspector();
              return;
            }
            selectOps([{ col: op.col, qubits }]);
            updateOp(index, { qubits });
          },
        },
        Array.from({ length: n }, (_, k) => h("option", { value: k, selected: k === q ? "selected" : null }, qubitText(k))),
      );
      rows.push(h("label", { class: "kc-field" }, h("span", { class: "kc-field-name" }, op.gate === "CNOT" ? ["control", "target"][i] : op.qubits.length > 1 ? `qubit ${i}` : "qubit"), select));
    });

    put(inspector, 
      h("div", { class: "kc-panel-title kc-gap" }, info.label, h("span", { class: "kc-muted" }, `column ${op.col}`)),
      h("div", { class: "kc-hint" }, info.description),
      ...rows,
      op.group ? h("div", { class: "kc-hint" }, `In group ${op.group}.`) : null,
      selectionActions([index]),
    );
  }

  // ---------- parameters ----------
  function renderParams() {
    const names = model.get("param_names");
    if (!names.length) {
      paramsPanel.style.display = "none";
      return;
    }
    paramsPanel.style.display = "";
    const values = model.get("params");
    // Same parameters: only sync the values, so a slider being dragged keeps
    // its focus. Each control updates every one except the focused input.
    const key = names.join(",");
    if (paramsPanel.dataset.names === key) {
      for (const row of paramsPanel.querySelectorAll(".kc-param")) {
        const v = values[row.dataset.name] ?? 0;
        for (const input of row.querySelectorAll("input")) {
          if (input !== document.activeElement) input.value = input.type === "range" ? v : Number(v.toFixed(4));
        }
      }
      return;
    }
    paramsPanel.dataset.names = key;
    const setValue = (name, v) => {
      if (!Number.isFinite(v)) return;
      model.set("params", { ...model.get("params"), [name]: v });
      model.save_changes();
    };
    put(paramsPanel, 
      h("div", { class: "kc-panel-title" }, "Parameters"),
      ...names.map((name) => {
        const v = values[name] ?? 0;
        const slider = h("input", {
          type: "range",
          min: (-2 * Math.PI).toFixed(4),
          max: (2 * Math.PI).toFixed(4),
          step: "0.001",
          value: v,
        });
        const box = h("input", {
          class: "kc-input kc-num",
          type: "number",
          step: "0.01",
          value: Number(v.toFixed(4)),
        });
        slider.addEventListener("input", () => {
          box.value = Number(Number(slider.value).toFixed(4));
          setValue(name, Number(slider.value));
        });
        box.addEventListener("change", () => {
          const x = Number(box.value);
          if (Number.isFinite(x)) slider.value = x;
          setValue(name, x);
        });
        const row = h("div", { class: "kc-param" }, h("span", { class: "kc-field-name", title: name }, pretty(name)), slider, box);
        row.dataset.name = name;
        return row;
      }),
    );
  }

  // ---------- state ----------
  // One ket per register: |01⟩|1⟩ as bits, |1⟩|1⟩ unsigned, |1⟩|−1⟩ signed
  // (two's complement). The first qubit of a register is its most
  // significant bit. Arithmetic, not bitwise, to handle 32 qubits.
  function basisKets(basis, format) {
    const n = nq();
    let start = 0;
    const kets = [];
    for (const [, size] of model.get("registers") || [["q", n]]) {
      const value = Math.floor(basis / 2 ** (n - start - size)) % 2 ** size;
      start += size;
      if (format === "bits") kets.push(value.toString(2).padStart(size, "0"));
      else if (format === "signed" && value >= 2 ** (size - 1)) kets.push(`−${2 ** size - value}`);
      else kets.push(String(value));
    }
    return kets.map((k) => `|${k}⟩`).join("");
  }

  const BASIS_FORMATS = [
    ["bits", "Bits", "Each register as a bitstring"],
    ["unsigned", "Unsigned", "Each register as an unsigned integer"],
    ["signed", "Signed", "Each register as a signed integer (two's complement)"],
  ];

  function renderState() {
    if (!model.get("simulate")) {
      statePanel.style.display = "none";
      return;
    }
    statePanel.style.display = "";
    const result = model.get("sim_result") || {};
    const simCol = model.get("sim_column");
    const title = h(
      "div",
      { class: "kc-panel-title" },
      simCol >= 0 ? `State after column ${simCol}` : "Final state",
      simCol >= 0 ? h("button", { class: "kc-link", onclick: () => { model.set("sim_column", -1); model.save_changes(); } }, "show final") : null,
      h("span", { class: "kc-spacer" }),
      h(
        "span",
        { class: "kc-seg", role: "group", "aria-label": "Basis state format" },
        BASIS_FORMATS.map(([key, label, tip]) =>
          h(
            "button",
            {
              class: `kc-seg-btn${state.basisFormat === key ? " kc-seg-on" : ""}`,
              title: tip,
              "aria-pressed": state.basisFormat === key ? "true" : "false",
              onclick: () => {
                state.basisFormat = key;
                renderState();
              },
            },
            label,
          ),
        ),
      ),
    );
    if (!result.states) {
      put(statePanel, title, h("div", { class: "kc-hint" }, "No state to show."));
      return;
    }
    const format = state.basisFormat;
    // Same width for every ket (monospace font), so that the bars line up.
    const ketWidth = Math.max(...result.states.map(([basis]) => basisKets(basis, format).length));
    // With an observable, the bars show the energy <b|H|b> of each basis
    // state, relative to the lowest and highest of the state.
    const withEnergy = typeof result.energy === "number" && Array.isArray(result.energy_range);
    const [lo, hi] = withEnergy ? result.energy_range : [0, 0];
    const rows = result.states.map(([basis, re, im, prob, ph, energy]) =>
      h(
        "div",
        {
          class: `kc-srow${withEnergy ? " kc-srow-energy" : ""}`,
          title: `${basisKets(basis, format)}: amplitude ${fmt(re, 5)} ${im < 0 ? "−" : "+"} ${fmt(Math.abs(im), 5)}i, phase ${fmt((ph * 180) / Math.PI, 1)}°${withEnergy ? `, energy ${fmtSigned(energy, 5)}` : ""}`,
        },
        h("span", { class: "kc-ket", style: `min-width:${ketWidth}ch` }, basisKets(basis, format)),
        h("span", { class: "kc-bar" }, h("span", { class: "kc-bar-fill", style: `width:${(prob * 100).toFixed(2)}%;background:${withEnergy ? energyColor(energy, lo, hi) : phaseColor(ph)}` })),
        h("span", { class: "kc-amp" }, `${fmt(re)}${im < 0 ? "−" : "+"}${fmt(Math.abs(im))}i`),
        withEnergy ? h("span", { class: "kc-energy", title: "Energy ⟨b|H|b⟩ of the basis state" }, `E = ${fmtSigned(energy)}`) : null,
        h("span", { class: "kc-prob" }, `${(prob * 100).toFixed(1)}%`),
      ),
    );
    const energyBox = withEnergy
      ? h(
          "div",
          { class: "kc-energy-box" },
          h("span", { class: "kc-energy-total", title: "Expected value ⟨ψ|H|ψ⟩ of the observable" }, "⟨H⟩ = ", h("b", {}, fmtSigned(result.energy, 5))),
          h("span", { class: "kc-spacer" }),
          h("span", { class: "kc-muted" }, fmtSigned(lo)),
          h("span", { class: "kc-energy-scale", style: `background:linear-gradient(to right, ${energyColor(0, 0, 1)}, ${energyColor(0.5, 0, 1)}, ${energyColor(1, 0, 1)})` }),
          h("span", { class: "kc-muted" }, fmtSigned(hi)),
        )
      : null;
    put(statePanel, 
      title,
      energyBox,
      h("div", { class: "kc-muted" }, `${result.num_states} basis state${result.num_states === 1 ? "" : "s"}${result.truncated ? `, showing the ${result.states.length} most probable` : ""} · bar color = ${withEnergy ? "energy ⟨b|H|b⟩, lowest to highest of the state" : "phase"} · one ket per register, ${format === "bits" ? `${qubitText(0)} is the leftmost bit` : "first qubit is the most significant bit"}`),
      h("div", { class: "kc-states" }, rows),
    );
  }

  // ---------- code ----------
  async function copyCode(e) {
    const button = e.currentTarget;
    const code = model.get("code");
    let ok = false;
    try {
      await navigator.clipboard.writeText(code);
      ok = true;
    } catch {
      // Clipboard API blocked (e.g. some notebook iframes): copy through a
      // temporary text area.
      const area = h("textarea", { class: "kc-copy-area", readonly: "readonly" });
      area.value = code;
      document.body.append(area);
      area.select();
      try {
        ok = document.execCommand("copy");
      } catch {
        ok = false;
      }
      area.remove();
    }
    button.textContent = ok ? "Copied" : "Copy failed";
    setTimeout(() => (button.textContent = "Copy code"), 1200);
  }

  // ---------- Bloch spheres ----------
  // Oblique view: x toward the viewer (lower left), y to the right, z up.
  const VIEW_AZ = (32 * Math.PI) / 180;
  const VIEW_EL = (24 * Math.PI) / 180;
  const RIGHT = [-Math.sin(VIEW_AZ), Math.cos(VIEW_AZ), 0];
  const UP = [-Math.cos(VIEW_AZ) * Math.sin(VIEW_EL), -Math.sin(VIEW_AZ) * Math.sin(VIEW_EL), Math.cos(VIEW_EL)];
  const DEPTH = [Math.cos(VIEW_AZ) * Math.cos(VIEW_EL), Math.sin(VIEW_AZ) * Math.cos(VIEW_EL), Math.sin(VIEW_EL)];
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

  function blochSphere(q, [bx, by, bz]) {
    const R = 46;
    const size = 2 * R + 34;
    const c = size / 2;
    const proj = (v, r = R) => [c + r * dot3(v, RIGHT), c - r * dot3(v, UP)];
    const pt = (v, r) => proj(v, r).map((t) => t.toFixed(2)).join(",");

    // A great circle, front half solid and back half dashed.
    const circle = (u, w, cls) => {
      const front = [];
      const back = [];
      for (let i = 0; i <= 72; i++) {
        const t = (i / 72) * 2 * Math.PI;
        const v = [0, 1, 2].map((k) => Math.cos(t) * u[k] + Math.sin(t) * w[k]);
        (dot3(v, DEPTH) >= 0 ? front : back).push(v);
      }
      const path = (pts) => {
        let d = "";
        let prev = null;
        for (const v of pts) {
          const p = proj(v);
          const far = prev && Math.hypot(p[0] - prev[0], p[1] - prev[1]) > R / 3;
          d += `${d && !far ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`;
          prev = p;
        }
        return d;
      };
      return [
        s("path", { class: `${cls} kc-bl-back`, d: path(back) }),
        s("path", { class: cls, d: path(front) }),
      ];
    };

    const len = Math.hypot(bx, by, bz);
    // Shorter than 1: the qubit is not in a pure state of its own
    // (entangled, or mixed). Drawn in another color.
    const mixed = len < 0.999;
    const tip = proj([bx, by, bz]);
    const axes = [
      [[1, 0, 0], "x"],
      [[0, 1, 0], "y"],
      [[0, 0, 1], "z"],
    ];
    const svgEl = s(
      "svg",
      { class: "kc-bl-svg", width: size, height: size, viewBox: `0 0 ${size} ${size}` },
      s("circle", { class: "kc-bl-ball", cx: c, cy: c, r: R }),
      circle([1, 0, 0], [0, 1, 0], "kc-bl-ring"),
      circle([1, 0, 0], [0, 0, 1], "kc-bl-ring kc-bl-faint"),
      axes.map(([v]) => s("line", { class: "kc-bl-axis", x1: proj(v.map((t) => -t))[0], y1: proj(v.map((t) => -t))[1], x2: proj(v)[0], y2: proj(v)[1] })),
      axes.map(([v, label]) => {
        const [x, y] = proj(v, R + 10);
        return s("text", { class: "kc-bl-label", x, y }, label);
      }),
      s("text", { class: "kc-bl-ket", x: c + 9, y: proj([0, 0, 1], R + 9)[1] }, "|0⟩"),
      s("text", { class: "kc-bl-ket", x: c + 9, y: proj([0, 0, -1], R + 9)[1] }, "|1⟩"),
      len > 1e-6 ? s("line", { class: `kc-bl-vec${mixed ? " kc-bl-mixed" : ""}`, x1: c, y1: c, x2: tip[0], y2: tip[1] }) : null,
      s("circle", { class: `kc-bl-tip${mixed ? " kc-bl-mixed" : ""}`, cx: tip[0], cy: tip[1], r: 4 }),
      s("title", {}, `${qubitText(q)}: x = ${fmt(bx)}, y = ${fmt(by)}, z = ${fmt(bz)}\nlength ${fmt(len)}${len < 0.999 ? " (entangled)" : ""}`),
    );
    return h(
      "div",
      { class: "kc-bl-item" },
      h("div", { class: "kc-bl-name" }, h("i", {}, qubitName(q)[0]), h("sub", {}, String(qubitName(q)[1]))),
      svgEl,
    );
  }

  function renderBloch() {
    const vectors = model.get("simulate") ? (model.get("sim_result") || {}).bloch : null;
    if (!vectors) {
      blochPanel.style.display = "none";
      return;
    }
    blochPanel.style.display = "";
    put(blochPanel, 
      h(
        "div",
        { class: "kc-panel-title kc-gap" },
        "Bloch spheres",
        h(
          "span",
          { class: "kc-muted" },
          "(⟨X⟩, ⟨Y⟩, ⟨Z⟩) of each qubit · ",
          h("span", { class: "kc-bl-key" }),
          " pure · ",
          h("span", { class: "kc-bl-key kc-bl-mixed" }),
          " entangled",
        ),
      ),
      h("div", { class: "kc-bl-grid" }, vectors.map((v, q) => blochSphere(q, v))),
    );
  }

  function renderError() {
    const err = model.get("error");
    errorBox.textContent = err;
    errorBox.style.display = err ? "" : "none";
  }

  function renderAll() {
    // Re-rendering removes the focused button: keep the focus in the editor
    // so that keyboard shortcuts (Ctrl+Z, Delete) keep working.
    const hadFocus = root.contains(document.activeElement);
    renderToolbar();
    renderCircuit();
    renderInspector();
    renderParams();
    renderState();
    renderBloch();
    renderError();
    if (hadFocus && !root.contains(document.activeElement)) root.focus({ preventScroll: true });
  }

  // ---------- keyboard ----------
  root.addEventListener("keydown", (e) => {
    const tag = e.target.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === "z" && !e.shiftKey) {
      undo();
    } else if (mod && (e.key.toLowerCase() === "y" || (e.key.toLowerCase() === "z" && e.shiftKey))) {
      redo();
    } else if ((e.key === "Delete" || e.key === "Backspace") && state.selection.size) {
      removeOps(selectedIndices());
    } else if (mod && e.key.toLowerCase() === "a") {
      select(ops().map((_, i) => i));
      renderAll();
    } else if (mod && e.key.toLowerCase() === "g" && e.shiftKey) {
      ungroupSelection();
    } else if (mod && e.key.toLowerCase() === "g") {
      groupSelection();
    } else if (mod && e.key.toLowerCase() === "c") {
      if (copySelection()) renderInspector();
    } else if (mod && e.key.toLowerCase() === "x") {
      if (copySelection()) removeOps(selectedIndices());
    } else if (mod && e.key.toLowerCase() === "v") {
      paste();
    } else if (mod && e.key.toLowerCase() === "d") {
      if (copySelection()) paste();
    } else if (e.key === "Escape") {
      state.selection = new Set();
      renderAll();
    } else {
      return;
    }
    // Keep Jupyter from treating the keys as notebook commands.
    e.preventDefault();
    e.stopPropagation();
  });

  // On the empty canvas: a click clears the selection, a drag selects.
  canvasWrap.addEventListener("pointerdown", (e) => {
    const cls = e.target.classList;
    if (e.target === svg || cls?.contains("kc-wire") || cls?.contains("kc-regsep") || cls?.contains("kc-groupbox") || cls?.contains("kc-dim")) {
      startMarquee(e);
    }
  });

  renderPalette();
  renderAll();
  const events = [
    "change:ops",
    "change:num_qubits",
    "change:registers",
    "change:sim_result",
    "change:sim_column",
    "change:simulate",
    "change:has_obs",
    "change:param_names",
    "change:code",
    "change:error",
    "change:file",
    "change:status",
  ];
  const onChange = () => renderAll();
  for (const ev of events) model.on(ev, onChange);
  model.on("change:params", renderParams);
  model.on("change:catalog", renderPalette);
  // Copies made in another composer, here or in another tab.
  const onClipboard = () => renderInspector();
  const onStorage = (e) => {
    if (e.key === CLIPBOARD_KEY) renderInspector();
  };
  globalThis.addEventListener?.(CLIPBOARD_EVENT, onClipboard);
  globalThis.addEventListener?.("storage", onStorage);

  return () => {
    globalThis.removeEventListener?.(CLIPBOARD_EVENT, onClipboard);
    globalThis.removeEventListener?.("storage", onStorage);
    for (const ev of events) model.off(ev, onChange);
    model.off("change:params", renderParams);
    model.off("change:catalog", renderPalette);
  };
}

export default { render };
