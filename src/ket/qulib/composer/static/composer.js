// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

// Front end of ket.qulib.composer.Composer (an anywidget module).
// The Python side validates the circuit, builds the gate and runs KBW;
// this side only edits the `ops` list and shows what Python sends back.

const SVG_NS = "http://www.w3.org/2000/svg";
const LABEL_W = 52;
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

// Show angles the way qulib.draw does: pi -> π, theta -> θ...
const PRETTY = { pi: "π", tau: "τ", theta: "θ", phi: "φ", lambda: "λ" };
function pretty(source) {
  return source
    .replace(/\b(pi|tau|theta|phi|lambda)\b/g, (m) => PRETTY[m])
    .replace(/\s*\*\s*/g, "")
    .replace(/\s+/g, "");
}

// Gate colors come from qulib.DRAW_STYLE, through the kc-gate-<name> classes.
const gateClass = (name) => `kc-gate-${name}`;

function clone(x) {
  return JSON.parse(JSON.stringify(x));
}

function fmt(x, digits = 3) {
  const v = Math.abs(x) < 0.5 * 10 ** -digits ? 0 : x;
  return v.toFixed(digits);
}

function phaseColor(ph) {
  // Phase 0 is blue; the hue turns with the phase.
  const hue = ((((ph * 180) / Math.PI + 215) % 360) + 360) % 360;
  return `hsl(${hue}, 70%, 50%)`;
}

function render({ model, el }) {
  const state = {
    selected: null, // {col, qubit} of the selected op
    drag: null,
    history: [],
    future: [],
  };

  const catalog = () => {
    const out = {};
    for (const g of model.get("catalog")) out[g.name] = g;
    return out;
  };
  const ops = () => model.get("ops");
  const nq = () => model.get("num_qubits");

  // ---------- model updates ----------
  function commit(newOps, extra = {}) {
    state.history.push({ ops: clone(ops()), num_qubits: nq() });
    if (state.history.length > 200) state.history.shift();
    state.future = [];
    model.set("ops", newOps);
    for (const [k, v] of Object.entries(extra)) model.set(k, v);
    model.save_changes();
  }

  function restore(from, to) {
    const snap = from.pop();
    if (!snap) return;
    to.push({ ops: clone(ops()), num_qubits: nq() });
    model.set("num_qubits", snap.num_qubits);
    model.set("ops", snap.ops);
    model.save_changes();
  }
  const undo = () => restore(state.history, state.future);
  const redo = () => restore(state.future, state.history);

  function findOp(sel) {
    if (!sel) return -1;
    return ops().findIndex(
      (op) => op.col === sel.col && op.qubits[0] === sel.qubit,
    );
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

  function place(gateName, col, row, movingIndex) {
    const current = ops();
    const moving = movingIndex === null ? null : current[movingIndex];
    const qubits = targetQubits(gateName, row, moving);
    if (!qubits) return;
    const info = catalog()[gateName];
    const op = moving
      ? { ...clone(moving), col, qubits }
      : {
          gate: gateName,
          col,
          qubits,
          params: info.params.map((p) => DEFAULT_PARAMS[p] ?? "0"),
        };
    const kept = current.filter(
      (o, i) =>
        i !== movingIndex &&
        !(o.col === col && o.qubits.some((q) => qubits.includes(q))),
    );
    kept.push(op);
    state.selected = { col, qubit: qubits[0] };
    commit(kept);
  }

  function removeOp(index) {
    if (index < 0) return;
    const next = ops().filter((_, i) => i !== index);
    state.selected = null;
    commit(next);
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

  function insertColumn(col) {
    const next = clone(ops()).map((op) =>
      op.col >= col ? { ...op, col: op.col + 1 } : op,
    );
    commit(next);
  }

  function compact() {
    const cols = [...new Set(ops().map((op) => op.col))].sort((a, b) => a - b);
    const map = new Map(cols.map((c, i) => [c, i]));
    const next = clone(ops()).map((op) => ({ ...op, col: map.get(op.col) }));
    state.selected = null;
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
  const codePanel = h("details", { class: "kc-panel kc-code" });

  root.append(
    toolbar,
    palette,
    h("div", { class: "kc-main" }, h("div", { class: "kc-left" }, canvasWrap, errorBox, codePanel), side),
  );
  side.append(inspector, paramsPanel, statePanel);

  // ---------- toolbar ----------
  function renderToolbar() {
    toolbar.replaceChildren(
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
      h("button", { class: "kc-btn", title: "Remove empty columns", onclick: compact }, "Compact"),
      h(
        "button",
        {
          class: "kc-btn kc-danger",
          title: "Remove every gate",
          onclick: () => {
            state.selected = null;
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
    palette.replaceChildren(
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
    const width = LABEL_W + cols * COL_W + MARG_W;
    const height = TOP + n * ROW_H + 6;
    const cat = catalog();
    const selIndex = findOp(state.selected);

    svg = s("svg", { class: "kc-svg", width, height, viewBox: `0 0 ${width} ${height}` });

    // Column headers.
    for (let c = 0; c < cols; c++) {
      const x = LABEL_W + c * COL_W;
      const active = simCol === c;
      const head = s(
        "g",
        { class: `kc-colhead${active ? " kc-active" : ""}` },
        s("rect", { x: x + 4, y: 2, width: COL_W - 8, height: TOP - 6, rx: 4 }),
        s("text", { x: x + COL_W / 2, y: TOP / 2 + 1 }, String(c)),
        s("title", {}, active ? "Show the final state" : `Show the state after column ${c}\nShift+click: insert an empty column`),
      );
      head.addEventListener("click", (e) => {
        if (e.shiftKey) {
          insertColumn(c);
          return;
        }
        model.set("sim_column", active ? -1 : c);
        model.save_changes();
      });
      svg.append(head);
    }

    // Wires.
    for (let q = 0; q < n; q++) {
      const y = TOP + q * ROW_H + ROW_H / 2;
      svg.append(
        s("text", { class: "kc-qlabel", x: LABEL_W - 12, y: y + 1 }, "q", s("tspan", { class: "kc-qsub", dy: 5 }, String(q))),
        s("line", { class: "kc-wire", x1: LABEL_W, y1: y, x2: LABEL_W + cols * COL_W, y2: y }),
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
      const x = LABEL_W + col * COL_W + COL_W / 2;
      const controls = list.filter(([op]) => op.gate === "CTRL" || op.gate === "ACTRL");
      const gates = list.filter(([op]) => op.gate !== "CTRL" && op.gate !== "ACTRL");
      // Like qulib.draw, controls and connectors take the color of the controlled gate.
      // Controlled gates other than X, Y and Z take draw's default gate color.
      const ctrlColor = (name) => gateClass(["X", "Y", "Z"].includes(name) ? name : "CTRL");
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
        const selected = index === selIndex;
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
        g.append(s("title", {}, `${info.description}${op.params?.length ? "\n" + info.params.map((p, i) => `${PARAM_SYMBOL[p] ?? p} = ${op.params[i]}`).join("\n") : ""}\nClick to edit, drag to move, drag off to remove`));
        g.addEventListener("pointerdown", (e) => startDrag(e, op.gate, index));
        layer.append(g);
      }
      svg.append(layer);
    }

    // Columns not included in the shown state.
    if (simulate && simCol >= 0) {
      const x = LABEL_W + (simCol + 1) * COL_W;
      svg.append(s("rect", { class: "kc-dim", x, y: TOP - 2, width: (cols - simCol - 1) * COL_W, height: n * ROW_H + 4 }));
    }

    // Probability of |1> per qubit.
    if (simulate && result.marginals) {
      const x0 = LABEL_W + cols * COL_W + 10;
      result.marginals.forEach((p, q) => {
        if (q >= n) return;
        const y = rowY(q);
        svg.append(
          s("rect", { class: "kc-marg-bg", x: x0, y: y - 8, width: 40, height: 16, rx: 3 }),
          s("rect", { class: "kc-marg", x: x0, y: y - 8, width: 40 * p, height: 16, rx: 3 }),
          s("text", { class: "kc-marg-text", x: x0 + 46, y: y + 1 }, `${Math.round(p * 100)}%`),
          s("title", {}, `P(q${q} = 1) = ${fmt(p, 4)}`),
        );
      });
    }

    highlight = s("g", { class: "kc-highlight" });
    svg.append(highlight);
    canvasWrap.replaceChildren(svg);
  }

  // ---------- dragging ----------
  function cellAt(clientX, clientY) {
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const col = Math.floor((x - LABEL_W) / COL_W);
    const row = Math.floor((y - TOP) / ROW_H);
    if (col < 0 || col >= numCols() || row < 0 || row >= nq()) return null;
    return { col, row };
  }

  function showHighlight(cell) {
    if (!highlight) return;
    highlight.replaceChildren();
    const d = state.drag;
    if (!cell || !d) return;
    const moving = d.index === null ? null : ops()[d.index];
    const qs = targetQubits(d.gate, cell.row, moving);
    if (!qs) return;
    const x = LABEL_W + cell.col * COL_W;
    for (const q of qs) {
      highlight.append(s("rect", { x: x + 3, y: TOP + q * ROW_H + 3, width: COL_W - 6, height: ROW_H - 6, rx: 6 }));
    }
  }

  function startDrag(e, gate, index) {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    root.focus({ preventScroll: true });
    const info = catalog()[gate];
    state.drag = { gate, index, x0: e.clientX, y0: e.clientY, moved: false, ghost: null };

    const move = (ev) => {
      const d = state.drag;
      if (!d) return;
      if (!d.moved && Math.hypot(ev.clientX - d.x0, ev.clientY - d.y0) < 4) return;
      if (!d.moved) {
        d.moved = true;
        d.ghost = h("div", { class: `kc-ghost kc-tile ${gateClass(info.name)}` }, info.name === "SWAP" ? "SWAP" : info.label);
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
        if (d.index !== null) {
          const op = ops()[d.index];
          state.selected = { col: op.col, qubit: op.qubits[0] };
          renderAll();
        }
        return;
      }
      const cell = ev.type === "pointerup" ? cellAt(ev.clientX, ev.clientY) : null;
      if (cell) place(d.gate, cell.col, cell.row, d.index);
      else if (d.index !== null && ev.type === "pointerup") removeOp(d.index);
    };

    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", up);
    document.addEventListener("pointercancel", up);
  }

  // ---------- inspector ----------
  function renderInspector() {
    const index = findOp(state.selected);
    if (index < 0) {
      state.selected = null;
      inspector.replaceChildren(
        h("div", { class: "kc-panel-title" }, "Gate"),
        h("div", { class: "kc-hint" }, "Drag gates from the palette onto the wires. Click a gate to edit it. ● and ○ control every gate in their column."),
      );
      return;
    }
    const op = ops()[index];
    const info = catalog()[op.gate];
    const n = nq();
    const rows = [];

    info.params.forEach((p, i) => {
      rows.push(
        h(
          "label",
          { class: "kc-field" },
          h("span", { class: "kc-field-name" }, PARAM_SYMBOL[p] ?? p),
          h("input", {
            class: "kc-input",
            value: op.params[i],
            spellcheck: "false",
            title: "Angle: a number, pi/4, 2*theta… Other names become gate parameters.",
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
            state.selected = { col: op.col, qubit: qubits[0] };
            updateOp(index, { qubits });
          },
        },
        Array.from({ length: n }, (_, k) => h("option", { value: k, selected: k === q ? "selected" : null }, `q${k}`)),
      );
      rows.push(h("label", { class: "kc-field" }, h("span", { class: "kc-field-name" }, op.qubits.length > 1 ? `qubit ${i}` : "qubit"), select));
    });

    inspector.replaceChildren(
      h("div", { class: "kc-panel-title kc-gap" }, info.label, h("span", { class: "kc-muted" }, `column ${op.col}`)),
      h("div", { class: "kc-hint" }, info.description),
      ...rows,
      h("button", { class: "kc-btn kc-danger", onclick: () => removeOp(index) }, "Remove gate"),
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
    paramsPanel.replaceChildren(
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
        const row = h("div", { class: "kc-param" }, h("span", { class: "kc-field-name" }, name), slider, box);
        row.dataset.name = name;
        return row;
      }),
    );
  }

  // ---------- state ----------
  function renderState() {
    if (!model.get("simulate")) {
      statePanel.style.display = "none";
      return;
    }
    statePanel.style.display = "";
    const result = model.get("sim_result") || {};
    const n = nq();
    const simCol = model.get("sim_column");
    const title = h(
      "div",
      { class: "kc-panel-title" },
      simCol >= 0 ? `State after column ${simCol}` : "Final state",
      simCol >= 0 ? h("button", { class: "kc-link", onclick: () => { model.set("sim_column", -1); model.save_changes(); } }, "show final") : null,
    );
    if (!result.states) {
      statePanel.replaceChildren(title, h("div", { class: "kc-hint" }, "No state to show."));
      return;
    }
    const rows = result.states.map(([basis, re, im, prob, ph]) =>
      h(
        "div",
        { class: "kc-srow", title: `|${basis}⟩: amplitude ${fmt(re, 5)} ${im < 0 ? "−" : "+"} ${fmt(Math.abs(im), 5)}i, phase ${fmt((ph * 180) / Math.PI, 1)}°` },
        h("span", { class: "kc-ket" }, `|${basis.toString(2).padStart(n, "0")}⟩`),
        h("span", { class: "kc-bar" }, h("span", { class: "kc-bar-fill", style: `width:${(prob * 100).toFixed(2)}%;background:${phaseColor(ph)}` })),
        h("span", { class: "kc-amp" }, `${fmt(re)}${im < 0 ? "−" : "+"}${fmt(Math.abs(im))}i`),
        h("span", { class: "kc-prob" }, `${(prob * 100).toFixed(1)}%`),
      ),
    );
    statePanel.replaceChildren(
      title,
      h("div", { class: "kc-muted" }, `${result.num_states} basis state${result.num_states === 1 ? "" : "s"}${result.truncated ? `, showing the ${result.states.length} most probable` : ""} · bar color = phase · q0 is the leftmost bit`),
      h("div", { class: "kc-states" }, rows),
    );
  }

  // ---------- code ----------
  function renderCode() {
    const open = codePanel.open;
    const code = model.get("code");
    const pre = h("pre", { class: "kc-pre" }, code);
    const copy = h(
      "button",
      {
        class: "kc-btn kc-small-text",
        onclick: async (e) => {
          e.preventDefault();
          try {
            await navigator.clipboard.writeText(code);
            copy.textContent = "Copied";
          } catch {
            const range = document.createRange();
            range.selectNodeContents(pre);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
            copy.textContent = "Selected";
          }
          setTimeout(() => (copy.textContent = "Copy"), 1200);
        },
      },
      "Copy",
    );
    codePanel.replaceChildren(h("summary", { class: "kc-panel-title" }, "Python code ", copy), pre);
    codePanel.open = open;
  }

  function renderError() {
    const err = model.get("error");
    errorBox.textContent = err;
    errorBox.style.display = err ? "" : "none";
  }

  function renderAll() {
    renderToolbar();
    renderCircuit();
    renderInspector();
    renderParams();
    renderState();
    renderCode();
    renderError();
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
    } else if ((e.key === "Delete" || e.key === "Backspace") && state.selected) {
      removeOp(findOp(state.selected));
    } else if (e.key === "Escape") {
      state.selected = null;
      renderAll();
    } else {
      return;
    }
    // Keep Jupyter from treating the keys as notebook commands.
    e.preventDefault();
    e.stopPropagation();
  });

  // Clicking the empty canvas clears the selection.
  canvasWrap.addEventListener("pointerdown", (e) => {
    if (e.target === svg || e.target.classList?.contains("kc-wire")) {
      root.focus({ preventScroll: true });
      if (state.selected) {
        state.selected = null;
        renderAll();
      }
    }
  });

  renderPalette();
  renderAll();
  const events = [
    "change:ops",
    "change:num_qubits",
    "change:sim_result",
    "change:sim_column",
    "change:simulate",
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

  return () => {
    for (const ev of events) model.off(ev, onChange);
    model.off("change:params", renderParams);
    model.off("change:catalog", renderPalette);
  };
}

export default { render };
