"use strict";

// Every value from the API is written with textContent, never innerHTML.

const $ = (selector) => document.querySelector(selector);
const fmt = (value, digits = 3) =>
  value === null || value === undefined ? "–" : Number(value).toFixed(digits);
const fmtRisk = (value) => (value < 1e-4 ? "< 0.0001" : value > 0.9999 ? "> 0.9999" : fmt(value, 4));

async function api(path, body) {
  const options = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map((item) => `${item.loc.slice(1).join(".")}: ${item.msg}`).join("; ")
      : payload.detail || response.statusText;
    throw new Error(detail);
  }
  return payload;
}

function cell(text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

function fillRows(tbody, rows) {
  tbody.replaceChildren(...rows.map((cells) => {
    const tr = document.createElement("tr");
    tr.append(...cells);
    return tr;
  }));
}

function badge(alert) {
  const span = document.createElement("span");
  span.className = alert ? "badge alert" : "badge clear";
  span.textContent = alert ? "▲ Alert" : "● No alert";
  const td = document.createElement("td");
  td.append(span);
  return td;
}

/* ---------- tabs ---------- */

function selectTab(name) {
  document.querySelectorAll('[role="tab"]').forEach((button) => {
    const selected = button.getAttribute("aria-controls") === `tab-${name}`;
    button.setAttribute("aria-selected", String(selected));
    document.getElementById(button.getAttribute("aria-controls")).hidden = !selected;
  });
}

function setupTabs() {
  // Each tab has a shareable address: /#triage, /#lab, /#method.
  document.querySelectorAll('[role="tab"]').forEach((button) => button.addEventListener("click", () => {
    const name = button.getAttribute("aria-controls").replace("tab-", "");
    history.replaceState(null, "", `#${name}`);
    selectTab(name);
  }));
  const requested = location.hash.slice(1);
  if (document.getElementById(`tab-${requested}`)) selectTab(requested);
}

/* ---------- triage ---------- */

let examples = [];

async function scoreExamples() {
  const policy = $("#policy").value;
  const scored = await api("/v1/triage/score", { policy, flows: examples.map((e) => e.flow) });
  $("#policy-threshold").textContent =
    `Alert when calibrated risk ≥ ${fmt(scored.threshold, 4)} · model ${scored.model_version}`;
  fillRows($("#examples-table tbody"), scored.results.map((result, index) => [
    cell(String(index + 1)),
    cell(examples[index].label === 1 ? `Attack: ${examples[index].attack_family}` : "Benign"),
    cell(fmtRisk(result.risk), "num"),
    badge(result.alert),
  ]));
}

function loadFlowIntoEditor() {
  const example = examples[Number($("#flow-source").value)];
  $("#flow-json").value = JSON.stringify(example.flow, null, 2);
  $("#flow-result").textContent = "";
}

async function scoreCustomFlow() {
  const output = $("#flow-result");
  output.className = "result";
  let flow;
  try {
    flow = JSON.parse($("#flow-json").value);
  } catch {
    output.className = "result error";
    output.textContent = "That is not valid JSON.";
    return;
  }
  try {
    const scored = await api("/v1/triage/score", { policy: $("#policy").value, flows: [flow] });
    const result = scored.results[0];
    output.textContent = `Calibrated risk ${fmtRisk(result.risk)} → ${result.alert ? "alert" : "no alert"} ` +
      `under the ${scored.policy} policy (threshold ${fmt(scored.threshold, 4)}).`;
  } catch (error) {
    output.className = "result error";
    output.textContent = error.message;
  }
}

function renderModelCard(card) {
  const v = card.validation;
  const metrics = [
    ["Validation flows", v.rows.toLocaleString("en")],
    ["PR-AUC", fmt(v.pr_auc)],
    ["Attack recall, strict policy", fmt(v.attack_recall_at_strict_threshold)],
    ["False alerts per 100k benign, strict", fmt(v.false_alerts_per_100k_benign_at_strict_threshold, 1)],
    ["Macro F1, balanced policy", fmt(v.macro_f1_at_max_macro_f1_threshold)],
    ["Model SHA-256", card.model_sha256.slice(0, 12)],
  ];
  $("#card-metrics").replaceChildren(...metrics.map(([term, value]) => {
    const group = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    group.append(dt, dd);
    return group;
  }));
  const s = card.shortcut_audit;
  $("#card-shortcut").textContent =
    `Shortcut-only control: macro F1 ${fmt(s.shortcut_only_control_macro_f1)}. ` +
    `Model without the shortcut features: ${fmt(s.shortcut_resistant_tier_macro_f1)} ` +
    `(${fmt(s.macro_f1_lost_when_shortcut_features_removed, 4)} lower). ${s.reading}`;
  $("#card-limitations").replaceChildren(...card.limitations.map((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    return li;
  }));
  $("#card-explanations").textContent = card.explanations;
  $("#card-citation").textContent = `Data: ${card.data_citation}. ${card.data_terms}`;
}

async function setupTriage() {
  const [loaded, card] = await Promise.all([api("/v1/triage/examples"), api("/v1/triage/model-card")]);
  examples = loaded;
  $("#flow-source").replaceChildren(...examples.map((example, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `#${index + 1} ${example.label === 1 ? example.attack_family : "Benign"}`;
    return option;
  }));
  loadFlowIntoEditor();
  renderModelCard(card);
  $("#policy").addEventListener("change", () => { scoreExamples(); $("#flow-result").textContent = ""; });
  $("#flow-source").addEventListener("change", loadFlowIntoEditor);
  $("#score-flow").addEventListener("click", scoreCustomFlow);
  await scoreExamples();
}

/* ---------- hybrid lab ---------- */

let datasets = [];

function describeDataset() {
  const d = datasets.find((item) => item.key === $("#lab-dataset").value);
  $("#dataset-info").textContent =
    `${d.field} · ${d.rows.toLocaleString("en")} rows, ${d.features} features, positive class: ` +
    `${d.positive_class} (${fmt(d.positive_rate * 100, 1)}%). ${d.preparation} ${d.citation} Licence: ${d.licence}`;
}

function settingsText(hyperparameters) {
  return Object.entries(hyperparameters)
    .filter(([key]) => key !== "tuned")
    .map(([key, value]) => `${key}=${typeof value === "number" ? Number(value.toPrecision(3)) : value}`)
    .join(", ") + (hyperparameters.tuned === false ? " (fixed)" : "");
}

function renderLab(report) {
  $("#lab-verdict").textContent = `${report.verdict} ${report.notice}`;
  fillRows($("#models-table tbody"), report.models.map((model) => [
    cell(model.name, model.tier === "matched quantum" ? "quantum" : ""),
    cell(model.tier),
    cell(fmt(model.cv_balanced_accuracy), "num"),
    cell(fmt(model.test.balanced_accuracy), "num"),
    cell(fmt(model.test.macro_f1), "num"),
    cell(fmt(model.test.roc_auc), "num"),
    cell(fmt(model.test.pr_auc), "num"),
    cell(settingsText(model.hyperparameters)),
    cell(fmt(model.seconds, 2), "num"),
  ]));
  const s = report.sample;
  $("#lab-sample").textContent =
    `${s.train_rows} training and ${s.test_rows} test rows; positive rate ${fmt(s.train_positive_rate * 100, 1)}% ` +
    `(train), ${fmt(s.test_positive_rate * 100, 1)}% (test). Inputs: ${s.representation}. Tuning: ${s.tuning}. ` +
    `Total ${fmt(report.elapsed_seconds, 1)} s.`;

  const q = report.quantum;
  fillRows($("#bandwidth-table tbody"), q.bandwidth_curve.map((point) => [
    cell(fmt(point.bandwidth, 2), "num"),
    cell(fmt(point.cv_balanced_accuracy), "num"),
    cell(fmt(point.offdiagonal_mean), "num"),
    cell(point.bandwidth === q.selected_bandwidth ? "◆ chosen" : ""),
  ]));
  const labels = {
    offdiagonal_mean: "Mean off-diagonal value",
    offdiagonal_std: "Std of off-diagonal values",
    kernel_target_alignment: "Kernel-target alignment",
    effective_rank: "Effective rank",
  };
  fillRows($("#diagnostics-table tbody"), Object.entries(labels).map(([key, label]) => [
    cell(label), cell(fmt(q.kernel_diagnostics[key]), "num"), cell(fmt(q.rbf_kernel_diagnostics[key]), "num"),
  ]));
  const r = q.resources;
  const resources = [
    ["Feature map", q.feature_map],
    ["Qubits", r.qubits],
    ["Feature-map depth / two-qubit gates", `${r.feature_map_depth} / ${r.feature_map_two_qubit_gates}`],
    ["Fidelity-circuit depth / two-qubit gates", `${r.fidelity_circuit_depth} / ${r.fidelity_circuit_two_qubit_gates}`],
    ["Statevectors simulated here", r.statevectors_simulated.toLocaleString("en")],
    ["Fidelity circuits real hardware would need", r.hardware_fidelity_circuits.toLocaleString("en")],
  ];
  fillRows($("#resources-table tbody"), resources.map(([label, value]) => [cell(label), cell(String(value), "num")]));
  $("#lab-results").hidden = false;
}

async function runLab(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const request = {
    dataset: form.get("dataset"),
    n_qubits: Number(form.get("n_qubits")),
    reps: Number(form.get("reps")),
    entanglement: form.get("entanglement"),
    n_train: Number(form.get("n_train")),
    seed: Number(form.get("seed")),
  };
  const status = $("#lab-status");
  const button = $("#lab-run");
  button.disabled = true;
  status.className = "status";
  status.textContent = "Running five models with cross-validated tuning… usually 2–5 seconds.";
  try {
    renderLab(await api("/v1/lab/run", request));
    status.textContent = "";
  } catch (error) {
    status.className = "status error";
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function renderReference(reference) {
  const groups = new Map();
  for (const run of reference.runs) {
    const key = `${run.dataset}|${run.qubits}`;
    if (!groups.has(key)) groups.set(key, { dataset: run.dataset, qubits: run.qubits, tuned: [], fixed: [], classical: [], control: [] });
    const group = groups.get(key);
    const quantum = run.models["Quantum-kernel SVM"].balanced_accuracy;
    if (run.variant === "fixed_pi") {
      group.fixed.push(quantum);
      continue;
    }
    group.tuned.push(quantum);
    const matched = Object.values(run.models).filter((m) => m.tier === "matched classical");
    group.classical.push(Math.max(...matched.map((m) => m.balanced_accuracy)));
    group.control.push(run.models["LightGBM (all features)"].balanced_accuracy);
  }
  const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
  const titles = Object.fromEntries(datasets.map((d) => [d.key, d.title]));
  fillRows($("#reference-table tbody"), [...groups.values()].map((g) => [
    cell(titles[g.dataset] || g.dataset), cell(String(g.qubits), "num"),
    cell(fmt(mean(g.tuned)), "num"), cell(fmt(mean(g.fixed)), "num"),
    cell(fmt(mean(g.classical)), "num"), cell(fmt(mean(g.control)), "num"),
  ]));
}

async function setupLab() {
  datasets = await api("/v1/lab/datasets");
  $("#lab-dataset").replaceChildren(...datasets.map((d) => {
    const option = document.createElement("option");
    option.value = d.key;
    option.textContent = `${d.title} — ${d.field}`;
    return option;
  }));
  describeDataset();
  $("#lab-dataset").addEventListener("change", describeDataset);
  $("#lab-qubits").addEventListener("input", (event) => { $("#qubits-value").textContent = event.target.value; });
  $("#lab-form").addEventListener("submit", runLab);
  renderReference(await api("/v1/lab/reference"));
}

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupTriage().catch((error) => { $("#policy-threshold").textContent = `Could not load triage: ${error.message}`; });
  setupLab().catch((error) => { $("#lab-status").textContent = `Could not load the lab: ${error.message}`; });
});
