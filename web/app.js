// Fitness Agents web UI. Plain JS, no build step, so it can be hosted on GitHub Pages as-is.

const API = (window.FITNESS_API_BASE || "").replace(/\/$/, "");
const state = { profile: null, summary: null, charts: {} };

// On the hosted site, each browser is its own user: a random ID kept in this browser.
// (The local server ignores it, so the web UI and the CLI share one profile there.)
const USER_ID = (() => {
  try {
    let id = localStorage.getItem("fitness-user-id");
    if (!id) localStorage.setItem("fitness-user-id", (id = crypto.randomUUID()));
    return id;
  } catch {
    return crypto.randomUUID(); // storage blocked: works for this visit only
  }
})();

// --- tiny DOM helper: text is always inserted as text, never as HTML -------------

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const $ = (sel) => document.querySelector(sel);
const fmt = (n, digits = 0) => (n == null ? "–" : Number(n).toLocaleString(undefined, { maximumFractionDigits: digits }));

async function api(path, options = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json", "X-User-Id": USER_ID },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

// --- tabs ---------------------------------------------------------------------

const loaders = { today: loadToday, progress: loadProgress, plans: loadPlans, profile: loadProfile };

function showView(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== `view-${name}`));
  loaders[name]?.().catch((e) => console.error(e));
  try { localStorage.setItem("fitness-tab", name); } catch {}
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));

// --- chat ---------------------------------------------------------------------

const messages = $("#messages");

function addMessage(kind, content) {
  const el = h("div", { class: `msg ${kind}` }, content);
  messages.append(el);
  el.scrollIntoView({ behavior: "smooth", block: "end" });
  return el;
}

function welcome() {
  messages.append(
    h("div", { class: "welcome" },
      h("h2", {}, "Talk to your fitness agents"),
      "Log what you ate, drank, weighed, ran, or lifted in plain English, or ask for a meal plan, a workout plan, or both. Plans are checked by a Reviewer agent and nothing is saved until you approve it.")
  );
}

async function send(text) {
  text = text.trim();
  if (!text) return;
  $(".welcome")?.remove();
  addMessage("user", text);
  const input = $("#composer-input");
  input.value = "";
  setBusy(true);
  const thinking = addMessage("agent thinking", "Agents working");
  const slow = setTimeout(() => (thinking.firstChild.textContent = "Still working… plan requests take 30–60 seconds"), 6000);
  try {
    const result = await api("/api/chat", { method: "POST", body: { message: text } });
    thinking.remove();
    handleResult(result);
  } catch (e) {
    thinking.remove();
    addMessage("error", e.message);
  } finally {
    clearTimeout(slow);
    setBusy(false);
    input.focus();
  }
}

function handleResult(result) {
  if (result.type === "approval") {
    const card = h("div", { class: "msg agent plan-card" });
    messages.append(card);
    renderApproval(card, result);
    card.scrollIntoView({ behavior: "smooth", block: "start" });
  } else {
    const msg = addMessage("agent", result.response || "Done.");
    if (result.trace) msg.append(traceView(result.trace));
  }
}

// The Tracker agent's run: thought -> tool calls -> observations -> answer.
function traceView(trace) {
  const steps = trace.filter((s) => s.step !== "answer");
  const answer = trace.find((s) => s.step === "answer");
  const short = (value) => {
    const text = JSON.stringify(value);
    return text.length > 600 ? text.slice(0, 600) + "…" : text;
  };
  const labels = { finish: "Answer", ask_user: "Asked you", step_limit: "Stopped at the step limit" };
  return h("details", { class: "trace" },
    h("summary", {}, `How the Tracker agent worked this out · ${steps.length} step${steps.length === 1 ? "" : "s"}`),
    h("ol", {}, steps.map((s) =>
      h("li", {},
        h("div", { class: "thought" }, h("strong", {}, "Thought "), s.thought,
          s.seconds != null ? h("span", { class: "muted" }, ` · ${s.model}, ${s.seconds} s`) : null),
        s.calls.map((c) => h("div", { class: "call" },
          h("code", {}, `${c.tool}(${JSON.stringify(c.input)})`),
          h("div", { class: "observation" }, "→ ", short(c.observation))))))),
    answer ? h("div", { class: "trace-answer" }, h("strong", {}, `${labels[answer.type] || "Answer"}: `), answer.text) : null);
}

function setBusy(busy) {
  document.querySelectorAll("#composer button, #suggestions button").forEach((b) => (b.disabled = busy));
}

$("#composer").addEventListener("submit", (e) => { e.preventDefault(); send($("#composer-input").value); });
$("#suggestions").addEventListener("click", (e) => { if (e.target.dataset.say) send(e.target.dataset.say); });

// --- plans --------------------------------------------------------------------

function mealPlanView(plan) {
  const t = plan.totals, m = plan.macro_targets;
  return h("div", { class: "plan-section" },
    h("h3", {}, "Meal plan"),
    h("div", { class: "plan-meta" },
      `${fmt(t.calories)} of ${fmt(plan.calorie_target)} kcal · protein ${t.protein_g}/${m.protein_g} g · carbs ${t.carbs_g}/${m.carbs_g} g · fat ${t.fat_g}/${m.fat_g} g`),
    plan.meals.map((meal) =>
      h("div", { class: "meal" },
        h("div", { class: "meal-head" },
          h("span", {}, meal.name),
          h("span", { class: "muted" }, `${fmt(meal.items.reduce((s, i) => s + i.calories, 0))} kcal`)),
        h("ul", {}, meal.items.map((i) => h("li", {}, `${i.amount} ${i.food} · ${fmt(i.calories)} kcal, ${fmt(i.protein_g)} g protein`))),
        meal.swap_options?.length ? h("div", { class: "swaps" }, "Swaps: " + meal.swap_options.join(" · ")) : null)),
    plan.notes ? h("div", { class: "swaps" }, plan.notes) : null
  );
}

function workoutPlanView(plan) {
  const weak = plan.weak_points?.length ? plan.weak_points.join(", ") : "none identified";
  return h("div", { class: "plan-section" },
    h("h3", {}, "Workout plan"),
    h("div", { class: "plan-meta" }, `Weak points: ${weak}. ${plan.weak_point_strategy}`),
    h("div", { class: "days" },
      plan.days.map((d) =>
        d.exercises.length
          ? h("div", { class: "day" },
              h("strong", {}, d.day), h("div", { class: "focus" }, d.focus),
              h("ul", {}, d.exercises.map((x) =>
                h("li", {}, `${x.name}: ${x.sets} × ${x.reps}${x.weight_lb ? ` @ ${fmt(x.weight_lb)} lb` : ""}`))))
          : h("div", { class: "day rest" }, h("strong", {}, d.day), "Rest"))),
    plan.notes ? h("div", { class: "swaps" }, plan.notes) : null
  );
}

function renderApproval(card, pending) {
  card.replaceChildren();
  if (pending.meal_plan) card.append(mealPlanView(pending.meal_plan));
  if (pending.workout_plan) card.append(workoutPlanView(pending.workout_plan));

  const flags = Object.values(pending.unresolved_issues || {}).flat();
  if (flags.length) {
    card.append(h("div", { class: "flags" }, h("strong", {}, "The Reviewer still flagged:"), h("ul", {}, flags.map((f) => h("li", {}, f)))));
  }

  const kinds = ["meal", "workout"].filter((k) => pending[`${k}_plan`]);
  const bar = h("div", { class: "approval" });
  const decide = async (body, label) => {
    bar.querySelectorAll("button, textarea, input").forEach((el) => (el.disabled = true));
    const status = h("span", { class: "decided" }, label);
    bar.append(status);
    try {
      const result = await api("/api/approve", { method: "POST", body: { thread_id: pending.thread_id, ...body } });
      bar.replaceChildren(h("span", { class: "decided" }, { approve: "Approved", reject: "Rejected", edit: "Sent back for changes" }[body.action]));
      handleResult(result);
    } catch (e) {
      status.remove();
      bar.querySelectorAll("button, textarea, input").forEach((el) => (el.disabled = false));
      addMessage("error", e.message);
    }
  };

  const approveBtn = h("button", { class: "primary", onclick: () => decide({ action: "approve" }, "Saving…") }, "Approve");
  const rejectBtn = h("button", { class: "secondary", onclick: () => decide({ action: "reject" }, "Discarding…") }, "Reject");
  const editBtn = h("button", { class: "secondary", onclick: () => {
    editBtn.remove();
    const textarea = h("textarea", { placeholder: "What should change? e.g. no oatmeal, more fish, only 3 gym days", "aria-label": "Requested changes" });
    const boxes = kinds.map((k) => h("label", {}, h("input", { type: "checkbox", value: k, checked: true }), ` ${k} plan`));
    const sendBtn = h("button", { class: "primary", onclick: () => {
      const targets = boxes.map((b) => b.firstChild).filter((c) => c.checked).map((c) => c.value);
      if (!textarea.value.trim() || !targets.length) return textarea.focus();
      decide({ action: "edit", targets, feedback: textarea.value }, "Redrafting… (30–60 seconds)");
    } }, "Send changes");
    bar.append(h("div", { class: "edit-box" }, textarea, kinds.length > 1 ? h("div", { class: "edit-targets" }, boxes) : null, h("div", {}, sendBtn)));
    textarea.focus();
  } }, "Edit");
  bar.append(approveBtn, editBtn, rejectBtn);
  card.append(bar);
}

async function loadPlans() {
  const { meal_plan, workout_plan } = await api("/api/plans");
  const box = $("#plans");
  if (!meal_plan && !workout_plan) {
    box.replaceChildren(h("div", { class: "card empty" }, "No approved plans yet. Ask for one in Chat, then approve it."));
    return;
  }
  box.replaceChildren(h("div", { class: "card plan-card" },
    meal_plan ? mealPlanView(meal_plan) : null,
    workout_plan ? workoutPlanView(workout_plan) : null));
}

// --- today --------------------------------------------------------------------

function tile(label, value, unit, sub, ratio) {
  const meter = ratio == null ? null
    : h("div", { class: `meter${ratio > 1.1 ? " over" : ""}`, role: "img", "aria-label": `${Math.round(ratio * 100)}% of target` },
        h("span", { style: `width:${Math.min(ratio, 1) * 100}%` }));
  return h("div", { class: "tile" },
    h("div", { class: "label" }, label),
    h("div", { class: "value" }, value, unit ? h("small", {}, ` ${unit}`) : null),
    sub ? h("div", { class: "sub" }, sub) : null,
    meter);
}

async function loadToday() {
  const { date, totals, entries, summary } = await api("/api/today");
  $("#today-date").textContent = new Date(date + "T12:00").toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric" });
  const tiles = $("#today-tiles");
  if (summary) {
    const target = summary.calories.target, m = summary.macros;
    const left = target - totals.calories;
    tiles.replaceChildren(
      tile("Calories", fmt(totals.calories), `/ ${fmt(target)}`, left >= 0 ? `${fmt(left)} left` : `${fmt(-left)} over`, totals.calories / target),
      tile("Protein", fmt(totals.protein_g), `/ ${m.protein_g} g`, null, totals.protein_g / m.protein_g),
      tile("Carbs", fmt(totals.carbs_g), `/ ${m.carbs_g} g`, null, totals.carbs_g / m.carbs_g),
      tile("Fat", fmt(totals.fat_g), `/ ${m.fat_g} g`, null, totals.fat_g / m.fat_g),
      tile("Water", fmt(totals.water_oz), `/ ${summary.water_goal_oz} oz`, "suggested goal", totals.water_oz / summary.water_goal_oz));
  } else {
    tiles.replaceChildren(h("p", { class: "notice" }, "Set up your profile to see your daily targets."));
  }

  $("#today-foods").replaceChildren(entries.foods.length
    ? h("ul", { class: "rows" }, entries.foods.map((f) =>
        h("li", {},
          h("span", {}, f.food, h("span", { class: `badge ${f.source}` }, f.source === "usda" ? "USDA" : "estimated")),
          h("span", { class: "right" }, `${fmt(f.calories)} kcal · ${fmt(f.protein_g)} g protein`))))
    : h("p", { class: "empty" }, "Nothing logged yet. Tell the agents what you ate in Chat."));

  $("#today-activities").replaceChildren(entries.activities.length
    ? h("ul", { class: "rows" }, entries.activities.map((a) =>
        h("li", {}, h("span", {}, a.description), h("span", { class: "right" }, activityDetail(a)))))
    : h("p", { class: "empty" }, "No runs or lifts logged today."));
}

function activityDetail(a) {
  if (a.kind === "lift" && a.weight_lb) return `${a.sets || 1} × ${a.reps} @ ${fmt(a.weight_lb)} lb`;
  const parts = [];
  if (a.distance_mi) parts.push(`${fmt(a.distance_mi, 2)} mi`);
  if (a.duration_min) parts.push(`${fmt(a.duration_min)} min`);
  if (a.distance_mi && a.duration_min) parts.push(pace(a.duration_min / a.distance_mi) + " /mi");
  return parts.join(" · ");
}
const pace = (minPerMile) => `${Math.floor(minPerMile)}:${String(Math.round((minPerMile % 1) * 60)).padStart(2, "0")}`;

// --- progress (charts) --------------------------------------------------------

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const shortDate = (d) => new Date(d + "T12:00").toLocaleDateString(undefined, { month: "short", day: "numeric" });

function chartTheme() {
  return {
    series: css("--series-1"), reference: css("--reference"), grid: css("--grid"),
    text: css("--text-2"), surface: css("--surface"), font: getComputedStyle(document.body).fontFamily,
  };
}

function baseOptions(t, unit) {
  return {
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false }, // single series: the card title names it
      tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt(c.parsed.y, 1)} ${unit}` } },
    },
    scales: {
      x: { grid: { display: false }, border: { color: t.grid }, ticks: { color: t.text, font: { family: t.font }, maxRotation: 0, autoSkipPadding: 16 } },
      y: { grid: { color: t.grid, lineWidth: 1 }, border: { display: false }, ticks: { color: t.text, font: { family: t.font } } },
    },
  };
}

function table(headers, rows) {
  return h("div", { class: "table-scroll" }, h("table", {},
    h("thead", {}, h("tr", {}, headers.map((x, i) => h("th", { class: i ? "num" : null }, x)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, r.map((x, i) => h("td", { class: i ? "num" : null }, x)))))));
}

function drawChart(key, canvas, config) {
  state.charts[key]?.destroy();
  state.charts[key] = new Chart(canvas, config);
}

let lastProgress = null;

async function loadProgress() {
  if (!window.Chart) return setTimeout(loadProgress, 100); // CDN script still loading
  let data;
  try {
    data = await api("/api/progress");
  } catch (e) {
    $("#progress-tiles").replaceChildren(h("p", { class: "notice" }, e.message));
    return;
  }
  lastProgress = data;
  renderProgress(data);
}

function renderProgress(data) {
  const t = chartTheme();
  const latest = data.weights.at(-1);
  const maxes = Object.entries(data.max_lifts).filter(([, v]) => v);
  const miles = data.runs.reduce((s, r) => s + (r.distance_mi || 0), 0);
  $("#progress-tiles").replaceChildren(
    tile("Weight", fmt(data.current_weight_lb, 1), "lb", `goal ${fmt(data.goal_weight_lb)} lb`),
    tile("BMI", fmt(data.current_bmi, 1), null, latest ? `as of ${shortDate(latest.day)}` : "from your profile"),
    tile("Runs", data.runs.length, null, `${fmt(miles, 1)} miles`),
    ...maxes.map(([lift, v]) => tile(`${{ ohp: "Overhead press" }[lift] || lift[0].toUpperCase() + lift.slice(1)} max`, fmt(v), "lb", "estimated 1RM")));

  // Body weight: one line plus the goal as a thin reference line.
  const hasWeights = data.weights.length > 0;
  $("#weight-empty").hidden = hasWeights;
  $("#weight-chart").parentElement.hidden = !hasWeights;
  $("#weight-subtitle").textContent = `· goal ${fmt(data.goal_weight_lb)} lb shown as gray line`;
  if (hasWeights) {
    const labels = data.weights.map((w) => shortDate(w.day));
    drawChart("weight", $("#weight-chart"), {
      type: "line",
      data: { labels, datasets: [
        { label: "Weight", data: data.weights.map((w) => w.weight_lb), borderColor: t.series, backgroundColor: t.series,
          borderWidth: 2, pointRadius: 4, pointHoverRadius: 6, pointBorderColor: t.surface, pointBorderWidth: 2, tension: 0.25 },
        { label: "Goal", data: labels.map(() => data.goal_weight_lb), borderColor: t.reference, borderWidth: 1, pointRadius: 0, pointHoverRadius: 0 },
      ] },
      options: baseOptions(t, "lb"),
    });
  }
  $("#weight-table").replaceChildren(table(["Date", "Weight (lb)", "BMI"], data.weights.map((w) => [w.day, fmt(w.weight_lb, 1), fmt(w.bmi, 1)])));

  // Calories: one bar per logged day plus the target as a reference line.
  const hasCalories = data.calories.length > 0;
  $("#calorie-empty").hidden = hasCalories;
  $("#calorie-chart").parentElement.hidden = !hasCalories;
  $("#calorie-subtitle").textContent = `· target ${fmt(data.calorie_target)} kcal shown as gray line`;
  if (hasCalories) {
    const labels = data.calories.map((c) => shortDate(c.day));
    drawChart("calories", $("#calorie-chart"), {
      data: { labels, datasets: [
        { type: "bar", label: "Eaten", data: data.calories.map((c) => c.total), backgroundColor: t.series,
          borderRadius: { topLeft: 4, topRight: 4 }, borderSkipped: "bottom", maxBarThickness: 28, categoryPercentage: 0.8 },
        { type: "line", label: "Target", data: labels.map(() => data.calorie_target), borderColor: t.reference, borderWidth: 1, pointRadius: 0, pointHoverRadius: 0 },
      ] },
      options: { ...baseOptions(t, "kcal"), scales: { ...baseOptions(t).scales, y: { ...baseOptions(t).scales.y, beginAtZero: true } } },
    });
  }
  $("#calorie-table").replaceChildren(table(["Date", "Calories", "vs target"], data.calories.map((c) =>
    [c.day, fmt(c.total), `${c.total >= data.calorie_target ? "+" : ""}${fmt(c.total - data.calorie_target)}`])));

  $("#runs").replaceChildren(data.runs.length
    ? table(["Date", "Miles", "Minutes", "Pace /mi"], data.runs.map((r) =>
        [r.day, fmt(r.distance_mi, 2), fmt(r.duration_min), r.distance_mi && r.duration_min ? pace(r.duration_min / r.distance_mi) : "–"]))
    : h("p", { class: "empty" }, "No runs logged yet. Try \"ran 2 miles in 18 minutes\" in Chat."));
}

// Charts read colors at draw time, so redraw when the OS theme flips.
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (lastProgress) renderProgress(lastProgress); });

// --- profile ------------------------------------------------------------------

const form = $("#profile-form");

async function loadProfile() {
  const { profile, summary } = await api("/api/profile");
  state.profile = profile;
  $("#profile-intro").hidden = !!profile;
  renderProfileTiles(summary);
  if (!profile) return;
  const set = (name, value) => { if (form.elements[name] && value != null) form.elements[name].value = value; };
  set("height_ft", Math.floor(profile.height_in / 12));
  set("height_in_rem", Math.round(profile.height_in % 12));
  for (const k of ["weight_lb", "goal_weight_lb", "weekly_rate_lb", "age", "sex", "activity", "food_likes", "food_avoid", "training_days", "equipment", "avoid_exercises"]) set(k, profile[k]);
  for (const [lift, v] of Object.entries(profile.max_lifts || {})) set(`lift_${lift}`, v ? Math.round(v) : "");
}

function renderProfileTiles(summary) {
  $("#profile-tiles").replaceChildren(...(summary ? [
    tile("BMI", fmt(summary.bmi, 1), null, summary.bmi_category),
    tile("Maintenance", fmt(summary.calories.maintenance), "kcal"),
    tile("Daily target", fmt(summary.calories.target), "kcal",
      `${summary.calories.weekly_rate_lb > 0 ? "+" : ""}${summary.calories.weekly_rate_lb} lb/week${summary.calories.floor_applied ? " (raised to safe minimum)" : ""}`),
    tile("Protein target", summary.macros.protein_g, "g"),
  ] : []));
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = form.elements, num = (n) => Number(f[n].value);
  const body = {
    height_in: num("height_ft") * 12 + num("height_in_rem"),
    weight_lb: num("weight_lb"), goal_weight_lb: num("goal_weight_lb"), weekly_rate_lb: num("weekly_rate_lb") || 0,
    age: num("age"), sex: f.sex.value, activity: f.activity.value,
    food_likes: f.food_likes.value, food_avoid: f.food_avoid.value,
    training_days: num("training_days") || 3, equipment: f.equipment.value, avoid_exercises: f.avoid_exercises.value,
    max_lifts: Object.fromEntries(["bench", "squat", "deadlift", "ohp"].map((l) => [l, f[`lift_${l}`].value])),
  };
  const status = $("#profile-status");
  status.textContent = "Saving…";
  try {
    const { summary } = await api("/api/profile", { method: "PUT", body });
    status.textContent = "Saved.";
    renderProfileTiles(summary);
    $("#profile-intro").hidden = true;
    setTimeout(() => (status.textContent = ""), 2500);
  } catch (err) {
    status.textContent = err.message;
  }
});

// --- start --------------------------------------------------------------------

(async function init() {
  welcome();
  // Free hosting puts the server to sleep when idle; waking it takes about a minute.
  const waking = setTimeout(() => {
    $("#model-status").textContent = "waking server…";
    addMessage("agent thinking", "Waking up the server (free hosting sleeps when idle, this takes about a minute)");
  }, 2500);
  try {
    const { provider, model, multi_user } = await api("/api/status");
    clearTimeout(waking);
    $(".msg.thinking")?.remove();
    $("#model-status").textContent = `${model} · ${provider}`;
    $("#hosted-note").hidden = !multi_user;
    const { profile } = await api("/api/profile");
    let tab = "chat";
    try { tab = localStorage.getItem("fitness-tab") || "chat"; } catch {}
    showView(profile ? tab : "profile");
  } catch {
    clearTimeout(waking);
    $(".msg.thinking")?.remove();
    $("#model-status").textContent = "API offline";
    addMessage("error", API
      ? "Can't reach the agents' server right now. It may still be waking up; refresh in a minute."
      : "Can't reach the agents' server. Start it with: uv run fitness-web");
  }
})();
