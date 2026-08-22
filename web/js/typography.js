/* Reading comfort: typeface, size, leading, measure, justification and the presets. */

import { $ } from "./dom.js";
import { prefs } from "./prefs.js";
import { drawSearchGraph } from "./search.js";

/* Top of the `line width` range means "no cap — follow the pane" rather than 100 characters.
 * Kept as a slider position rather than a separate checkbox so there is one control for one
 * decision, and so dragging left from `full width` is a continuous gesture. */
const MEASURE_FULL = 100;

/* One-time reset. Filling the pane is the new default, but anyone who used a previous build
 * has a fixed measure in localStorage that would silently override it — the change would
 * look like it had not shipped. Runs once, and only touches this one key. */
(function migrateMeasureToFull() {
  if (prefs.get("measureFull", "") === "1") return;
  prefs.set("measure", String(MEASURE_FULL));
  prefs.set("measureFull", "1");
})();

/* Presets bundle theme, face, size, leading and measure, because those five interact:
 * a serif at 17px wants more leading and a wider column than a 14px sans, and setting one
 * without the others usually makes reading worse rather than better. Touching any
 * individual control switches the preset to "custom" instead of silently disagreeing with
 * the label. */

const FONTS = {
  system: "var(--font-system)", inter: "var(--font-inter)",
  charter: "var(--font-charter)", literata: "var(--font-literata)",
  times: "var(--font-times)", mono: "var(--font-mono)",
  wide: "var(--font-wide)", atkinson: "var(--font-atkinson)",
};

/* Every preset now fills the pane. A preset that pinned its own column width would undo
 * the pane-following behaviour the moment someone picked one, which reads as the setting
 * randomly reverting. Narrowing stays available on the slider, independent of preset. */
const PRESETS = {
  default:  { theme: "auto",     font: "system",   size: 16, leading: 1.60, measure: MEASURE_FULL, justify: false },
  // Warm page, serif — closest to a printed journal.
  paper:    { theme: "light",    font: "charter",  size: 17, leading: 1.65, measure: MEASURE_FULL, justify: true  },
  night:    { theme: "dark",     font: "system",   size: 16, leading: 1.72, measure: MEASURE_FULL, justify: false },
  sepia:    { theme: "sepia",    font: "literata", size: 17, leading: 1.70, measure: MEASURE_FULL, justify: true  },
  // More text per screen for skimming, at the cost of comfort over long sessions.
  compact:  { theme: "auto",     font: "system",   size: 14, leading: 1.45, measure: MEASURE_FULL, justify: false },
  // Accessibility: Atkinson Hyperlegible was designed to disambiguate similar glyphs.
  contrast: { theme: "contrast", font: "atkinson", size: 19, leading: 1.85, measure: MEASURE_FULL, justify: false },
};

/* Sepia out of the box. The individual defaults are read from the preset rather than
 * repeated as literals: a fresh browser has no stored prefs, so `currentStyle` is what
 * actually decides the first paint, and a preset name that disagreed with the values
 * beside it would show "Sepia" in the picker over a completely different page. */
const DEFAULT_PRESET = "sepia";
const D = PRESETS[DEFAULT_PRESET];

function currentStyle() {
  return {
    theme:   prefs.get("theme", D.theme),
    font:    prefs.get("font", D.font),
    size:    Number(prefs.get("fontsize", String(D.size))),
    leading: Number(prefs.get("leading", String(D.leading))),
    measure: Number(prefs.get("measure", String(D.measure))),
    justify: prefs.get("justify", D.justify ? "1" : "0") === "1",
  };
}

export function applyTypography() {
  const st = currentStyle();
  const root = document.documentElement;
  if (st.theme === "auto") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", st.theme);

  root.style.setProperty("--paper-font", FONTS[st.font] || FONTS.system);
  root.style.setProperty("--paper-size", st.size + "px");
  root.style.setProperty("--paper-leading", String(st.leading));
  // Measure is in characters, converted at roughly half the font size per character —
  // the conventional approximation for average glyph advance in running text. At the top
  // of the range it resolves to `none` instead, which is what lets the column follow the
  // pane rather than sitting at a fixed width with dead space beside it.
  root.style.setProperty(
    "--measure",
    st.measure >= MEASURE_FULL ? "none" : Math.round(st.size * st.measure * 0.5) + "px",
  );
  root.style.setProperty("--paper-align", st.justify ? "justify" : "start");
  root.style.setProperty("--paper-hyphens", st.justify ? "auto" : "manual");

  $("#theme").value = st.theme;
  $("#font").value = st.font;
  $("#fontsize").value = String(st.size);
  $("#fontsize-val").textContent = String(st.size);
  $("#leading").value = String(st.leading);
  $("#leading-val").textContent = st.leading.toFixed(2);
  $("#measure").value = String(st.measure);
  $("#measure-val").textContent =
    st.measure >= MEASURE_FULL ? "full width" : `${st.measure} chars`;
  $("#justify").checked = st.justify;
  $("#preset").value = prefs.get("preset", DEFAULT_PRESET);
  drawSearchGraph();
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  prefs.set("preset", name);
  prefs.set("theme", p.theme);
  prefs.set("font", p.font);
  prefs.set("fontsize", String(p.size));
  prefs.set("leading", String(p.leading));
  prefs.set("measure", String(p.measure));
  prefs.set("justify", p.justify ? "1" : "0");
  applyTypography();
}

$("#preset").addEventListener("change", (e) => {
  if (e.target.value === "custom") { prefs.set("preset", "custom"); return; }
  applyPreset(e.target.value);
});

// Any manual change means the active preset no longer describes the state.
function markCustom() {
  prefs.set("preset", "custom");
  $("#preset").value = "custom";
}

$("#theme").addEventListener("change", (e) => { prefs.set("theme", e.target.value); markCustom(); applyTypography(); });
$("#font").addEventListener("change", (e) => { prefs.set("font", e.target.value); markCustom(); applyTypography(); });
$("#fontsize").addEventListener("input", (e) => { prefs.set("fontsize", e.target.value); markCustom(); applyTypography(); });
$("#leading").addEventListener("input", (e) => { prefs.set("leading", e.target.value); markCustom(); applyTypography(); });
$("#measure").addEventListener("input", (e) => { prefs.set("measure", e.target.value); markCustom(); applyTypography(); });
$("#justify").addEventListener("change", (e) => { prefs.set("justify", e.target.checked ? "1" : "0"); markCustom(); applyTypography(); });
