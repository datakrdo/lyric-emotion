import * as d3 from "d3";
import "./style.css";
import type { Album, Bundle, Song, Validation } from "./types";
import { colorOf, mountDisc, word, type Disc } from "./vinyl";

const REPO = "https://github.com/datakrdo/lyric-emotion";
const base = import.meta.env.BASE_URL;
const pct = (x: number) => Math.round(x * 100);
const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};

const [bundle, validation]: [Bundle, Validation] = await Promise.all([
  fetch(`${base}data/bundle.json`).then((r) => r.json()),
  fetch(`${base}data/validation.json`).then((r) => r.json()),
]);
const { findings: f, albums, songs } = bundle;
const [A, B] = f.bands;
const COLOR: Record<string, string> = { [A]: "#b583ff", [B]: "#ffb454" };
const win = f.window;
const span = `${win.start_year}–${win.end_year}`;

// The fingerprint's headline only holds while the data says so; otherwise fall back to the
// pipeline's own verdict sentence.
const plain =
  !f.emotions_compared[0].significant &&
  f.bands.every((b) => {
    const p = f.profile[b];
    return (
      p.top[0].emotion === "annoyance" &&
      p.top.slice(0, 4).some((r) => r.emotion === "disappointment") &&
      p.sadness_rank >= 4
    );
  });
const headline = plain ? "Annoyed. Disappointed. <em>Not sad.</em>" : f.verdict;

const bandSongs = (band: string) => songs.filter((s) => s.artist === band);
const typical = (band: string, emotion: string) =>
  median(bandSongs(band).map((s) => s[emotion] as number)) * 100;
const cover = (s: Song) => s.cover_url;

const esc = (t: string) => t.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);

// ---------- page skeleton ----------
document.querySelector<HTMLElement>("#app")!.innerHTML = `
<section class="hero" id="top">
  <div class="kick">${span} · ${win.n_songs} songs · two bands</div>
  <h1>${headline}</h1>
  <p class="lede">What a language model hears in the lyrics of the two “saddest” bands of the 80s.
  Each record is a band’s typical song: the wider the slice, the stronger the emotion.
  Touch a slice.</p>
  <div class="discs">${f.bands.map(deck).join("")}</div>
  <p class="hint">Hover a slice to stop the record · click it to compare · ★ marks what is the band’s own</p>
  <div class="panel" id="panel" aria-live="polite"></div>
</section>
<section class="reveal" id="notsad">${notSad()}</section>
<section class="reveal" id="songs"><h2>Every song, one dot</h2>
  <p class="lede">${win.n_songs} songs placed by how bright they read (left to right) and by the
  overall mix of emotions that best tells the bands apart (top to bottom).</p>
  <div id="scatter"></div></section>
<section class="reveal" id="albums">${albumSection()}</section>
<section class="reveal" id="sure">${sureSection()}</section>
<section class="reveal" id="made">${madeSection()}</section>
<footer>Built by datakrdo · Lyrics are never stored or shown here, only scores. Album covers are
linked from Genius, not hosted. Data: GoEmotions (Apache-2.0), EmoBank (CC BY-SA 4.0),
MoodyLyrics, Deezer DMDD. <a href="${REPO}">Source</a></footer>
<div class="tip" id="tip"></div>`;

function deck(band: string): string {
  const arm = `<svg class="arm" viewBox="0 0 100 100" aria-hidden="true"><path d="M92 8 L40 60" stroke="#c9c9d6" stroke-width="3.5" stroke-linecap="round"/><circle cx="92" cy="8" r="6" fill="#2a2a36" stroke="#c9c9d6"/><rect x="30" y="56" width="16" height="9" rx="2" transform="rotate(45 38 60)" fill="#c9c9d6"/></svg>`;
  const items = f.profile[band].top
    .map(
      (r) => `<li><button data-e="${r.emotion}" aria-pressed="false" class="${r.distinct ? "own" : ""}">
      <span><span class="dot" style="background:${colorOf(r.emotion)}"></span>${word(r.emotion)}${r.distinct ? " ★" : ""}</span>
      <span>${pct(r.score)}</span></button></li>`,
    )
    .join("");
  return `<div class="deck" data-band="${esc(band)}"><div class="disc-wrap">${arm}</div>
    <h3 style="color:${COLOR[band]}">${band}</h3><ul class="legend">${items}</ul></div>`;
}

// ---------- interaction: discs + legend + panel ----------
const tip = document.querySelector<HTMLElement>("#tip")!;
const discs: Disc[] = [];
let selected: string | null = null;

function choose(emotion: string | null) {
  selected = selected === emotion ? null : emotion;
  discs.forEach((d) => d.select(selected));
  document.querySelectorAll<HTMLButtonElement>(".legend button").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.e === selected)),
  );
  renderPanel();
}

function hover(band: string, emotion: string | null, e?: PointerEvent) {
  if (!emotion) return void (tip.style.opacity = "0");
  const r = f.profile[band].top.find((t) => t.emotion === emotion)!;
  tip.textContent = `${band} · ${word(emotion)} ${pct(r.score)}/100`;
  tip.style.opacity = "1";
  if (e) {
    tip.style.left = `${Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8)}px`;
    tip.style.top = `${e.clientY + 14}px`;
  }
}

document.querySelectorAll<HTMLElement>(".deck").forEach((el) => {
  const band = el.dataset.band!;
  discs.push(
    mountDisc(el.querySelector(".disc-wrap")!, band, COLOR[band], f.profile[band].top, {
      hover: (emotion, e) => hover(band, emotion, e),
      pick: choose,
    }),
  );
  el.querySelectorAll<HTMLButtonElement>(".legend button").forEach((b) =>
    b.addEventListener("click", () => choose(b.dataset.e!)),
  );
});
// The SVG was appended after the tonearm: make sure the record sits under it.
document.querySelectorAll(".disc-wrap").forEach((w) => {
  const svg = w.querySelector(":scope > svg:not(.arm)");
  if (svg) w.insertBefore(svg, w.firstChild);
});

function renderPanel() {
  const panel = document.querySelector<HTMLElement>("#panel")!;
  if (!selected) {
    panel.innerHTML = `<p class="verdict-chip" style="margin:0">Pick an emotion to see both bands side by side and the songs that carry it most.</p>`;
    return;
  }
  const e = selected;
  const cmp = f.emotions_compared.find((r) => r.emotion === e);
  const note = cmp?.significant
    ? `<b>${cmp.higher}</b>’s typical song scores higher, and it holds up even counting whole albums.`
    : cmp
      ? "No difference between the bands."
      : "Not a clear difference between the bands.";
  const col = (band: string) => {
    const best = [...bandSongs(band)].sort((a, b) => (b[e] as number) - (a[e] as number)).slice(0, 3);
    return `<div><div class="band" style="color:${COLOR[band]}">${band}</div>
      <div class="big">${Math.round(typical(band, e))}<small> / 100 typical song</small></div>
      <ul class="songs">${best
        .map(
          (s) => `<li>${cover(s) ? `<img src="${cover(s)}" alt="" loading="lazy" referrerpolicy="no-referrer">` : `<span class="ph" aria-hidden="true">♪</span>`}
          <div><b>${esc(s.title)}</b><span>${esc(s.album)} · ${pct(s[e] as number)}</span></div></li>`,
        )
        .join("")}</ul></div>`;
  };
  panel.innerHTML = `<h3 style="color:${colorOf(e)}">${word(e)}</h3>
    <div class="vs">${col(A)}${col(B)}</div><p class="verdict-chip">${note}</p>`;
}
renderPanel();
choose("joy"); // open on the emotion that sets The Cure apart

// ---------- section 2: not sad ----------
function notSad(): string {
  const rank = (b: string) => f.profile[b].sadness_rank;
  const rows = f.emotions_compared
    .map((r) => {
      const bar = (band: string, v: number) =>
        `<div class="bar"><span class="name" style="color:${COLOR[band]}">${band}</span>
        <span class="track"><span class="fill" style="--w:${v * 100}%;background:${COLOR[band]}"></span></span>
        <span class="val">${Math.round(v * 100)}</span></div>`;
      return `<div class="row"><h4>${word(r.emotion)} <span>${r.significant ? `· higher in ${r.higher}` : "· no difference"}</span></h4>
        ${bar(A, r.median_a)}${bar(B, r.median_b)}</div>`;
    })
    .join("");
  const q = f.question_line_share;
  return `<h2>Neither band is the sad one</h2>
  <p class="lede">Sadness ranks #${rank(A)} among the emotions the model finds in ${A}, and #${rank(B)} in ${B}.
  The two are tied. What does separate them is a handful of other emotions (0–100, typical song):</p>
  <div class="bars">${rows}</div>
  <p class="lede">A fact you can check: ${(q[B] * 100).toFixed(1)}% of ${B}’s lines end with “?”, against ${(q[A] * 100).toFixed(1)}% for ${A}.
  A song can be sad and joyful at once, so “equally sad” and “more joyful” don’t contradict each other.</p>`;
}

// ---------- section 3: scatter ----------
function scatter() {
  const W = 960, H = 520, m = { t: 20, r: 20, b: 56, l: 56 };
  const svg = d3.select("#scatter").append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const x = d3.scaleLinear().domain(d3.extent(songs, (s) => s.valence_z) as [number, number]).nice().range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain(d3.extent(songs, (s) => s.style) as [number, number]).nice().range([H - m.b, m.t]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${y(0)})`).call(d3.axisBottom(x).ticks(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${x(0)},0)`).call(d3.axisLeft(y).ticks(0));
  const poles = f.poles;
  svg.append("text").attr("class", "axis").attr("x", W / 2).attr("y", H - 14).attr("text-anchor", "middle")
    .attr("fill", "#a2a2b2").attr("font-size", 13).text("← darker                    brighter →");
  svg.append("text").attr("x", m.l + 8).attr("y", m.t + 14).attr("fill", COLOR[A]).attr("font-size", 13)
    .text(`▲ more ${poles[A].slice(0, 3).join(", ")}`);
  svg.append("text").attr("x", m.l + 8).attr("y", H - m.b - 8).attr("fill", COLOR[B]).attr("font-size", 13)
    .text(`▼ more ${poles[B].slice(0, 3).join(", ")}`);
  svg.append("g").selectAll("circle").data(songs).join("circle")
    .attr("cx", (s) => x(s.valence_z)).attr("cy", (s) => y(s.style)).attr("r", 0)
    .attr("fill", (s) => COLOR[s.artist]).attr("fill-opacity", 0.8).attr("stroke", "#0d0d12")
    .on("pointermove", (e: PointerEvent, s) => {
      tip.textContent = `${s.title} · ${s.album} (${s.year})`;
      tip.style.opacity = "1";
      tip.style.left = `${Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8)}px`;
      tip.style.top = `${e.clientY + 14}px`;
    })
    .on("pointerleave", () => (tip.style.opacity = "0"))
    .transition().duration(900).delay((_, i) => i * 6).attr("r", 6);
}

// ---------- section 4: albums ----------
function albumCard(a: Album): string {
  return `<div class="album" style="--c:${COLOR[a.artist]}"><img src="${a.cover_url}" alt="${esc(a.album)} cover" loading="lazy" referrerpolicy="no-referrer">
    <div><b>${esc(a.album)}</b><small class="who">${a.artist} · ${a.year} · ${a.n} songs</small>
    <div class="meter"><i style="left:${a.valence_pct}%"></i></div>
    <div class="meter-l"><span>darker</span><span>brighter</span></div>
    <small>Stands out for: ${a.stands_out}</small></div></div>`;
}

function albumSection(): string {
  const studio = albums
    .filter((a: Album) => a.kind === "studio" && a.year >= win.start_year && a.year <= win.end_year)
    .sort((a, b) => a.year - b.year);
  const years = [...new Set(studio.map((a) => a.year))];
  const cell = (y: number, band: string) =>
    studio.filter((a) => a.year === y && a.artist === band).map(albumCard).join("");
  const timeline = years
    .map(
      (y) => `<li class="tl-row reveal"><div class="tl-side l">${cell(y, A)}</div>
      <span class="tl-year">${y}</span><div class="tl-side r">${cell(y, B)}</div></li>`,
    )
    .join("");
  const col = (band: string) =>
    `<div><h3 style="color:${COLOR[band]}">${band}</h3>${studio.filter((a) => a.artist === band).map(albumCard).join("")}</div>`;
  return `<h2>Album by album</h2><p class="lede">Each marker shows where the album’s brightness sits
  against all other albums of both bands.</p>
  <div class="seg" role="group" aria-label="Album view">
    <button data-view="tl" aria-pressed="true">Timeline</button><button data-view="bands" aria-pressed="false">By band</button></div>
  <ol class="tl" id="alb-tl">${timeline}</ol>
  <div class="cols" id="alb-bands" hidden>${f.bands.map(col).join("")}</div>`;
}

// ---------- section 5: how sure ----------
function sureSection(): string {
  const v = f.valence, s = f.style;
  const lo = (1 + v.ci_low) / 2, mid = (1 + v.delta) / 2, hi = (1 + v.ci_high) / 2;
  const rows: [string, string, string, string][] = [
    ["How dark", v.status === "none_detected" ? "none" : "clear", v.status === "none_detected" ? "No clear difference" : "Difference found",
      `Take one song from each band: the brighter one is ${A}’s ${pct(mid)}% of the time in this sample (plausible range ${pct(lo)}%–${pct(hi)}%). A perfect tie is 50%.`],
    ["Overall feel", s.status, s.status === "clear" ? "Clear" : s.status === "suggestive" ? "Suggestive" : "Not detected",
      `Given one song’s emotional profile, a model names the right band about ${pct(s.auc)}% of the time (guessing gives 50%), tested on albums it had not seen.`],
    ["Single emotions", f.emotions.status, f.emotions.status === "clear" ? "Clear" : f.emotions.status === "suggestive" ? "Suggestive" : "Not detected",
      "Counting whole albums rather than songs, since songs on one album sound alike."],
    ["Human check", validation.status === "pending" ? "pending" : "clear", validation.status === "pending" ? "Pending" : "Checked",
      validation.status === "pending" ? "Not yet checked against ratings from a human reader: the 80-song gold set is being annotated." : (validation.summary ?? "")],
  ];
  return `<h2>How sure are we?</h2><div class="badges">${rows
    .map(([h, cls, label, text]) => `<div class="badge"><div><b>${h}</b><br><span class="pill ${cls}">${label}</span></div><div>${text}</div></div>`)
    .join("")}</div>`;
}

// ---------- section 6: how it was made ----------
function madeSection(): string {
  return `<h2>How it was made</h2>
  <p class="lede">A reproducible pipeline, not an off-the-shelf sentiment classifier.</p>
  <div class="steps">
    <div class="step"><b>Fine-tuned model</b><span>ModernBERT trained on GoEmotions (28 emotions) and EmoBank (valence), then adapted to lyrics.</span></div>
    <div class="step"><b>Clean corpus</b><span>Studio songs only, covers and live takes removed, ${span} so both bands are compared over the same years.</span></div>
    <div class="step"><b>Album-level statistics</b><span>Albums, not songs, are resampled and permuted, with FDR correction across all 28 emotions.</span></div>
    <div class="step"><b>This page</b><span>Every number and sentence is generated by the pipeline and covered by tests; the page only draws them.</span></div>
  </div>
  <p class="limits"><b>Limits.</b> Lyrics only (no melody, voice or tempo), English only. Irony and fictional narrators can fool the model, and scores compare songs with each other rather than measure absolute sadness.</p>
  <a class="cta" href="${REPO}">Read the code</a><a class="cta ghost" href="${REPO}/tree/main/docs">Methods and reports</a>`;
}

// ---------- album view switch ----------
document.querySelectorAll<HTMLButtonElement>(".seg button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".seg button").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
    document.querySelector<HTMLElement>("#alb-tl")!.hidden = b.dataset.view !== "tl";
    document.querySelector<HTMLElement>("#alb-bands")!.hidden = b.dataset.view !== "bands";
  }),
);

// ---------- reveal on scroll ----------
scatter();
const io = new IntersectionObserver(
  (entries) => entries.forEach((e) => e.isIntersecting && (e.target.classList.add("in"), io.unobserve(e.target))),
  { threshold: 0.15 },
);
document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
