import * as d3 from "d3";
import type { Top } from "./types";

export const EMOTION_COLOR: Record<string, string> = {
  annoyance: "#ff6b6b", approval: "#4dd0a8", disappointment: "#7b8cff",
  realization: "#ffd166", sadness: "#4aa3ff", joy: "#ff9f1c",
  confusion: "#c77dff", curiosity: "#2ec4b6", disapproval: "#f06595",
};
// Plain words for the model's label names.
export const WORD: Record<string, string> = {
  annoyance: "irritation", confusion: "doubt", curiosity: "questions", disapproval: "reproach",
};
export const word = (e: string) => WORD[e] ?? e;
export const colorOf = (e: string) => EMOTION_COLOR[e] ?? "#8a8a99";

const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const BASE_SPEED = REDUCED ? 0 : 9; // degrees per second, 33⅓ rpm is far too fast to read

export interface Disc {
  select(emotion: string | null): void;
}

/** One record: slice width = how strongly the band's typical song shows that emotion.
 *  The disc eases to a stop under the pointer so a slice can be read and clicked. */
export function mountDisc(
  host: HTMLElement,
  band: string,
  accent: string,
  top: Top[],
  on: { hover(emotion: string | null, e?: PointerEvent): void; pick(emotion: string): void },
): Disc {
  const R = 54;
  const svg = d3
    .select(host)
    .append("svg")
    .attr("viewBox", "-60 -60 120 120")
    .attr("role", "img")
    .attr("aria-label", `${band}: emotion record`);
  svg.append("circle").attr("r", 58).attr("fill", "#15151c");
  const spin = svg.append("g").attr("class", "spin");

  const pie = d3.pie<Top>().value((d) => d.score).sort(null).padAngle(0.012);
  const arc = d3.arc<d3.PieArcDatum<Top>>().innerRadius(17).outerRadius(R).cornerRadius(1.5);
  const slices = pie(top);

  const wedges = spin
    .selectAll<SVGPathElement, d3.PieArcDatum<Top>>("path")
    .data(slices)
    .join("path")
    .attr("class", "wedge")
    .attr("d", arc)
    .attr("fill", (s) => colorOf(s.data.emotion))
    .attr("tabindex", 0)
    .attr("role", "button")
    .attr("aria-label", (s) => `${word(s.data.emotion)}, ${Math.round(s.data.score * 100)} out of 100`);

  for (const r of [46, 38, 30, 23]) {
    spin.append("circle").attr("r", r).attr("fill", "none").attr("stroke", "#0d0d12")
      .attr("opacity", 0.3).style("pointer-events", "none");
  }
  spin.append("circle").attr("r", 12).attr("fill", accent).style("pointer-events", "none");
  spin.append("circle").attr("r", 2.2).attr("fill", "#0d0d12").style("pointer-events", "none");

  // Glint that stays still while the record turns: sells the vinyl.
  svg.append("path").attr("d", "M-40 -38 A55 55 0 0 1 38 -40").attr("fill", "none")
    .attr("stroke", "#fff").attr("stroke-opacity", 0.07).attr("stroke-width", 7)
    .attr("stroke-linecap", "round").style("pointer-events", "none");

  let angle = Math.random() * 360;
  let speed = BASE_SPEED;
  let hovering = false;
  let last = performance.now();
  const frame = (now: number) => {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    speed += ((hovering ? 0 : BASE_SPEED) - speed) * (1 - Math.exp(-dt * 7));
    angle = (angle + speed * dt) % 360;
    spin.attr("transform", `rotate(${angle.toFixed(2)})`);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);

  wedges
    .on("pointerenter", (e, s) => { hovering = true; on.hover(s.data.emotion, e); })
    .on("pointermove", (e, s) => on.hover(s.data.emotion, e))
    .on("pointerleave", () => { hovering = false; on.hover(null); })
    .on("focus", (_, s) => { hovering = true; on.hover(s.data.emotion); })
    .on("blur", () => { hovering = false; on.hover(null); })
    .on("click", (_, s) => on.pick(s.data.emotion))
    .on("keydown", (e: KeyboardEvent, s) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); on.pick(s.data.emotion); }
    });

  return {
    select(emotion) {
      wedges.each(function (s) {
        const hit = s.data.emotion === emotion;
        const [cx, cy] = arc.centroid(s);
        const k = hit ? 0.09 : 0; // lift the chosen slice off the record
        d3.select(this)
          .classed("sel", hit)
          .classed("dim", emotion !== null && !hit)
          .attr("transform", `translate(${cx * k},${cy * k})`);
      });
    },
  };
}
