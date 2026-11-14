// English display. The page is written in Japanese; in English mode every piece of text is replaced just before it is shown.
// Nothing else in the page has to know about languages: the texts go through `t`, and a watcher applies it to whatever enters the page.
// Data (phrase names, tags stored in the library, paths) is never changed, only how it is shown.
import { EXACT, PATTERNS, TAGS } from "./i18n.en";

export type Lang = "ja" | "en";

const KEY = "dsb.lang";
const JAPANESE = /[぀-ヿ一-鿿]/;

function stored(): Lang | null {
  try {
    const value = localStorage.getItem(KEY);
    return value === "ja" || value === "en" ? value : null;
  } catch { return null; }
}

/** The language of this page. It is fixed while the page is open; changing it reloads the page. */
export const lang: Lang = stored() ?? ((navigator.language ?? "ja").toLowerCase().startsWith("ja") ? "ja" : "en");

export function setLang(next: Lang) {
  try { localStorage.setItem(KEY, next); } catch { /* the choice just does not survive a restart */ }
  location.reload();
}

const SPACED = / ・ | ・$|^・ /;
const exact = new Map<string, string>([...Object.entries(TAGS), ...Object.entries(EXACT)]);
const patterns = PATTERNS.map(([source, out]) => [new RegExp(`^${source}$`, "s"), out] as const);
const cache = new Map<string, string>();
/** Texts that stayed Japanese in English mode (to find what the dictionary lacks). */
export const untranslated = new Set<string>();

/** One text as a whole, or null when the dictionary does not have it. */
function whole(text: string): string | null {
  const hit = exact.get(text);
  if (hit !== undefined) { return hit; }
  if (SPACED.test(text)) { return null; }  // "A ・ B" is done part by part, so that no pattern reaches across the parts
  for (const [pattern, out] of patterns) {
    const match = pattern.exec(text);
    if (match) { return out.replace(/\$(\d)/g, (_, index) => t(match[Number(index)] ?? "")); }
  }
  return null;
}

/** A list ("a・b", "a、b") or a run of sentences, only when every part is known. */
function parts(text: string, separator: RegExp, join: string, end = ""): string | null {
  const pieces = text.split(separator);
  if (pieces.length < 2) { return null; }
  const closed = pieces[pieces.length - 1] === "";
  const used = closed ? pieces.slice(0, -1) : pieces;
  const out: string[] = [];
  for (const piece of used) {
    const done = JAPANESE.test(piece) ? whole(piece.trim()) : piece.trim();
    if (done === null) { return null; }
    out.push(done);
  }
  return out.join(join) + (closed ? end : "");
}

function translate(text: string): string {
  const direct = whole(text);
  if (direct !== null) { return direct; }
  if (text.includes("\n")) { return text.split("\n").map((line) => t(line)).join("\n"); }
  // "A ・ B ・ C": each part on its own; the parts that are data stay as they are.
  if (SPACED.test(text)) {
    const spaced = text.split(/( ・ | ・$|^・ )/); return spaced.map((piece, index) => (index % 2 ? piece.replace("・", "·") : t(piece))).join("");
  }
  return parts(text, /。/, ". ", ".") ?? parts(text, /・/, ", ") ?? parts(text, /、/, ", ") ?? text;
}

/** The text in the page's language. Text without Japanese, and text the dictionary does not know, comes back unchanged. */
export function t(text: string): string {
  if (lang === "ja" || !JAPANESE.test(text)) { return text; }
  const known = cache.get(text);
  if (known !== undefined) { return known; }
  const core = text.trim();
  const start = text.indexOf(core);
  const out = text.slice(0, start) + translate(core) + text.slice(start + core.length);
  if (cache.size > 5000) { cache.clear(); }
  cache.set(text, out);
  if (JAPANESE.test(out)) { untranslated.add(core); }
  return out;
}

const fromEnglish = new Map<string, string>();
for (const [japanese, english] of Object.entries(TAGS)) {
  const key = english.toLowerCase();
  if (!fromEnglish.has(key)) { fromEnglish.set(key, japanese); }
  if (!fromEnglish.has(key.replace(/[- ]/g, ""))) { fromEnglish.set(key.replace(/[- ]/g, ""), japanese); }
}

/** A tag typed in English as the tag the library stores ("kick" -> "キック"). Anything else comes back unchanged. */
export function tagToJapanese(word: string): string {
  if (JAPANESE.test(word)) { return word; }
  const key = word.trim().toLowerCase();
  return fromEnglish.get(key) ?? fromEnglish.get(key.replace(/[- ]/g, "")) ?? fromEnglish.get(key.replace(/s$/, "")) ?? word;
}

/** A search typed in English, with the words that name a tag turned into that tag ("high kick, jump" -> "ハイキック, ジャンプ"). */
export function queryToJapanese(query: string): string {
  if (!query.trim() || JAPANESE.test(query)) { return query; }
  const whole = tagToJapanese(query);
  if (whole !== query) { return whole; }
  return query.split(/([\s,]+)/).map((piece, index) => (index % 2 ? piece : tagToJapanese(piece))).join("");
}

const ATTRIBUTES = ["title", "placeholder", "aria-label", "data-label"];
const written = new WeakMap<Node, string>();

function apply(node: Node) {
  if (node.nodeType === Node.TEXT_NODE) {
    const value = node.nodeValue ?? "";
    if (!JAPANESE.test(value) || written.get(node) === value) { return; }
    const parent = node.parentElement;
    if (parent && (parent.tagName === "SCRIPT" || parent.tagName === "STYLE" || parent.closest('[translate="no"]'))) { return; }
    const out = t(value);
    written.set(node, out);
    if (out !== value) { node.nodeValue = out; }
    return;
  }
  if (node.nodeType !== Node.ELEMENT_NODE) { return; }
  const element = node as Element;
  if (element.tagName === "SCRIPT" || element.tagName === "STYLE" || element.getAttribute("translate") === "no") { return; }
  for (const name of ATTRIBUTES) {
    const value = element.getAttribute(name);
    if (value && JAPANESE.test(value)) {
      const out = t(value);
      if (out !== value) { element.setAttribute(name, out); }
    }
  }
  for (let child = element.firstChild; child; child = child.nextSibling) { apply(child); }
}

/** Start showing the page in English. Does nothing in Japanese. Runs before the first paint, and again on every change. */
export function initLanguage() {
  document.documentElement.lang = lang;
  if (lang === "ja") { return; }
  (window as any).dsbUntranslated = untranslated;
  apply(document.documentElement);
  // The callback runs before the browser paints, so the Japanese text is never seen.
  new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === "childList") { record.addedNodes.forEach(apply); }
      else if (record.type === "characterData") { apply(record.target); }
      else if (record.target.nodeType === Node.ELEMENT_NODE && !(record.target as Element).closest('[translate="no"]')) {
        const element = record.target as Element;
        const value = element.getAttribute(record.attributeName!);
        if (value && JAPANESE.test(value)) {
          const out = t(value);
          if (out !== value) { element.setAttribute(record.attributeName!, out); }
        }
      }
    }
  }).observe(document.documentElement, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRIBUTES });
}
