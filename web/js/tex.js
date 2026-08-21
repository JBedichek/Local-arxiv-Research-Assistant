/* Inline LaTeX, rendered without a library.
 *
 * Pure: text in, HTML out, no DOM and no imports. */

/* ── Inline LaTeX ─────────────────────────────────────────────────────────────
 *
 * 66% of corpus chunks contain $math$. Papers themselves render fine — LaTeXML
 * ships MathML and browsers handle that natively — but chunk *text* is stored as
 * each element's `alttext`, i.e. the original LaTeX. So every excerpt, answer and
 * extracted claim showed raw `$O(\epsilon^{-3})$` until now.
 *
 * This is deliberately not KaTeX. Vendoring it means ~1 MB of third-party code and
 * webfonts inside a proprietary repository, and the math that actually appears in
 * these answers is inline notation — subscripts, Greek, big-O, fractions — not
 * displayed multi-line equations. The command set was measured rather than guessed:
 * frac, mathcal, in, left/right, mathbf, leq, displaystyle, bm, sum, prime, text,
 * alpha, textbf, theta, hat, mathbb, tilde account for nearly all of it.
 *
 * Anything it cannot express degrades to the source text with the delimiters
 * removed, which is what was displayed before — never worse, usually much better.
 * If full fidelity is wanted later, KaTeX drops into `texToHtml` unchanged.
 */
const MATH_SYMBOLS = {
  alpha:"α",beta:"β",gamma:"γ",delta:"δ",epsilon:"ε",varepsilon:"ε",zeta:"ζ",eta:"η",
  theta:"θ",vartheta:"ϑ",iota:"ι",kappa:"κ",lambda:"λ",mu:"μ",nu:"ν",xi:"ξ",pi:"π",
  rho:"ρ",sigma:"σ",tau:"τ",upsilon:"υ",phi:"φ",varphi:"φ",chi:"χ",psi:"ψ",omega:"ω",
  Gamma:"Γ",Delta:"Δ",Theta:"Θ",Lambda:"Λ",Xi:"Ξ",Pi:"Π",Sigma:"Σ",Upsilon:"Υ",
  Phi:"Φ",Psi:"Ψ",Omega:"Ω",
  times:"×",cdot:"·",cdots:"⋯",ldots:"…",dots:"…",div:"÷",pm:"±",mp:"∓",
  leq:"≤",le:"≤",geq:"≥",ge:"≥",neq:"≠",ne:"≠",approx:"≈",sim:"∼",simeq:"≃",
  equiv:"≡",propto:"∝",ll:"≪",gg:"≫",
  in:"∈",notin:"∉",subset:"⊂",subseteq:"⊆",supset:"⊃",supseteq:"⊇",cup:"∪",cap:"∩",
  emptyset:"∅",forall:"∀",exists:"∃",neg:"¬",land:"∧",lor:"∨",
  to:"→",rightarrow:"→",Rightarrow:"⇒",leftarrow:"←",Leftarrow:"⇐",
  leftrightarrow:"↔",Leftrightarrow:"⇔",mapsto:"↦",implies:"⟹",
  infty:"∞",partial:"∂",nabla:"∇",sum:"∑",prod:"∏",int:"∫",oint:"∮",
  sqrt:"√",angle:"∠",perp:"⊥",parallel:"∥",star:"⋆",ast:"∗",circ:"∘",
  oplus:"⊕",otimes:"⊗",odot:"⊙",bullet:"•",dagger:"†",prime:"′",ell:"ℓ",hbar:"ℏ",
  Re:"ℜ",Im:"ℑ",aleph:"ℵ",top:"⊤",bot:"⊥",vdots:"⋮",ddots:"⋱",lVert:"‖",rVert:"‖",
  dotsc:"…",dotsb:"⋯",dotso:"…",natural:"♮",flat:"♭",sharp:"♯",
  langle:"⟨",rangle:"⟩",lfloor:"⌊",rfloor:"⌋",lceil:"⌈",rceil:"⌉",quad:" ",qquad:"  ",
};
const BLACKBOARD = {R:"ℝ",N:"ℕ",Z:"ℤ",Q:"ℚ",C:"ℂ",E:"𝔼",P:"ℙ",1:"𝟙"};
const CALLIGRAPHIC = {A:"𝒜",B:"ℬ",C:"𝒞",D:"𝒟",E:"ℰ",F:"ℱ",G:"𝒢",H:"ℋ",I:"ℐ",J:"𝒥",
  K:"𝒦",L:"ℒ",M:"ℳ",N:"𝒩",O:"𝒪",P:"𝒫",Q:"𝒬",R:"ℛ",S:"𝒮",T:"𝒯",U:"𝒰",V:"𝒱",W:"𝒲",
  X:"𝒳",Y:"𝒴",Z:"𝒵"};
const UPRIGHT_FNS = ["log","exp","min","max","argmin","argmax","sin","cos","tan","det",
  "dim","ker","deg","gcd","lim","sup","inf","ln","Pr","tr","diag","softmax","sign","rank"];

/* Read the balanced {...} beginning at `i`, or a single following character.
   LaTeX groups nest, so a regex cannot do this: \frac{\frac{a}{b}}{c} is common. */
function texGroup(s, i) {
  if (s[i] !== "{") return [s[i] ?? "", i + 1];
  let depth = 0;
  for (let j = i; j < s.length; j++) {
    if (s[j] === "{") depth++;
    else if (s[j] === "}" && --depth === 0) return [s.slice(i + 1, j), j + 1];
  }
  return [s.slice(i + 1), s.length];
}

function texToHtml(src) {
  let out = "";
  for (let i = 0; i < src.length; ) {
    const ch = src[i];
    if (ch === "\\") {
      const m = /^\\([a-zA-Z]+|.)/.exec(src.slice(i));
      if (!m) { i++; continue; }
      const cmd = m[1];
      i += m[0].length;
      if (cmd === "frac" || cmd === "dfrac" || cmd === "tfrac") {
        const [num, i1] = texGroup(src, i); const [den, i2] = texGroup(src, i1);
        i = i2;
        out += `<span class="mfrac"><span class="mnum">${texToHtml(num)}</span>` +
               `<span class="mden">${texToHtml(den)}</span></span>`;
      } else if (cmd === "sqrt") {
        const [a, i1] = texGroup(src, i); i = i1;
        out += `√<span class="msqrt">${texToHtml(a)}</span>`;
      } else if (cmd === "mathbb") {
        const [a, i1] = texGroup(src, i); i = i1;
        out += BLACKBOARD[a] || `<span class="mbb">${texToHtml(a)}</span>`;
      } else if (cmd === "mathcal" || cmd === "mathscr") {
        const [a, i1] = texGroup(src, i); i = i1;
        out += CALLIGRAPHIC[a] || `<i>${texToHtml(a)}</i>`;
      } else if (cmd === "mathbf" || cmd === "bm" || cmd === "boldsymbol" || cmd === "textbf") {
        const [a, i1] = texGroup(src, i); i = i1;
        out += `<b>${texToHtml(a)}</b>`;
      } else if (cmd === "text" || cmd === "mathrm" || cmd === "textrm" || cmd === "operatorname") {
        const [a, i1] = texGroup(src, i); i = i1;
        out += `<span class="mup">${texToHtml(a)}</span>`;
      } else if (cmd === "mathit" || cmd === "textit" || cmd === "emph") {
        const [a, i1] = texGroup(src, i); i = i1;
        out += `<i>${texToHtml(a)}</i>`;
      } else if (cmd === "hat" || cmd === "tilde" || cmd === "bar" || cmd === "vec" ||
                 cmd === "dot" || cmd === "widehat" || cmd === "widetilde") {
        const [a, i1] = texGroup(src, i); i = i1;
        const acc = {hat:"̂",widehat:"̂",tilde:"̃",widetilde:"̃",
                     bar:"̄",vec:"⃗",dot:"̇"}[cmd];
        out += texToHtml(a) + acc;             // combining mark renders over the glyph
      } else if (UPRIGHT_FNS.includes(cmd)) {
        out += `<span class="mup">${cmd}</span>`;
      } else if (cmd === "left" || cmd === "right" || cmd === "displaystyle" ||
                 cmd === "textstyle" || cmd === "limits" || cmd === "nolimits" ||
                 cmd === "!" || cmd === "," || cmd === ";" || cmd === ":") {
        // Sizing and spacing hints with no meaning outside a real typesetter.
        if (cmd === "," || cmd === ";" || cmd === ":") out += " ";
      } else if (MATH_SYMBOLS[cmd] !== undefined) {
        out += MATH_SYMBOLS[cmd];
      } else if (cmd === "\\") {
        out += "<br>";
      } else if (cmd === "lx" && src.slice(i, i + 12) === "@sectionsign") {
        out += "§"; i += 12;                        // LaTeXML internal, common in alttext
      } else if (/^[a-zA-Z]+$/.test(cmd)) {
        out += `<span class="mup">${cmd}</span>`;   // unknown command: show its name
      } else {
        out += cmd;                                 // escaped literal: \{ \% \_ \&
      }
    } else if (ch === "^" || ch === "_") {
      const [a, i1] = texGroup(src, i + 1); i = i1;
      out += ch === "^" ? `<sup>${texToHtml(a)}</sup>` : `<sub>${texToHtml(a)}</sub>`;
    } else if (ch === "{" || ch === "}") {
      i++;                                          // grouping braces are not content
    } else if (ch === "&") {
      // Already-escaped entity from the caller (&amp; &lt; &gt;): copy it whole.
      const ent = /^&[a-z]+;|^&#\d+;/.exec(src.slice(i));
      if (ent) { out += ent[0]; i += ent[0].length; } else { out += "&amp;"; i++; }
    } else {
      out += ch; i++;
    }
  }
  return out;
}

/* Replace $…$ and $$…$$ in an ALREADY HTML-ESCAPED string.
   Escaped input is the precondition: this emits tags, so running it on raw text
   would let a passage containing markup inject it. */
/* Prose, not maths. Chunking can split a formula, leaving a chunk with an odd number
   of `$`; pairing then runs from one formula's closing delimiter to the next one's
   opening one and swallows the sentence between them — the observed case was
   "$X$ -QAM scheme, where $Y$" rendering the prose as maths. Real inline maths is short
   and carries at least one LaTeX marker, so anything long, marker-free and full of
   ordinary words is left exactly as it was. */
function looksLikeProse(src) {
  if (/[\\^_{}]/.test(src)) return false;
  const words = src.trim().split(/\s+/);
  return words.length >= 4 || src.length > 60;
}

export function renderMath(escaped) {
  if (!escaped || escaped.indexOf("$") < 0) return escaped;
  return escaped.replace(/\$\$([\s\S]+?)\$\$|(?<!\\)\$([^$\n]+?)(?<!\\)\$/g,
    (whole, display, inline) => {
      const src = display ?? inline;
      if (looksLikeProse(src)) return whole;
      try {
        const html = texToHtml(src);
        return display
          ? `<span class="math-display">${html}</span>`
          : `<span class="math">${html}</span>`;
      } catch {
        return src;                 // never worse than the raw source it replaced
      }
    });
}
