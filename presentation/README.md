# ClimateClaimVerifier — Final Presentation

A fully version-controlled talk: [Reveal.js](https://revealjs.com) slides with
architecture diagrams authored in [D2](https://d2lang.com). No PowerPoint, no
Google Slides — everything here is HTML, CSS, JS, and `.d2` source.

Built to the Week-8 deliverable: an ~8-minute demo (problem → pipeline →
live demo → eval → reflection).

## Two decks (both maintained)

There are **two complete decks** in this folder, sharing the same slides, theme,
and diagrams:

| Deck | Open | Purpose |
|---|---|---|
| **Deck 2 — condensed** ⭐ | `index-v2.html` | **The demo-video presentation.** 9 slides: problem → one clickable architecture slide (the whole system, layer by layer) → what didn't work → user workflow → live demo → future → thanks. This is the deck to record the demo video with. |
| **Deck 1 — detailed** | `index.html` | The **week-by-week build story** — how the project was built module by module (Weeks 1–8), one slide per week plus the full architecture, results, and Q&A appendix. Use this as the in-depth walkthrough. |

Both are wired the same way (`js/deck.js` and `js/deck-v2.js` each hold a `SLIDES`
array). Deck 2's architecture slide adds `css/theme-v2.css` and a click-to-highlight
hook in `deck-v2.js`; nothing in Deck 1 depends on those, so editing one never
touches the other.

## Run it

```bash
cd presentation
npm install        # reveal.js + a static file server (one-time)
npm start          # serves at http://localhost:8000 and opens a browser
```

Then open **`http://localhost:8000/index-v2.html`** for the demo deck (Deck 2), or
**`http://localhost:8000/index.html`** for the detailed week-by-week deck (Deck 1).

The deck **must be served over HTTP**, not opened as a `file://` path — slides
are loaded with `fetch()`, which browsers block on `file://`. If you open
`index.html` directly you'll get a one-slide message explaining exactly this.
Any static server works; if you have no Node.js: `python -m http.server 8000`.

## Present

- **Arrows / Space** — next / previous. **Esc** — slide overview grid.
- **`S`** — speaker-view: notes, timer, and next-slide preview in a second window.
  Every slide's talk track (with segment timings) lives in its speaker notes.
- **`F`** — fullscreen. **`?`** — all shortcuts.
- The main talk ends at **Reflection**. Press **↓** from there to reach the
  **appendix** backup slides (red-flag logic, deployment, eval discipline, the
  LoRA frontier, known gaps) — one per likely Q&A question.
- Export to PDF: open `http://localhost:8000/?print-pdf` and print to PDF from
  the browser.

## How it's organized

```
presentation/
├── index.html            # Deck 1 shell (detailed, week-by-week)
├── index-v2.html         # Deck 2 shell (condensed, demo video) ⭐
├── slides/               # one .html file per slide, injected in order
│   ├── NN-*.html         #   Deck 1's per-week slides
│   └── v2-*.html         #   Deck 2's consolidated slides (architecture, didn't-work)
├── diagrams/             # one .d2 source + its rendered .svg per concept
│   ├── _theme.d2         #   shared colour language, imported by every diagram
│   └── NN-*.d2 / .svg
├── css/theme.css         # academic theme layered over reveal's `white`
├── css/theme-v2.css      # Deck 2 additions (architecture boxes, detail cards)
├── js/deck.js            # Deck 1 loader — fetches slides/, boots Reveal
├── js/deck-v2.js         # Deck 2 loader — its own SLIDES + click-to-highlight hook
└── build-diagrams.sh     # re-render every .d2 -> .svg
```

**One diagram per concept**, built up week by week so the architecture accretes
in the same order the course did. Each slide references its SVG; the slide text
stays minimal and the detail lives in the speaker notes.

## Editing

- **A slide:** edit the matching file in `slides/`. To add/remove/reorder, update
  the `SLIDES` array in `js/deck.js`.
- **A diagram:** edit the `.d2` source, then re-render:
  ```bash
  npm run diagrams          # all of them
  ./build-diagrams.sh 06    # just the ones matching "06"
  ```
  Needs the D2 CLI: `winget install Terrastruct.D2` (or see https://d2lang.com).
  Diagram `09` uses the `elk` layout engine (its 8-way fan-in routes badly under
  the default `dagre`); the build script picks the engine per file.

### The colour language

Every box is coloured by *what kind of stage it is*, identically on a slide and
inside a diagram (defined once in `diagrams/_theme.d2` and `css/theme.css`):

| Colour | Meaning |
|---|---|
| slate | data — sources, storage, corpora |
| indigo | LLM — anything that calls a language model |
| teal | embedding / vector retrieval |
| amber | the multimodal (vision) path |
| green | what the reader is shown |
| red | the red flag |
| grey, dashed | built, measured, deliberately **not** shipped |

## A note on the numbers

The eval figures on the results slide are the **reproducible** ones from
`../data/eval_history.jsonl` (classifier recall **0.875**, precision **0.70** on
the frozen gold set), not the higher historical peak some project docs still
quote. The Week-5 LoRA frontier table shows that era's base measurement (0.938)
for a like-for-like comparison — the appendix speaker notes flag the difference.
This is deliberate: the talk's own thesis is measurement discipline.
