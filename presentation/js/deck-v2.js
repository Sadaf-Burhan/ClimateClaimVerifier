/* CONDENSED deck (v2) loader — a parallel build to deck.js.
 *
 * Same machinery as deck.js, different running order: the week-by-week slides
 * (04–11) are replaced by one consolidated "How Climate Scanner works" section
 * (v2-*), while the intro, workflow, reflection, appendix, future, and thank-you
 * slides are REUSED verbatim from the same slides/ directory. deck.js and
 * index.html are untouched — this is a second deck, not an edit of the first.
 */

const SLIDES = [
  '00-title.html',
  '01-problem.html',
  '02-principle.html',
  // --- one clickable architecture slide (replaces weeks 1–8 + the static image) ---
  'v2-architecture.html',
  // --- the honest limits: LoRA, embeddings-can't-corroborate, headline-only bottleneck ---
  'v2-didnt-work.html',
  // --- reused verbatim from the original deck ---
  '11c-workflow.html',   // user workflow
  '12-demo.html',        // live demo
  '16-future.html',      // next steps
  '17-thankyou.html',
];

async function loadSlides() {
  const container = document.querySelector('.reveal .slides');

  const parts = await Promise.all(
    SLIDES.map(async (name) => {
      const res = await fetch(`slides/${name}`);
      if (!res.ok) throw new Error(`${name} -> HTTP ${res.status}`);
      return `<!-- ${name} -->\n${await res.text()}`;
    })
  );

  container.innerHTML = parts.join('\n');
}

function showFileProtocolHelp(err) {
  document.querySelector('.reveal .slides').innerHTML = `
    <section>
      <h2>The deck needs a web server</h2>
      <p class="lede">Slides are loaded with <code>fetch()</code>, which the browser
      blocks on <code>file://</code> for security. Run one of these from
      <code>presentation/</code> and open the URL it prints:</p>
      <pre><code>npm start
# or, with no Node.js:
python -m http.server 8000</code></pre>
      <p class="takeaway flag"><strong>${err}</strong></p>
    </section>`;
  Reveal.initialize({ hash: false });
}

loadSlides()
  .then(() => {
    Reveal.initialize({
      hash: true,
      slideNumber: 'c/t',
      controls: true,
      progress: true,
      center: false,
      transition: 'fade',
      transitionSpeed: 'fast',
      width: 1280,
      height: 760,
      margin: 0.06,
      pdfSeparateFragments: false,
      plugins: [RevealNotes, RevealHighlight, RevealMarkdown],
    });

    // Architecture slide: highlight the map box whose detail card is currently open.
    // The detail card carries data-fragment-index N; the box carries data-box="N".
    const syncArchHighlight = () => {
      document.querySelectorAll('.abox.hot').forEach((b) => b.classList.remove('hot'));
      document.querySelectorAll('.arch').forEach((a) => a.classList.remove('dimmed'));
      const card = document.querySelector('.dcard.current-fragment');
      if (!card) return;
      // NOTE: read data-box-ref, NOT data-fragment-index — Reveal rewrites
      // data-fragment-index to 0-based on init, which would shift the highlight.
      const box = document.querySelector('.abox[data-box="' + card.getAttribute('data-box-ref') + '"]');
      if (!box) return;
      box.classList.add('hot');
      const arch = box.closest('.arch');
      if (arch) arch.classList.add('dimmed');
    };
    Reveal.on('fragmentshown', syncArchHighlight);
    Reveal.on('fragmenthidden', syncArchHighlight);
    Reveal.on('slidechanged', syncArchHighlight);
  })
  .catch((err) => showFileProtocolHelp(err.message));
