/* Loads the modular slide files listed in SLIDES, injects them into the
 * .slides container in order, then boots Reveal.
 *
 * Slides live in their own files so they stay reviewable in a diff — a talk
 * track is source, and a 900-line index.html is not.
 *
 * NOTE: this uses fetch(), so the deck must be served over http://, not opened
 * as a file:// path. `npm start` does the right thing; if someone double-clicks
 * index.html instead, showFileProtocolHelp() explains why the screen is blank.
 */

const SLIDES = [
  '00-title.html',
  '01-problem.html',
  '02-principle.html',
  '03-build-divider.html',
  '04-week1-classifier.html',
  '05-week2-embeddings.html',
  '06-week3-backbone.html',
  '07-week4-eval.html',
  '08-week5-lora.html',
  '09-week6-rag.html',
  '10-week7-vision.html',
  '11-full-system.html',
  '11c-workflow.html',
  '11b-week8-eval.html',
  '12-demo.html',
  '16-future.html',
  '17-thankyou.html',
  // Removed at the user's request: results, reflection, and the Q&A appendix.
  // The files still exist in slides/ — re-add them here to restore.
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
      hash: true,              // deep-link to a slide; survives a reload mid-talk
      slideNumber: 'c/t',
      controls: true,
      progress: true,
      center: false,           // top-aligned: diagrams jump around less between slides
      transition: 'fade',
      transitionSpeed: 'fast',
      width: 1280,
      height: 760,
      margin: 0.06,
      pdfSeparateFragments: false,  // one page per slide in the print-to-PDF export
      plugins: [RevealNotes, RevealHighlight, RevealMarkdown],
    });
  })
  .catch((err) => showFileProtocolHelp(err.message));
