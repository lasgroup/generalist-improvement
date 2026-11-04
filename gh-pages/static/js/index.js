'use strict';

const backToTop = document.querySelector('.scroll-to-top');
const syncScrollButton = () => backToTop.classList.toggle('visible', window.scrollY > 500);
window.addEventListener('scroll', syncScrollButton, { passive: true });
syncScrollButton();

const copyButton = document.querySelector('.copy-bibtex-btn');
const copyStatus = document.querySelector('.copy-status');
copyButton.addEventListener('click', async () => {
  const citation = document.querySelector('#bibtex-code code').textContent;
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = 'Copied!';
    copyStatus.textContent = 'Citation copied to clipboard.';
    window.setTimeout(() => { copyButton.textContent = 'Copy citation'; }, 2500);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex-code'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Citation selected. Use your browser’s Copy command, or download the .bib file.';
  }
});

const dialog = document.querySelector('#figure-dialog');
const enlargedFigure = document.querySelector('#enlarged-figure');
let figureTrigger;
document.querySelectorAll('[data-zoom]').forEach(link => {
  link.addEventListener('click', event => {
    if (!dialog.showModal || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const original = link.querySelector('img');
    figureTrigger = link;
    enlargedFigure.src = link.href;
    enlargedFigure.alt = original.alt;
    enlargedFigure.classList.toggle('portrait', original.naturalHeight > original.naturalWidth * 0.85);
    document.querySelector('#figure-dialog-title').textContent = original.alt;
    dialog.showModal();
    document.body.classList.add('figure-open');
    dialog.querySelector('.zoom-scroll').scrollTo(0, 0);
  });
});
dialog.querySelector('.close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) {
  const box = dialog.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
}});
dialog.addEventListener('close', () => {
  document.body.classList.remove('figure-open');
  figureTrigger?.focus({ preventScroll: true });
});

const explorer = document.querySelector('.object-explorer');
if (explorer) {
  const methods = {
    bon: { label: 'BoN', base: 'static/videos/molmo/' },
    fsft: { label: 'FSFT', base: 'static/videos/molmo/fsft/' },
  };
  let selectedMethod = 'bon';
  let selectedTask = Number(explorer.dataset.defaultTask);
  let figureRates;
  const video = explorer.querySelector('video');
  const observation = explorer.querySelector('.rollout-observation');
  const title = explorer.querySelector('#explorer-title');
  const objectPhoto = explorer.querySelector('.explorer-object-photo');
  const relatives = explorer.querySelector('.shape-relatives');
  const noShapeMatch = explorer.querySelector('.shape-no-match');
  const relationArrow = explorer.querySelector('.shape-relation-arrow');
  const relatedTrainingObjects = [...document.querySelectorAll('[data-shape-match]')];
  const status = explorer.querySelector('.explorer-status');
  const objectButtons = [...explorer.querySelectorAll('[data-rollout-task]')];
  const methodControls = explorer.querySelector('.rollout-methods');
  const methodButtons = [...methodControls.querySelectorAll('[data-rollout-method]')];
  const result = explorer.querySelector('.explorer-result');
  const chart = result.querySelector('.rate-chart');
  const deltaLabel = result.querySelector('[data-rate-delta]');
  const percent = value => `${(value * 100).toFixed(1)}%`;

  const showShapeRelation = () => {
    const hasMatch = selectedTask === 140 || selectedTask === 226;
    relatives.hidden = !hasMatch;
    relationArrow.hidden = !hasMatch;
    noShapeMatch.hidden = hasMatch;
    relatedTrainingObjects.forEach(object => { object.dataset.shapeMatch = String(hasMatch); });
    return hasMatch
      ? 'Similar shapes added at 144 tasks: salt shaker and soap dispenser.'
      : 'No close shape match among the added objects.';
  };

  const showRates = () => {
    const object = figureRates?.objects.find(item => item.task_id === selectedTask);
    const rates = object?.[selectedMethod];
    const index64 = figureRates?.training_tasks.indexOf(64);
    const index144 = figureRates?.training_tasks.indexOf(144);
    if (!rates?.[index64] || !rates?.[index144]) {
      result.hidden = true;
      explorer.classList.remove('has-rates');
      return;
    }
    const rate64 = rates[index64];
    const rate144 = rates[index144];
    const delta = Math.round((rate144.mean - rate64.mean) * 1000) / 10;
    const signed = delta > 0 ? `+${delta.toFixed(1)}` : delta < 0 ? `−${Math.abs(delta).toFixed(1)}` : '0.0';
    result.querySelector('#figure-result-title').textContent = `${methods[selectedMethod].label} · Success rate`;
    deltaLabel.textContent = `≈ ${signed}`;
    deltaLabel.dataset.direction = delta > 0 ? 'up' : delta < 0 ? 'down' : 'flat';
    [[64, rate64], [144, rate144]].forEach(([tasks, rate]) => {
      const row = result.querySelector(`[data-rate-row="${tasks}"]`);
      row.querySelector('strong').textContent = percent(rate.mean);
      row.querySelector('.rate-fill').style.width = `${rate.mean * 100}%`;
      const error = row.querySelector('.rate-error');
      const low = Math.max(0, rate.low);
      const high = Math.min(1, rate.high);
      error.style.left = `${low * 100}%`;
      error.style.width = `${(high - low) * 100}%`;
    });
    const summary = `${methods[selectedMethod].label}, ${object.object}: figure success rate ${percent(rate64.mean)} at 64 tasks and ${percent(rate144.mean)} at 144 tasks; change ${signed} percentage points. Values rounded from the figure.`;
    chart.setAttribute('aria-label', summary);
    result.hidden = false;
    explorer.classList.add('has-rates');
    return summary;
  };

  const showRollout = (play = false, announce = false) => {
    const method = methods[selectedMethod];
    const examples = method.examples || [];
    const example = examples.find(item => item.task_id === selectedTask);
    objectButtons.forEach(button => {
      const task = Number(button.dataset.rolloutTask);
      button.disabled = !examples.some(item => item.task_id === task && item.combined_file);
      button.setAttribute('aria-pressed', String(task === selectedTask));
    });
    methodButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.rolloutMethod === selectedMethod)));
    explorer.dataset.method = selectedMethod;
    if (example) {
      title.textContent = example.object;
      objectPhoto.src = objectButtons.find(button => Number(button.dataset.rolloutTask) === selectedTask).querySelector('img').getAttribute('src');
      observation.textContent = example.visual_review_note || '';
      observation.hidden = !example.visual_review_note;
      const source = `${method.base}${example.combined_file}`;
      if (video.getAttribute('src') !== source) {
        video.pause();
        video.src = source;
        video.poster = source.replace(/\.mp4$/, '.jpg');
        video.querySelector('a').href = source;
        video.load();
      }
      video.setAttribute('aria-label', `${method.label} ${example.object.toLowerCase()}: 64 tasks ${example.success_64 ? 'Success' : 'Failure'}; 144 tasks ${example.success_144 ? 'Success' : 'Failure'}. Outcomes from the simulator evaluator; synchronized wrist and exterior cameras.`);
      if (play) video.play().catch(() => {});
    }
    const summary = showRates();
    const shapeSummary = showShapeRelation();
    if (announce) status.textContent = `${summary || `${method.label}: ${title.textContent}, 64 versus 144 training tasks.`} ${shapeSummary}`;
  };

  objectButtons.forEach(button => {
    button.addEventListener('click', () => {
      selectedTask = Number(button.dataset.rolloutTask);
      showRollout(true, true);
    });
    button.addEventListener('keydown', event => {
      const enabled = objectButtons.filter(item => !item.disabled);
      let next = enabled.indexOf(button);
      if (event.key === 'ArrowRight') next++;
      else if (event.key === 'ArrowLeft') next--;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = enabled.length - 1;
      else return;
      event.preventDefault();
      const target = enabled[(next + enabled.length) % enabled.length];
      target.focus({ preventScroll: true });
      selectedTask = Number(target.dataset.rolloutTask);
      showRollout(!video.paused, true);
    });
  });
  methodButtons.forEach(button => button.addEventListener('click', () => {
    const method = button.dataset.rolloutMethod;
    if (selectedMethod === method || !methods[method].examples) return;
    selectedMethod = method;
    showRollout(!video.paused, true);
  }));

  Object.entries(methods).forEach(([key, method]) => {
    fetch(`${method.base}selected_examples.json?v=20261103-outcomes`).then(response => {
      if (!response.ok) throw new Error('Video manifest unavailable');
      return response.json();
    }).then(data => {
      if (!Array.isArray(data.examples) || !objectButtons.every(button => data.examples.some(item => item.task_id === Number(button.dataset.rolloutTask) && item.combined_file))) {
        throw new Error('Video examples incomplete');
      }
      method.examples = data.examples;
      methodButtons.find(button => button.dataset.rolloutMethod === key).disabled = false;
      methodControls.hidden = false;
      if (key === selectedMethod) showRollout();
    }).catch(() => {
      methodButtons.find(button => button.dataset.rolloutMethod === key).disabled = true;
      if (key === selectedMethod) objectButtons.forEach(button => { button.disabled = true; });
    });
  });
  fetch('static/videos/molmo/figure_rates.json?v=20261103-outcomes').then(response => {
    if (!response.ok) throw new Error('Figure data unavailable');
    return response.json();
  }).then(data => {
    if (!Array.isArray(data.objects) || !Array.isArray(data.training_tasks)) throw new Error('Invalid figure data');
    figureRates = data;
    showRates();
  }).catch(() => { result.hidden = true; explorer.classList.remove('has-rates'); });
}
