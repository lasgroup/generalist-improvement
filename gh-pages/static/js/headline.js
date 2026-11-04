'use strict';

// Repack the audited mosaic without trimming, omitting, or stretching a tile.
function headlineTileLayout(count, sourceRows, sourceColumns, sourceWidth, sourceHeight, width, height) {
  if (!Number.isInteger(count) || count < 1 || sourceRows * sourceColumns < count ||
      ![sourceRows, sourceColumns, sourceWidth, sourceHeight, width, height].every(value => Number.isFinite(value) && value > 0)) return null;
  const sw = sourceWidth / sourceColumns;
  const sh = sourceHeight / sourceRows;
  const aspect = sw / sh;
  let best = null;
  for (let columns = 1; columns <= count; columns++) {
    const rows = Math.ceil(count / columns);
    const tileWidth = Math.min(width / columns, height * aspect / rows);
    const empty = rows * columns - count;
    if (!best || tileWidth > best.tileWidth + 0.000001 ||
        (Math.abs(tileWidth - best.tileWidth) < 0.000001 && empty < best.empty)) {
      best = { rows, columns, tileWidth, tileHeight: tileWidth / aspect, empty };
    }
  }
  const top = (height - best.rows * best.tileHeight) / 2;
  return {
    ...best,
    tiles: Array.from({ length: count }, (_, index) => {
      const row = Math.floor(index / best.columns);
      const column = index % best.columns;
      const rowCount = Math.min(best.columns, count - row * best.columns);
      return {
        sx: (index % sourceColumns) * sw,
        sy: Math.floor(index / sourceColumns) * sh,
        sw, sh,
        x: (width - rowCount * best.tileWidth) / 2 + column * best.tileWidth,
        y: top + row * best.tileHeight,
        width: best.tileWidth,
        height: best.tileHeight,
      };
    }),
  };
}

function createHeadlineCanvas(panel, video, stage, onUnavailable) {
  const canvas = panel.querySelector('[data-headline-canvas]');
  const frame = panel.querySelector('.headline-frame');
  const context = canvas?.getContext('2d', { alpha: false });
  const renderer = { available: Boolean(canvas && frame && context) };
  if (!renderer.available) return renderer;
  let layout = null;
  let width = 0;
  let height = 0;
  let pixelRatio = 1;
  let animation = null;
  const draw = () => {
    if (!renderer.available || !layout || video.readyState < 2 || !video.videoWidth) return;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.fillStyle = '#09131d';
    context.fillRect(0, 0, width, height);
    try {
      for (const tile of layout.tiles) {
        context.drawImage(video, tile.sx, tile.sy, tile.sw, tile.sh,
          tile.x, tile.y, tile.width, tile.height);
      }
      panel.classList.toggle('headline-canvas-ready', true);
    } catch (_) {
      renderer.available = false;
      panel.classList.toggle('headline-canvas-ready', false);
      onUnavailable();
    }
  };
  const resize = () => {
    const rect = frame.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    width = rect.width;
    height = rect.height;
    pixelRatio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(width * pixelRatio));
    canvas.height = Math.max(1, Math.round(height * pixelRatio));
    layout = headlineTileLayout(stage.task_count, stage.rows, stage.columns,
      video.videoWidth || stage.width, video.videoHeight || stage.height, width, height);
    if (!layout) {
      renderer.available = false;
      onUnavailable();
      return;
    }
    video.setAttribute('aria-label', `${stage.title}: ${stage.subtitle}. ${stage.task_count} rollouts in ${layout.rows} rows, up to ${layout.columns} columns.`);
    draw();
  };
  if (typeof ResizeObserver === 'function') new ResizeObserver(resize).observe(frame);
  else window.addEventListener('resize', resize);
  video.addEventListener('loadedmetadata', resize);
  video.addEventListener('loadeddata', resize);
  video.addEventListener('seeked', draw);
  video.addEventListener('pause', draw);
  video.addEventListener('ended', draw);
  document.addEventListener('fullscreenchange', resize);
  if (video.requestVideoFrameCallback) {
    const nextFrame = () => {
      draw();
      if (renderer.available) video.requestVideoFrameCallback(nextFrame);
    };
    video.requestVideoFrameCallback(nextFrame);
  } else {
    const nextFrame = () => {
      animation = null;
      draw();
      if (renderer.available && !video.paused && !video.ended && !document.hidden) {
        animation = requestAnimationFrame(nextFrame);
      }
    };
    video.addEventListener('play', () => {
      if (animation === null) animation = requestAnimationFrame(nextFrame);
    });
  }
  resize();
  return renderer;
}

const headline = document.querySelector('.headline-collage');
if (headline) {
  const base = 'static/videos/molmo/headline/';
  fetch(`${base}manifest.json?v=20261104-cleaner`).then(response => {
    if (!response.ok) throw new Error('Headline manifest unavailable');
    return response.json();
  }).then(manifest => {
    if (!manifest.ready || manifest.stages?.length !== 3) return;
    const panels = [...headline.querySelectorAll('[data-headline-stage]')];
    const videos = panels.map(panel => panel.querySelector('video'));
    const renderers = [];
    const error = headline.querySelector('.headline-error');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let inView = false;
    let wantsPlayback = !reducedMotion.matches;
    let starting = false;
    let failed = false;
    let ready = false;
    let playGeneration = 0;
    let loadGeneration = 0;

    const renderCounts = () => panels.forEach((panel, index) => {
      const stage = manifest.stages.find(stage => stage.id === panel.dataset.headlineStage);
      const label = panel.querySelector('[data-headline-count]');
      if (!Array.isArray(stage.outcomes)) return;
      const succeeded = stage.outcomes.filter(outcome => outcome.success &&
        Number.isFinite(outcome.success_seconds) && videos[index].currentTime >= outcome.success_seconds).length;
      label.textContent = `${succeeded} / ${stage.task_count} succeeded`;
      label.title = 'Simulator-confirmed successes in the displayed clips';
    });
    const updateControls = () => {
      // Native controls provide manual playback when autoplay is unavailable.
      videos.forEach((video, index) => {
        const native = reducedMotion.matches || failed || renderers[index]?.available === false;
        video.controls = native;
        panels[index].classList.toggle('headline-native-controls', native);
      });
    };
    const pause = () => {
      playGeneration++;
      videos.forEach(video => video.pause());
      updateControls();
    };
    const play = async () => {
      if (!ready || starting || !wantsPlayback || !inView || document.hidden) return;
      starting = true;
      const generation = ++playGeneration;
      failed = false;
      error.hidden = true;
      try {
        await Promise.all(videos.map(video => video.play()));
        if (!wantsPlayback || !inView || document.hidden) pause();
      } catch (_) {
        // pause() intentionally rejects an outstanding play() promise.
        if (generation === playGeneration) {
          failed = true;
          videos.forEach(video => video.pause());
          error.hidden = false;
        }
      } finally {
        starting = false;
        updateControls();
        if (generation !== playGeneration && wantsPlayback && inView && !document.hidden && !failed) play();
      }
    };
    const restart = () => {
      videos.forEach(video => { video.currentTime = 0; });
      renderCounts();
      play();
    };
    videos.forEach((video, index) => {
      const panel = panels[index];
      const stage = manifest.stages.find(stage => stage.id === panel.dataset.headlineStage);
      if (!stage) throw new Error('Missing headline stage');
      video.muted = true;
      video.poster = `${base}${stage.poster}`;
      video.setAttribute('aria-label', `${stage.title}: ${stage.subtitle}. ${stage.rows} by ${stage.columns} rollout collage.`);
      // Visible captions are authored in index.html; the manifest supplies media and outcomes.
      video.addEventListener('error', () => {
        failed = true;
        pause();
        error.hidden = false;
      });
      video.addEventListener('timeupdate', renderCounts);
      video.addEventListener('ended', () => {
        if (videos.every(item => item.ended) && wantsPlayback && !reducedMotion.matches) restart();
        else if (videos.every(item => item.ended)) {
          wantsPlayback = false;
          updateControls();
        }
      });
      panel.querySelector('[data-headline-fullscreen]').addEventListener('click', () => {
        if (panel.requestFullscreen) panel.requestFullscreen().catch(() => {});
        else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
      });
      renderers.push(createHeadlineCanvas(panel, video, stage, updateControls));
      video.src = `${base}${stage.video}`;
      video.preload = 'auto';
    });
    const load = () => {
      const generation = ++loadGeneration;
      ready = false;
      failed = false;
      error.hidden = true;
      pause();
      const loaded = videos.map(video => new Promise((resolve, reject) => {
        const cleanup = () => {
          video.removeEventListener('loadeddata', onLoaded);
          video.removeEventListener('error', onError);
        };
        const onLoaded = () => { cleanup(); resolve(); };
        const onError = () => { cleanup(); reject(new Error('Video failed to load')); };
        video.addEventListener('loadeddata', onLoaded, { once: true });
        video.addEventListener('error', onError, { once: true });
        video.load();
      }));
      Promise.all(loaded).then(() => {
        if (generation !== loadGeneration) return;
        ready = true;
        updateControls();
        play();
      }).catch(() => {
        if (generation !== loadGeneration) return;
        failed = true;
        error.hidden = false;
        updateControls();
      });
    };
    // The paired training views share a timeline, including pause and replay.
    videos[0].addEventListener('timeupdate', () => {
      if (!wantsPlayback || videos[0].paused || videos[0].ended) return;
      videos.slice(1).forEach(video => {
        if (video.readyState >= 3 && !video.ended && Math.abs(video.currentTime - videos[0].currentTime) > 0.25) {
          video.currentTime = videos[0].currentTime;
        }
      });
    });
    new IntersectionObserver(entries => {
      inView = entries[0].isIntersecting;
      if (inView && wantsPlayback) play();
      else pause();
    }, { threshold: 0.1 }).observe(headline);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) pause();
      else play();
    });
    reducedMotion.addEventListener('change', event => {
      if (event.matches) {
        wantsPlayback = false;
        pause();
      }
    });
    headline.querySelector('[data-headline-legend]').hidden = !manifest.stages.every(stage => Array.isArray(stage.outcomes));
    document.querySelector('.teaser-scenes').hidden = true;
    headline.hidden = false;
    renderCounts();
    load();
  }).catch(() => {}); // Original stills remain available if the manifest cannot load.
}
