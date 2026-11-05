# Research project website

Static GitHub Pages site for **Reinforcement Learning Enables Scaling Generalist Robot Policy Improvement**, in the `gh-pages/` directory on the `gh-pages` branch.

## Preview

Run `python3 -m http.server 8000 --directory gh-pages` from the repository root and open <http://localhost:8000>. No build step or package installation is required. Styles, scripts, figures, and the manuscript are served locally; there are no third-party runtime dependencies.

## Publish

The workflow in [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) publishes `gh-pages/` as the website root using GitHub Actions. It runs when website files, the workflow, or its preparation script change on the `gh-pages` branch, and supports manual dispatch on that branch. It checks out the website and `.github/scripts/prepare-pages.py`, stages the static files, then uploads and deploys the staged directory.

The preparation script includes the videos and posters referenced by the headline manifest and the BoN/FSFT `selected_examples.json` manifests. Archived MP4/JPG files stay in the repository for provenance and are excluded from deployment. Missing active media stops deployment before upload. To inspect the deployment locally, run `python3 .github/scripts/prepare-pages.py gh-pages /tmp/generalist-pages-preview` with a new destination directory.

One-time setup:

1. In the repository's **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**.
2. If the `github-pages` environment restricts deployment branches, allow `gh-pages` under **Settings → Environments → github-pages**.
3. Commit the website directory, `.github/workflows/pages.yml`, and `.github/scripts/prepare-pages.py`, then push the `gh-pages` branch. Follow the **Deploy research website** run in the Actions tab.

The deployed site will be <https://lasgroup.github.io/generalist_improvement/>. The folder name does not add another `/gh-pages/` to the public URL. Deployment uses the built-in `GITHUB_TOKEN`; no personal access token is needed. The workflow and website must be pushed before deployment can run.

See GitHub's [custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) for repository setup and deployment permissions.

## Content and provenance

- Favicon: the [official LAS group favicon](https://las.inf.ethz.ch/wp-content/uploads/2015/10/favicon.ico), reused unchanged at its native 32×32 size.
- Template: `/Users/yardas/vla-post-training-website/`, preserving its title/authors/resource links, teaser, summary (now TL;DR), motivation, evaluation, highlights, citation, and attribution structure. Bulma is copied from the supplied template; page styling and JavaScript are adapted for this paper.
- Source: `_ICLR__27__Multi_task_policy_learning_SwissAI (1).zip`, supplied from Downloads. The teaser and scaling plots were subsequently replaced with the supplied `hook_variant_4.pdf` and `main_figure_variant_8.pdf`, extending the displayed training-task range to 144. Other figure variants follow the active paper source.
- `static/paper/generalist-policy-improvement.pdf` is compiled from that ZIP with `latexmk -pdf main_ICLR.tex`; it retains the source manuscript's anonymous submission formatting. The website author list and affiliations were supplied separately by the user. Department and alternate-name entries are consolidated into eight institutions (ETH Zurich, MPI for Intelligent Systems, TUM, CMU, University of Warsaw, UC Berkeley, Microsoft, and UT Austin), with author numbers remapped consistently. The generic Max-Planck entry was matched to MPI for Intelligent Systems using [Leander Diaz-Bone’s profile](https://leanderdiazbone.github.io/). No affiliations were inferred for authors without a number.
- All numerical statements describe the manuscript's simulation experiments. The page labels the manuscript as a submission, not an accepted publication.
- Source figure mapping: `unseen-objects` → `hook_variant_5`; `scaling` → `main_figure_variant_9`; `generalization` → `many_categories_generalization_variant_8`; `recipe-ablations` → `all_ablations`; `policy-extraction` → `actor_critic_ablation_variant_3`; `overview` → TikZ `figs/flagship_ogpo.tex`, compiled standalone with `pdflatex` (times, `arrows.meta`, `matrix`). All plot images, enlarged views, and figure downloads use SVG converted directly from the source PDFs. The teaser, scaling, generalization, and shape-transfer SVGs use PyMuPDF (`page.get_svg_image(text_as_path=True)`); the remaining SVGs use Poppler (`pdftocairo -svg input.pdf output.svg`). The SVGs contain vector paths, including outlined labels, with no embedded raster images or external font dependencies. PDF originals are retained as source assets; the full manuscript is still available as a PDF. The PNG social-preview image is only used for link previews.
- Task images and labels follow `figs/molmo_visualization_small.tex`.
- The per-object `shape-transfer` figure comes directly from the supplied
  `generalization_per_task_variant_4.pdf`; its SVG was exported with PyMuPDF and
  outlined text, and the original PDF is retained.

Paths below are relative to `gh-pages/`. Edit `index.html` for text/authors/citation, `static/css/index.css` for styling, and `static/js/index.js` for figure zoom and clipboard controls. Keep the citation in `static/paper/citation.bib` consistent with the displayed BibTeX.

## BoN rollout comparisons

The video viewer uses actual MolmoSpaces checkpoint rollouts, with exterior and
wrist views in `static/videos/molmo/`. The page has one object explorer: selecting an object picture updates the
64-versus-144 video and its figure-derived success rates.
Each video shows wrist above exterior and 64 tasks beside 144 tasks on one
synchronized playback timeline. Both panels start from the same simulator
state and play at native speed; the shorter rollout holds its final frame.
The exports use H.264, `yuv420p`, and fast-start metadata. Only the selected video
loads when played. A shared picture selector covers all six objects; changing objects or methods
replaces the single video. The original separate-camera exports remain available.

The 64-task checkpoints use step 499999 and the 144-task checkpoints use step
750000. The combined videos display only training-task counts;
checkpoint steps, outcomes, seeds, and initial-state details remain in the manifest. `selected_examples.json` records the
training seed, episode, outcomes, selection rule, and all audited paired outcomes.
Wine bottle can be selected manually for a clearly visible difference; control
examples show both models succeeding. Fork uses training seed 1, episode 2;
remote control uses training seed 0, episode 3 (episode numbers are one-based).
These were selected from the existing recordings without additional sampling.
These clips illustrate individual behaviors;
the supplied figure is the separate aggregate result.

Recording, audit, and export scripts are in the main checkout under
`scripts/molmo_videos/`. After the initial atomizer trials all succeeded with
64 tasks, a second fixed batch extends seed 1 to 12 matched initial states,
keeping both checkpoints unchanged. Every attempt is retained and the manifest
discloses this additional sampling. The complete set contains 108 episodes
(216 raw camera videos), with five selected pairs exported in both camera views
and five combined-camera videos. Run `combine_camera_views.py` after composing
the individual camera views to regenerate the combined videos and manifest.

## Attribution

Website template adaptations are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), preserving the supplied template's attribution to the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) and [Nerfies](https://nerfies.github.io/). The template license does not relicense the research manuscript, figures, or repository code; the existing repository `LICENSE` is unchanged.

## Verification

Visually reviewed browser screenshots at 1440px (desktop), 768px (tablet), 390px and 360px (phones). Checked image loading, horizontal overflow, local resource responses, section anchors, all 24 author entries, affiliation expansion, figure zoom and Escape dismissal, and citation clipboard/download consistency. No missing local resources or JavaScript errors were found. The site uses relative asset paths and was tested under the `/vla-post-training/` project path.

The rollout viewer was checked in Chrome at desktop and 390px phone widths,
including object and camera selection. All 108 recorded episodes passed checks
for completeness, checkpoint provenance, frame counts, native playback rate,
and browser-compatible encoding. All 54 paired simulator initial states and
initial camera images match exactly. All three Slurm jobs completed with exit 0.

The comparison blocks are capped at 860px wide, with video height adapting to
the viewport to fit a laptop screen. Repeated object headings and visible
provenance captions have been removed.

The four added-category cards use 1024px renders of the actual MolmoSpaces
target meshes: salt shaker (task 124), soap dispenser (12), mug (160), and fruit
(241). `scripts/molmo_videos/render_object_closeups.py` isolates target visual
geometry, renders a close camera at native output resolution, and obtains
transparency from the simulator segmentation pass. No image upscaling is used.
An arrow connects these training categories to a separate block containing all
six unseen evaluation objects: atomizer (140), wine bottle (226), fork (345),
knife (155), remote control (357), and bowl (356). Their 1024px transparent
images use the same native-rendering process, implemented in
`scripts/molmo_videos/render_unseen_closeups.py`. Selected camera angles and
mesh identifiers are recorded in `static/images/unseen-object-provenance.json`.
The training categories sit above a downward-pointing arrow leading to the
interactive evaluation viewer. The six unseen-object pictures act as selectors
below its video and success-rate comparison.
The visible rollout disclosure paragraph was removed at the author's request;
the full selection manifest remains with the video assets.

The object pictures support click, Enter/Space, and Left/Right/Home/End keys.
Clicking an object starts its muted video; changing methods preserves playback
state. The viewer does not auto-advance. It starts with BoN on wine bottle.

## Archived OGPO rollout comparisons

OGPO videos are retained in `static/videos/molmo/ogpo/`, but are not exposed in
the webpage carousel. Their checkpoint identifiers, outcomes, initial-state
checks, and file paths remain in each example's `ogpo` manifest entry.
The five comparisons use `mt64_bronet_qvr16_seed0_10k` and
`mt144_bronet_qvr16_r15_seed0_15k`. Raw recordings and additional attempts are
retained under `outputs/molmo-videos/ogpo/` and `outputs/molmo-videos/ogpo64/`.

## Added bowl comparison

Bowl (`molmo_356`) is in the “Little change” selector. Job `3841501` completed
with exit 0, recording three matched initial states for seed 0 at both BoN
checkpoint sizes. All six episodes passed the video/state audit. Outcomes
(64/144) were success/success, success/success, and timeout/success. The first
pair (zero-based episode 0) is shown; both succeed. All initial physics states
and camera images match exactly across training sizes. The six new episodes
bring the BoN archive to 114 episodes and 57 matched pairs.

Raw bowl recordings are in `outputs/molmo-videos/bowl-clips/`; exported videos
are in `outputs/molmo-videos/bowl-website/`. The combined website video retains
the existing compact layout, native playback rate, and final-frame hold.


The archived OGPO wine-bottle selection now uses episode 4: the 64-task policy
times out at 450 steps and the 144-task policy succeeds at 30 steps. The
checkpoints start from exactly matching physics states and camera images.
The additional fixed six-state sample passed all 12 recording audits: 64 tasks
succeeded on none, 144 tasks succeeded only on episode 4. This small,
subsequently selected sample is not an aggregate performance estimate.
`static/videos/molmo/ogpo/wine-additional-sample.json` retains every outcome.
The original episode-1 comparison is still in the repo, referenced in
`ogpo_previous_examples`; the new file includes `episode-04` in its name.
Neither OGPO video is exposed in the page controls.


## FSFT comparisons in the main page

The object explorer includes a BoN/FSFT method switch. Changing methods retains
the selected object and loads its video; BoN remains the default. FSFT uses
`static/videos/molmo/fsft/selected_examples.json` and the six previously audited
combined-camera clips. Knife and bowl show success at both sizes; fork and
remote control show failure at both sizes. Atomizer and wine bottle are selected
failure-to-success examples. Full sampled outcomes and selection provenance
remain in the FSFT archive. OGPO remains archived and hidden.

## Interactive figure comparison

The object explorer pairs each qualitative clip with a separate aggregate
success-rate comparison. `static/videos/molmo/figure_rates.json` contains means
and error-bar endpoints recovered directly from the vector coordinates of
`static/figures/shape-transfer.svg`. It is not a raw evaluation export. The UI
rounds percentages and percentage-point changes to one decimal, marks changes
as approximate, and preserves negative changes (including FSFT wine bottle and
bowl). It never derives success rates from selected video outcomes. The same
0–100% bar scale is used for every object and method.

Regenerate after updating the source figure:
`python3 scripts/molmo_videos/extract_figure_rates.py`. The extractor validates
the six-object, five-training-size structure and records the figure SHA-256.
Error bars retain the figure's endpoints; no interval definition is inferred.
The full original figure remains directly below the viewer.

Chrome visual checks covered the desktop viewer, all six thumbnails, actual
FSFT playback, method switching, and the stacked 390px phone layout.


## Outcome labels and visual review (2026-11-03)

Object thumbnails follow the figure order: atomizer, bowl, fork, knife, remote
control, wine bottle. Video title bars show Success/Failure from each completed
recording's simulator success flag; outcomes are never inferred from duration.
`scripts/molmo_videos/export_outcome_comparisons.py` exports the full native
recordings, retaining both camera views, exact frame rates, every terminal
frame, and the shorter rollout's final-frame hold. Original combined clips
remain in the repository and are referenced by `original_combined_file`.

The BoN remote-control comparison now uses training seed 2, episode 2, from the
existing audited sample, for a clearer visible lift. Both checkpoints report
success. The previous seed-0 selection remains in `previous_bon_selections`.
The BoN 64-task knife's abrupt terminal movement is present in the raw recording,
not introduced during export; a fixed three-state follow-up sample (episodes
3–5, seed 0) uses the same checkpoints and unchanged recorder. All attempts are
retained. Figure-derived numerical values remain explicitly approximate until
the original plotting data is supplied.

The knife follow-up job `3843659` completed successfully. New matched outcomes
(64/144) were success/failure for episode 3, failure/failure for episode 4, and
failure/success for episode 5. No new pair met the required both-success rule.
Episode 0 remains selected, with a visible note about its abrupt terminal motion.
`static/videos/molmo/knife-review.json` preserves the audit of all six seed-0
knife states. All 12 completed knife episodes passed frame, encoding, checkpoint,
and initial-state checks; the BoN archive now contains 120 completed episodes.

The object explorer also shows the proposed shape relationship: atomizer and
wine bottle connect to photos of salt shaker and soap dispenser, and selecting
either highlights those added training categories. Other objects show no close
match among the newly added categories. These hints describe the shape-transfer
hypothesis, independently of the selected method; the accompanying explanation
refers specifically to the BoN aggregate trend. Method-specific rates and rollout
outcomes remain unchanged.

## Headline video collages (2026-11-04)

The page opens with a viewport-height collage before the paper title, authors,
and links. Three simultaneous panels show the base policy on 64 training tasks,
OGPO on those same tasks, and successful OGPO examples on held-out tasks. They
sit side by side on desktop; narrow screens use horizontal swiping between
panels. Visitors scroll below the opening collage to reach the paper details
and remaining page. Both OGPO panels use the supplied
`mt64_bronet_qvr16_seed0_10k` checkpoint. The unchanged aggregate figure sits
alongside the TL;DR.

The source training videos remain 8×8 grids with one object per row and eight
scenes per object. A canvas repacks their tiles to fit each panel's available
width and height, making every tile as large as possible without cropping or
stretching it. All 64 tasks and their row-major order are preserved, and the
paired training panels use the same arrangement at equal dimensions. The final
partial row is centered. The held-out source remains a 3×3 showcase with nine
verified successes: five new-scene tasks,
then atomizer, fork, knife, and remote-control tasks. Cup uses follow-up episode 1,
fork uses episode 2, ladle uses reviewed episode 3, and remote control uses
reviewed episode 2; the other examples use episode 0. The ladle and remote tiles
were replaced after visual review with direct pickups taking 1.32s and 2.64s,
respectively, instead of the original 18.81s and 18.018s successes.

Every training task contributes episode 0, without outcome filtering. The 64 base/OGPO
training pairs use identical initial states, action seeds, and tile ordering.
All 163 original and follow-up recordings passed frame, encoding, checkpoint,
and paired-state checks; all nine displayed held-out rollouts are successful.
Wrist videos preserve native timing, with completed rollouts holding their final
frame; both raw cameras remain archived locally and on Clariden. The held-out
panel is explicitly a selection of successful examples, with all attempted
recordings and selection provenance retained in the audit.

`static/videos/molmo/headline/manifest.json` defines the three panels and
`audit-v3-cleaner.json` retains the fixed plan, attempts, selection, and outcomes. Regenerate them with
`scripts/molmo_videos/export_headline.py`; recording instructions are in
`scripts/molmo_videos/README.md`. Edit visible panel headings and subtitles
directly in `index.html`; JavaScript does not overwrite those captions.
Green tint and borders begin on the exact
terminal success frame and remain during the final hold. Failed training clips
never receive a success overlay. Each panel counts successes in the displayed
clips as they occur; these counters are not paper evaluation rates.

`static/js/headline.js` synchronizes automatic playback and looping across all
three panels, with individual panel-fullscreen controls, offscreen pausing, and
reduced-motion preferences. Canvas layouts respond to resizing and fullscreen
changes; rendering follows decoded video frames and caps pixel density at 2×.
The canvas replaces the source video only after its first successful draw.
Native video controls and the original mosaic remain available for reduced
motion, blocked autoplay, or unavailable canvas rendering. The separate
Pause/Replay buttons are removed. The original stills provide a fallback when
the completed manifest cannot be loaded. This presentation change does not
modify the audited video assets, their timing, or their success overlays.

Validation includes exact success-frame overlay boundaries, unmarked failure
clips, and synchronized playback behavior including retries, user pauses,
offscreen/tab-hidden pauses, looping, reduced motion, and fallback behavior.
Canvas checks cover complete tile coverage and ordering, uncropped aspect
ratios, non-overlapping placement, resize behavior, paired-layout equivalence,
and progressive fallback to the source video.
Chrome visual checks cover desktop and 390px phone layouts.

The three headline videos use `*-web.mp4` encodings for delivery, totaling
16.71 MiB compared with 39.54 MiB for the archived originals. They were encoded
with FFmpeg/libx264, preset `slow`, CRF 26, `yuv420p`, and `+faststart`.
All 482 frame timestamps, the 1872×1056 resolution, and the 31.812-second
duration match each source. The manifest records source and delivery SHA-256
hashes. Sampled VMAF means were 95.16, 96.36, and 96.55, with representative
source and compressed frames also reviewed visually. Deployment staging
reduces the complete website from approximately 265 MiB to 72 MiB.
