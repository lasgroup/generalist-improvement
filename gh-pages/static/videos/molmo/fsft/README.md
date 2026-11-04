# FSFT 64-versus-144 video archive

These are selected qualitative examples, not an aggregate evaluation. The main project page includes these clips through the BoN/FSFT method switch.

| Object | Recorded successes, 64 | Recorded successes, 144 |
| --- | --- | --- |
| Atomizer | 3/10 | 2/10 |
| Wine bottle | 1/4 | 1/4 |
| Fork | 0/9 | 1/9 |
| Knife | 1/3 | 1/3 |
| Remote control | 1/9 | 2/9 |
| Bowl | 3/3 | 2/3 |

The initial fixed sample contained 20 matched states. A second fixed sample added six states each for atomizer, fork, and remote control after the desired patterns were absent. All 38 matched pairs / 76 completed episodes are retained in `outputs/molmo-videos/fsft/clips/`, with the combined audit in `outputs/molmo-videos/fsft/clips/audit.json`.

The user explicitly allowed failure/failure for FSFT little-change videos. Selection prefers success/success, then failure/failure for those objects; improvement objects require failure/success. Within the chosen outcome group, the existing webpage episode is preferred, then the first matching episode.

The 64-task source is the supplied seed-1 step-2499 checkpoint. The 144-task source is the user-designated copy at `/capstor/store/cscs/swissai/a0220/mbagatella/params`, identified as seed 0 / step 2500 from the supplied original run. The original path is inaccessible, so its identity could not be independently compared. The copied weights use the shared DROID normalization supplied from the 64-task checkpoint; its hash matches the existing OGPO normalization. Exact paths and hashes are retained in both inventory files and every recording sidecar.

Job 3842291 completed the first sample. Job 3842546 encountered an EGL physical-GPU-selection error before its 64-task step could record; completed 144-task clips were retained and one interrupted fork episode was archived under `fsft-interrupted-clips/`. Job 3842609 resumed with `MUJOCO_EGL_DEVICE_ID` set to the GPU assigned by Slurm. The recorder and inference settings were unchanged. All completed pairs, including those rendered on different GPUs, passed the exact-state/image audit.

The standalone preview is `index.html` in this archive. These assets are also available in the main webpage object explorer. Resume job 3842609 completed with exit 0; both steps completed successfully.
