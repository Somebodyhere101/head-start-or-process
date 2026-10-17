# Head Start or Process?

Code, preregistrations and results for **"Head Start or Process? Decomposing How an Experience-Acquired Abstraction Shapes Later Learning in a Small Language Model"** (Sam Ramdan, October 2026).

- Paper: [`scaffold/paper/head-start-or-process.pdf`](scaffold/paper/head-start-or-process.pdf) · Zenodo: [doi:10.5281/zenodo.23432800](https://doi.org/10.5281/zenodo.23432800)
- Frozen forecast for the four held-out adapters: [`scaffold/FSIGN_FORECAST.md`](scaffold/FSIGN_FORECAST.md) · Zenodo: [doi:10.5281/zenodo.23432798](https://doi.org/10.5281/zenodo.23432798)
- Weights (not in git): the [v1.0 release](https://github.com/Somebodyhere101/head-start-or-process/releases/tag/v1.0) carries every learned B adapter (`b-adapters-part01-of-09.zip` to `part09`) and the A, control and Stage-0 tensors (`a-and-control-adapters.zip`)

## What the paper asks and finds

When a network that has acquired an abstraction *A* learns a second abstraction *B* faster, does *A* supply information (a head start, reusable features) or change the *process* by which *B* is learned? The paper decomposes a gradient learner's function-space update into starting predictions, evidence and tangent kernel. It isolates the kernel channel with an **output-matched learner**: *A*'s network with a fixed stop-gradient offset on the answer log-probabilities, so that it starts *B* with the naive model's predictions. *A* is a rank-8 LoRA on Qwen3-1.7B that learned a hidden two-cue conjunction from outcome-labelled episodes.

- **Where *A* is.** The compact layer-24 mediator of *A*'s transfer is the answer-letter readout. No tested position carries a letter-free decision variable (one adapter tested fully, five on the training domains).
- **Head start, not process.** *A*'s raw advantage is a head start plus a network effect that works only from *A*'s own saturated predictions. From matched starts, *A*'s network gives no detectable mean speed-up.
- **One run is not a reference.** Near-inert replicates of the base learner differ by up to 0.28 in mean held-out balanced accuracy on identical data, so single-run comparisons, including the preregistered ones, are dominated by trajectory noise. Against replicated references (post hoc), the mean kernel-channel benefit is below +0.02 at the onset of learning on a task whose rule *A* does not share, and below +0.09 on a task that extends *A*; whether *A* is a scaffold there is unresolved.
- **The method is probably not blind.** Post hoc, all three shuffled-outcome adapters slowed matched-start learning (not significant at the adapter level), and one of four kernel statistics (sign-gradient alignment) ranked the six adapters' matched-start advantage at Spearman 0.94 on both tasks (uncorrected). That ranking is frozen as a forecast for four held-out adapters.
- **Verdict.** The preregistered process-only positive control failed calibration, so the frozen verdict is formally uninterpretable. The evidence supports no detectable mean benefit from a matched start, not the absence of process effects.

All claims are about one 1.7B model, rank-8 adapters and a templated world. See the paper's Limitations.

## Layout

The layout is the one the code was run from; scripts find each other by relative path.

| path | contents |
|---|---|
| `scaffold/` | This study. `h2lib.py` (the frozen update, adapters, offsets, evaluation), `h2run.py` (one B-learning cell: arms base, A_raw, prior, A_matched, A_matched_fade, controls, learning-rate arms), `bworld.py` (the B tasks), `stage0.py` (interchange tests), `train_controls.py` (shuf/single/perm adapters), `forecast.py` (t = 0 kernel statistics), `analyze.py` (preregistered Stage-2 analysis), `explore_*.py` (E1, E2), `robustness.py` (E3), `forecast_eval.py` (E4), `make_forecast.py` (E5), `run_*.ps1` (the chains that were run; they contain the original absolute paths). |
| `scaffold/*.md` | Preregistrations (Stages 1-3), outcome notes, the deviation log `DEVIATIONS.md` (D1-D6, E1-E5), `RESULTS.md`, `FSIGN_FORECAST.md`, and `PREREGISTRATIONS_SHA256SUMS`. |
| `scaffold/results/` | Every cell's JSON, logs and stdout, controls' summaries, Stage-0 results, forecasts and analyses (`stage2/ANALYSIS.json`, `EXPLORE_E1-E3.json`, `FORECAST_EVAL.json`). The `.pt` tensors are in the release. |
| `scaffold/paper/` | The PDF, its LaTeX source, `make_tables.py` and `make_figs.py` (write every table and figure from `scaffold/results/`). |
| `ced/` | Modules of the author's earlier consequence-learning study that this code imports (world, questions, training), unchanged. `ced/results/<name>/lora_none.pt` receives the ten A adapters from the release. |
| `retrogenesis/local_models.py` | Model loader and the hand-written LoRA. |

## Environment

Python 3.12 and `pip install -r requirements.txt` (torch 2.7.1 with CUDA 12.8, transformers 5.16.1). Install the CUDA build of torch from the PyTorch index, for example `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`. The model is Qwen3-1.7B, Hugging Face snapshot `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, in bf16. `local_models.py` looks for it in the Hugging Face cache; set `CED_MODEL_DIR` to a local path or to `Qwen/Qwen3-1.7B` to load it from elsewhere. All runs used one laptop RTX 5070 (8 GB); a B-learning arm takes 20-120 s.

## Rerunning

Download the release assets and unzip them at the repository root (they restore `ced/results/<A>/lora_none.pt`, `scaffold/results/controls/*.pt` and every `*_phi.pt`).

```
cd scaffold
python h2run.py --rule bind --bseed 101 --passes 1 --eval_every 24 --arms base,A_raw,prior,A_matched,A_matched_fade --adapter ST2_20260919 --tag stage2
python h2run.py --rule bext --bseed 101 --passes 3 --eval_every 72 --arms base,A_raw,prior,A_matched,A_matched_fade --adapter ST2_20260919 --tag stage2
python h2run.py --rule bind --bseed 101 --passes 1 --eval_every 24 --arms A_matched --adapter results/controls/shuf_20260919.pt --tag stage2
python analyze.py --tag stage2      # preregistered Stage-2 analysis -> results/stage2/ANALYSIS.json
python robustness.py                # E3: replicated references, three-way bootstrap, permutation tests
python forecast_eval.py             # E4: the preregistered forecast and the adapter-level kernel check
```

`run_grid.ps1` lists the full Stage-2 grid (6 A adapters x B seeds 101-103 x 2 tasks, plus 9 control adapters). The analyses need only the JSON files in `scaffold/results/`, so they run without a GPU.

**The paper.**

```
cd scaffold/paper
python make_tables.py && python make_figs.py stage2
pdflatex main && pdflatex main
```

## Preregistration

| stage | file | frozen (+04:00) | SHA-256 (first 16) |
|---|---|---|---|
| 1 | `PREREGISTRATION.md` | 2026-10-16 12:35 | `37506f0cc5994493` |
| 2 | `PREREGISTRATION_STAGE2.md` (with `STAGE0_OUTCOME.md`, `PILOT_OUTCOME.md` and the code) | 2026-10-16 23:32 | `0e369defb4c5f59b` |
| 3 | `PREREGISTRATION_STAGE3.md` (frozen, stopped during its pilot, not reported) | 2026-10-17 09:28 | `89b30e9f0477087f` |
| E5 | `FSIGN_FORECAST.md` (forecast for the four held-out adapters) | 2026-10-17 10:47 | `0400cbfb03436aa5` |

Check them with `sha256sum -c PREREGISTRATIONS_SHA256SUMS` in `scaffold/`. Every preregistration document matches. The lines that do not match are expected: `DEVIATIONS.md` was appended after the Stage-2 freeze (D6, E1-E5), and the Stage-2 hashes of `h2run.py`, `bworld.py`, `analyze.py` and `train_controls.py` belong to versions later extended for Stage 3; their Stage-3 hashes match. `h2lib.py`, which implements the update, is unchanged since Stage 2. The time-stamp lines are reported as improperly formatted. The hashes are self-hosted. All three stages were frozen within 21 hours and their design decisions were delegated to the AI system, so treat the label as a record of intent and timing, not as an external guarantee. `.gitattributes` keeps these files byte-exact; do not convert their line endings.

## Use of AI assistance

Substantial AI assistance was used throughout this project. The question of whether an experience-acquired abstraction changes the process by which a later abstraction is formed, rather than only supplying information, is the author's. An AI system (Claude, Anthropic), working under the author's direction, audited the earlier studies, reviewed the literature, designed the study, wrote the preregistrations and code, ran the experiments and analyses, and drafted the paper. A separate instance of the same system reviewed the draft; that review led to the post hoc analyses E3 and E4. The preregistrations record that the author delegated the design decisions to the AI system. The author directed the work and is responsible for its content.

## Citation

See [`CITATION.cff`](CITATION.cff):

> Ramdan, S. (2026). *Head Start or Process? Decomposing How an Experience-Acquired Abstraction Shapes Later Learning in a Small Language Model.* Zenodo. [doi:10.5281/zenodo.23432800](https://doi.org/10.5281/zenodo.23432800)

## License

MIT (code). Qwen3-1.7B is Alibaba's (Apache 2.0) and is not included.
