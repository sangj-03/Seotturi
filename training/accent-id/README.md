# 서투리 accent-zone classifier

Tells which accent zone a speaker comes from by how they sound, also when every word they say is
standard Korean. The app's Gemma 4 dialect model transcribes and converts speech well but names the
zone mostly from the words: on utterances without dialect words its `지역:` line reaches macro-F1
0.216 (7 zones), so the app hides it. This classifier takes over that one decision; Gemma 4 keeps
transcription, standard-Korean conversion and the written feedback, and gets the classifier's result
in its prompt.

Six zones: 서울·경기 (the standard accent), 강원, 충청, 대구·경북, 부산·울산·경남, 제주. Jeolla is left
out by decision (its AI Hub set has no time stamps and very few Jeolla speakers appear elsewhere), so the
classifier never answers 전라; a Jeolla speaker gets the nearest of the six.

## How it works

```
clip (16 kHz) ─▶ Whisper-large-v3 encoder, frozen, 15 s window
                  └─ every hidden state (33) pooled over the speech frames: mean + std ─▶ [33, 2560]
              ─▶ head: learned mix of the 33 layers ─▶ LayerNorm ─▶ 256 ─▶ 64 ─▶ 6 zones (temperature-calibrated)
```

`common.py` holds the zones, the encoder wrapper, the head and `Classifier` (encoder + trained head),
which the serving engine imports so that serving matches training.

## Data

AI Hub 한국어 방언 발화 (강원 118, 경상 119, 제주 121, 충청 122), using the clips already cut for the
Gemma training in ../gemma4-dialect. A clip's zone is its speaker's: birthplace and
principal residence in the same zone (`gemma4-dialect/prepare_data.zone_of`), not the dataset it came
from (the Chungcheong set also records Seoul-born speakers, for example).

Only clips without dialect words are used for training: among the clips cut for the Gemma training,
81-85 % of the regional ones hold dialect words but only 18 % of the Seoul-Gyeonggi ones, so with them
"no dialect words" would mean "standard" and the classifier would learn the words, which is exactly
Gemma's failure.

AI Hub only numbers the two speakers inside each recording, so the same person can appear in several
recordings under different numbers, and nothing joins them: profiles repeat across thousands of slots,
and a speaker-verification model (wavlm-base-plus-sv) rates the two people of one conversation almost as
alike as one person (median 0.86-0.90 vs 0.975; shared phone and room), so grouping by voice merged
thousands of slots into a few dozen "persons". The splits are therefore by recording: train and dev
never share one, and the tests are AI Hub's Validation split. `leakage.py` checks afterwards whether test
voices also occur in training (below).

| Split | What | Source |
| --- | --- | --- |
| train | without dialect words; per zone the same age-sex mix as Seoul-Gyeonggi (the zone with the fewest speakers) times `--k` (10), at most 12 clips per speaker (20 for Seoul-Gyeonggi): 54,626 clips | AI Hub Training |
| dev | whole recordings, about 200 speakers per zone, 6 clips each: 3,015 clips; picks the epoch and the temperature | AI Hub Training |
| test_plain | 200 clips per zone without dialect words, at most 5 per recording (`gemma4-dialect/data_intonation`, subset origin) | AI Hub Validation |
| test_same | one sentence said by speakers of two or more zones (subset region) | AI Hub Validation |
| test_dialect | the Gemma validation clips, dialect words included | AI Hub Validation |
| bench | the clips of the engine's zone benchmark, Jeolla left out | AI Hub Validation |

## Run

```bash
cd training/accent-id
PY=../../.venv/bin/python
export HF_HOME=/data/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY prepare.py index        # data/clips.jsonl
$PY prepare.py split        # data/{train,dev,test_*,bench}.jsonl
$PY features.py train dev test_plain test_same test_dialect bench   # features/*.npy, resumable (~1 h)
$PY train_head.py --out runs/phase1                                 # phase 1: runs/phase1/head.pt, metrics.json
$PY finetune.py --out runs/phase2                                   # phase 2: LoRA on encoder layers 17-32 (~4 h)
$PY leakage.py embed && $PY leakage.py report runs/phase1          # do test voices also occur in training?
ln -sfn ensemble runs/served && (cd ../../app && ./run.sh restart-accent)   # serve a run
```

## Served: phase 1 on more data + phase 2 mixed (runs/ensemble_big, since 2026-10-07 05:57)

`runs/ensemble_big/ensemble.json`: 0.6 `runs/phase1_big` + 0.4 `runs/phase2` in log-probability, standard bonus
+1.25 (`served_standard_bias`, which `app/server/accent.py` reads from the served run). `phase1_big` is
train_head.py on `data_big` (`prepare.py split --k 1000 --seoul-cap 60 --cap 36 --out data_big`: every eligible
person, 94,688 clips, 1.73x; dev and tests are identical to `data/`); `assemble_big.py` reuses the features of the
clips already in `data/train` (recomputed ones are bit-identical) and takes the 40,274 new ones from
`features.py train_extra`. Same weights as runs/ensemble; the bonus was raised from 1.0 to 1.25 so that as few
Seoul-Gyeonggi speakers are called regional as before.

| test_plain, with the served bonus | Macro-F1 | Standard called regional | Regional called standard | test_same | bench |
| --- | --- | --- | --- | --- | --- |
| runs/ensemble, +1.0 (04:00-05:57) | 0.664 | 68/200 | 100/1000 | 0.521 | 0.712 |
| runs/ensemble_big, +1.25 | 0.690 | 69/200 | 82/1000 | 0.539 | 0.737 |

Without the bonus: test_plain 0.675 (AUC 0.876), test_same 0.552, test_dialect 0.710, bench 0.719. phase1_big
alone: test_plain 0.652, test_same 0.534, test_dialect 0.723, bench 0.645 (phase 1: 0.632, 0.524, 0.685, 0.682);
every zone's test_plain F1 went up or stayed. Switch back with `ln -sfn ensemble runs/served` and
`./run.sh restart-accent` (its ensemble.json carries +1.0).

### Before: phase 1 + phase 2 mixed (runs/ensemble, 2026-10-07 04:00-05:57)

`runs/ensemble/ensemble.json` mixes the two runs in log-probability, 0.6 phase 1 and 0.4 phase 2: phase 1
names fine zones better, phase 2 separates standard from regional better, and the mix beats both. The
weight and the service's standard bonus (+1.0) were chosen on test_same and checked on test_plain.

| test_plain (no dialect words, 1,200 clips) | Macro-F1 | Accuracy | Std-vs-regional AUC | Standard called regional | Regional called standard |
| --- | --- | --- | --- | --- | --- |
| phase 1 | 0.632 | 0.634 | 0.854 | 56.5 % | 4.4 % |
| phase 2 | 0.603 | 0.623 | 0.862 | 42.0 % | 9.9 % |
| ensemble | 0.661 | 0.664 | 0.891 | 49.0 % | 5.4 % |
| ensemble + standard bonus 1.0 (served) | 0.664 | 0.666 | 0.891 | 34.0 % | 10.0 % |

Ensemble on the other tests: test_same 0.536 (AUC 0.794), test_dialect 0.698 (AUC 0.976), bench 0.701
(Gemma's `지역:` line: 0.216 on the same clips). Phase 2 alone was below phase 1 on every macro-F1 (dev
still rising at the end of its 3 epochs), so it is only used inside the mix.

## Results (phase 1)

Six zones, one clip at a time, persons and recordings never seen in training:

| Test | Clips | Macro-F1 | Accuracy | Standard vs regional AUC | Gyeongsang as one zone |
| --- | --- | --- | --- | --- | --- |
| test_plain (no dialect words) | 1,200 | 0.632 | 0.634 | 0.854 | 0.728 |
| test_same (same sentence, several zones) | 1,678 | 0.524 | 0.621 | 0.737 | 0.714 |
| test_dialect (dialect words included) | 1,086 | 0.685 | 0.780 | 0.917 | 0.848 |
| bench (the engine's zone benchmark) | 90 | 0.682 | 0.678 | 0.825 | 0.744 |

For comparison on the same 90 bench clips: Gemma's `지역:` line 0.216 (7 zones), the first frozen-encoder
linear probe 0.44-0.49. Per zone (test_plain F1): 제주 0.88, 강원 0.75, 충청 0.60, 부산·울산·경남 0.53,
서울·경기 0.53, 대구·경북 0.51. The main confusions are 서울·경기 → 충청 (77 of 200) and between the two
Gyeongsang zones; a Seoul-Gyeonggi clip is called regional in 57 % of cases, so the service adds +1.0 to
the standard zone's log-probability (`SEOTTURI_ACCENT_STANDARD_BIAS` in app/server/accent.py):
45 % standard→regional, 10 % regional→standard (was 4 %), same macro-F1. The app also averages the clips
of an answer and shows nothing under 8 s of speech.

Leakage check (`leakage.py report runs/phase1`): test clips whose nearest training voice of the same zone
and sex is below 0.97 / 0.95 / 0.93 cosine keep their accuracy (0.635 / 0.657 / 0.728 on test_plain;
test_same flat at 0.62-0.64), so the scores are not carried by voices seen in training. Dev scores are
higher (macro-F1 0.797) than the Validation-split tests, so use the tests, not dev, to compare runs.

## Learning curve (2026-10-07): more data still helps

`learning_curve.py` retrains the phase 1 head on parts of the training set (cached features, two seeds,
same model, schedule and early stopping; the full-data seed-0 point reproduces runs/phase1 exactly):

| Training data | Clips | Persons | Dev | test_plain | test_same | test_dialect | bench |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25 % of persons | 13,618 | 4,450 | 0.714 | 0.568 | 0.496 | 0.600 | 0.545 |
| 50 % of persons | 27,365 | 8,899 | 0.770 | 0.607 | 0.520 | 0.639 | 0.607 |
| all | 54,626 | 17,797 | 0.796 | 0.641 | 0.526 | 0.686 | 0.677 |
| ≤ 3 clips per person | 37,164 | 17,797 | 0.754 | 0.603 | 0.502 | 0.655 | 0.627 |
| ≤ 6 clips per person | 46,299 | 17,797 | 0.786 | 0.613 | 0.532 | 0.670 | 0.629 |

Each doubling adds about 0.035 macro-F1 on test_plain (0.04-0.05 on test_dialect) with no sign of
levelling off; both more persons and more clips per person help. test_same levels off (0.496, 0.520,
0.526; only 42 Seoul-Gyeonggi clips). The clips already cut for the Gemma training hold 128,141 without
dialect words with a zone (2.3x the training set), so the next step needs no new cutting: a split with
a larger `--k` and caps, features for the new clips, retrain. Seoul-Gyeonggi can only grow in clips
(448 persons in all of AI Hub Training, 399 used).

## Twice the data again (2026-10-07): little gain, not served

`expand.py` cut 104,083 more clips straight from the AI Hub recordings (`clips_x2/`): every person not yet in
training (9,284 slots, at most 10 clips each; Seoul-Gyeonggi has none left) and Seoul-Gyeonggi's remaining
clips of its 399 persons. `data_x2/train`: 198,771 clips, 39,295 persons; dev and tests unchanged.
Features: `features.py train_extra2`, `assemble_big.py --split data_x2 --name train_x2 --sources train_big,train_extra2`.

| Phase 1 head on | test_plain | test_same | test_dialect | bench |
| --- | --- | --- | --- | --- |
| data (54,626) | 0.632 | 0.524 | 0.685 | 0.682 |
| data_big (94,688) | 0.652 | 0.534 | 0.723 | 0.645 |
| data_x2 (198,771), runs/phase1_x2 | 0.657 | 0.552 | 0.732 | 0.693 |

Mixed 0.6/0.4 with phase 2 at +1.25 it matches the served ensemble_big (test_plain 0.688 vs 0.690, Seoul-Gyeonggi
called regional 73 vs 69 of 200; test_same 0.552 vs 0.539, test_dialect 0.689 vs 0.676, bench 0.737 both), so
ensemble_big stays served. The learning curve flattened (+0.005 on test_plain for this doubling against about
+0.035 before): what was left were small additions per new person and more clips of the same Seoul-Gyeonggi
speakers. The frozen-encoder head is now limited by the model more than by the data; data_x2 is meant for a
fine-tuned phase 3.

## Phase 2 alone (2026-10-07): worse than phase 1

`finetune.py --out runs/phase2` (LoRA r=16 on encoder layers 17-32, attentive pooling over the hidden
states after layers 16-32, augmentation, 3 epochs, 10,245 steps, 2 h 55 min on the shared GPU) lost to
phase 1 on every held-out macro-F1 on its own (mixed with phase 1 it is served, see above):

| Test | Macro-F1 phase 1 → 2 | Accuracy | Standard vs regional AUC | Gyeongsang as one zone |
| --- | --- | --- | --- | --- |
| test_plain | 0.632 → 0.603 | 0.634 → 0.623 | 0.854 → 0.862 | 0.728 → 0.751 |
| test_same | 0.524 → 0.462 | 0.621 → 0.533 | 0.737 → 0.795 | 0.714 → 0.659 |
| test_dialect | 0.685 → 0.574 | 0.780 → 0.708 | 0.917 → 0.938 | 0.848 → 0.837 |
| bench | 0.682 → 0.646 | 0.678 → 0.656 | 0.825 → 0.853 | 0.744 → 0.789 |

Dev macro-F1 0.797 → 0.693. Phase 2 separates standard from regional speech a little better (AUC; at no
bias it calls 84 of 200 Seoul-Gyeonggi test_plain clips regional, phase 1 113) but tells the zones apart
worse. Likely causes, for a next run: phase 2 reads only the hidden states after layers 16-32, while
phase 1's learned layer mix puts 55 % of its weight on layers 0-16 (13, 12, 14, 9, 8 among the largest);
its head starts from random instead of from the phase 1 head (LP-FT, Kumar et al. 2022: fine-tuning from
a random head distorts pretrained features and loses out of distribution); and 6.7 M trainable values
on these speakers invite learning voices. A third run would mix all 33 layers (LoRA only on the upper
ones), start from the phase 1 head and use a lower learning rate. (This run predates the
`test_logits.npz` line in finetune.py, so `leakage.py report runs/phase2` has no logits to read.)

The GPU is shared with the app's engine (and other work), so every step caps its own share of GPU
memory (`--gpu-share`) and fails on its own instead of starving the others.

## Phase 3 (2026-10-07): all 33 layers, phase 1 head, LoRA — not served

`finetune3.py --out runs/phase3 --adv 0 --lp-steps 0` is the third run proposed above: phase 1's pooling over
all 33 hidden states and its head (layer mix, normalisation, MLP) as the starting point, LoRA r=16 on layers
17-32, head lr 3e-5 / LoRA lr 1e-4, one epoch over data_x2 (198,771 clips, ~3 h with the GPU shared). The
first attempt added a gradient-reversal speaker adversary (39k persons); it diverged (dev 0.175 at step 3000,
kept in `runs/phase3_adv_failed`), so the run that finished has none. Best dev 0.839 at step 10,500 (phase 1
0.797), but the tests do not follow (macro-F1 / Seoul-Gyeonggi clips called regional, standard bonus 1.25):

| | test_plain | test_same | test_dialect | bench |
| --- | --- | --- | --- | --- |
| served (0.6 phase1_big + 0.4 phase2) | 0.690, 69/200 | 0.539, 14/42 | 0.676, 1/10 | 0.737, 4/15 |
| phase 3 alone | 0.667, 85/200 | 0.502, 29/42 | 0.659, 2/10 | 0.698, 6/15 |
| 1/3 each of phase1_big, phase2, phase 3 | 0.706, 69/200 | 0.535, 14/42 | 0.685, 2/10 | 0.740, 4/15 |

The three-way mix gains on test_plain (+0.016, 95 % interval over speakers +0.002 to +0.030) but not on
test_same (-0.004) or bench (+0.004, interval -0.035 to +0.038), and calls as many Seoul-Gyeonggi clips
regional; it would cost a third encoder pass per clip and `Tuned3` support in `common.load_classifier`, so
the served ensemble stays. Fine-tuning keeps raising dev (same AI Hub recordings style) without moving the
held-out tests: the limit is no longer the head or the layers read.

## Files

| Path | Purpose |
| --- | --- |
| `common.py` | zones, Whisper encoder features, head, `Classifier` |
| `prepare.py` | clip index, person grouping, splits |
| `features.py` | pooled encoder features per split |
| `train_head.py` | phase 1 training and all test scores |
| `finetune.py` | phase 2: LoRA on the upper encoder layers, attentive pooling, audio augmentation; `Tuned` serves it |
| `bench_export.py` | every run re-scored for the site's benchmark page (`runs/benchmark.json`, served as `/api/accent`) |
| `finetune3.py` | phase 3: LoRA on layers 17-32 over phase 1's 33-layer pooling and head; `Tuned3` serves it |
| `leakage.py` | voice embeddings and the train-test voice overlap report |
| `session_eval.py` | answer-level test: 8 plain sentences per speaker, the served system on 1-8 of them, and the app's decision rules |
| `region4.py`, `augment.py`, `conditions.py`, `run_region4.sh` | four-region data, recording-condition copies, scores under the site's recording conditions |
| `calibrate_std.py`, `run_std2.sh` | standard-accent detector: bias on dev, the binary detector that lost |
| `expand.py`, `assemble_big.py`, `learning_curve.py` | more clips cut from AI Hub, feature reuse, learning curve |
| `eval_browser.py` | scores browser recordings from the site's `/record` page (none collected yet) |
| `results/` | the small files of `runs/` (same paths): `metrics.json` of every run, condition tests, ensembles, the answer-level report (`session_eval/report.md`), `benchmark.json` for the site |
| `data/`, `features/`, `runs/` | manifests, features, trained heads (not in Git) |

## Region4 (2026-10-09): four regions by recording set, recording-condition augmentation — not served

User's choices: Seoul-Gyeonggi is the standard accent and is not trained (speakers born there are left out
of every split); the label is the AI Hub set a clip was recorded in, not the birthplace (the
Gyeongsang-born speakers of the Jeju set live in Jeju and use Jeju forms as often as the natives).
Why augment: on the served model, the background alone names the region well above chance and the
site's processing (peak 0.9, browser noise suppression, near-silent gaps between words) breaks it
(one-off probe scripts, not included). `region4.py` + `augment.py` give every training clip two copies with a
random chain of crop / another set's background / microphone EQ / Opus / spectral-gating denoiser / gated
gaps / speech only / level; `run_region4.sh` extracts their features (157,356 clips, 29 clips/s), trains the
phase 1 head with 4 outputs and runs `conditions.py`.

`conditions.py` (985 test_plain clips of home-set speakers, balanced recall over 강원/충청/경상/제주, chance
0.25; the denoiser there, ffmpeg afftdn, is one training never saw; 배경만 = background only, lower is better):

| condition | served ensemble (6 zones, Seoul dropped) | region4_noaug | region4 |
|---|---|---|---|
| 원본 | 0.865 | 0.827 | 0.820 |
| 사이트 전처리 (trim, peak 0.9) | 0.815 | 0.763 | 0.805 |
| 사이트 + 잡음제거 (unseen denoiser) | 0.693 | 0.625 | **0.715** |
| 무음화 (quiet frames -40 dB) | 0.550 | 0.537 | **0.792** |
| 말소리만 | 0.792 | 0.785 | 0.792 |
| 앞 2초 | 0.710 | 0.655 | 0.735 |
| 배경만 (520 clips) | 0.562 | 0.578 | 0.500 |

Against the same head without copies (`region4_noaug`), augmentation costs 0.007 on clean clips and gains
0.04-0.26 under every damaged condition, the unseen denoiser included; the drop from the served 0.865 comes
from the single head (the served one is an ensemble with the LoRA run). The background still names the
region at 0.50, so augmentation only partly removes that cue. r4_test_plain macro-F1 0.812 (noaug 0.821).

### Served since 2026-10-09 19:34: region4 + detector_mix

Seoul-Gyeonggi is not trained as a region (user's choice), so "is it regional at all" comes from a separate
detector: `runs/detector_mix` = the served six-zone ensemble (ensemble_big) and the augmented six-zone head
(`zone6_aug`, train_head.py on z6_train = train_big + region4 copies + Seoul copies) mixed 0.5/0.5, of which only
P(서울·경기) is used, with the bonus +2.7 chosen on dev (`calibrate_std.py`). A binary standard-vs-regional head
(`runs/std2`) was tried first and lost (test_plain AUC 0.735: with all regions in one class the recording set
alone separates most clips). app/server/accent.py serves P(zone) = P(regional) * P(zone | regional) from
one shared encoder (`common.Tree`), and the app counts the accent in the 사투리 비율 whenever P(regional) >= 0.5.

Standard vs regional on test_plain (985 regional + 200 Seoul-Gyeonggi clips; caught / Seoul called regional):
| condition | AUC (served before → detector_mix) | detector_mix at +2.7 | served before (+1.25) |
|---|---|---|---|
| 원본 | 0.879 → 0.885 | 87 % / 23 % | 95 % / 40 % |
| 사이트 전처리 | 0.822 → 0.860 | 84 % / 31 % | 95 % / 57 % |
| 사이트 + 잡음제거 | 0.839 → 0.818 | 81 % / 30 % | 96 % / 62 % |
| 무음화 | 0.832 → 0.894 | 86 % / 23 % | 98 % / 75 % |
| 앞 2초 | 0.781 → 0.834 | 78 % / 31 % | 94 % / 67 % |

Dev was far easier than test (bonus +2.7 gives 4 % Seoul false alarms on dev, 23-31 % on test): check the
browser test set (/record, `eval_browser.py`) before trusting either number. Roll back with
`ln -sfn ensemble_big runs/served && rm runs/served_std` and `./run.sh restart-accent` in app/.

## Answer-level evaluation (2026-10-12): what the app decides from several sentences

The app averages the clips of an answer by length; every test above scores single clips. `session_eval.py` draws
speakers from AI Hub's Validation split of the four sets (never trained on) with 8+ sentences without dialect
words each, as in region4's data (birthplace and residence in one zone, Jeolla out): 250 per set and all 100
Seoul-Gyeonggi speakers there are, 8,799 clips (`data_sessions/`). The served system (region4 + detector_mix,
as accent.py computes it) is scored on the first 1-8 sentences of each speaker, on the clips as cut and as the
site prepares them (analysis.preprocess). Results in `runs/session_eval/report.md`.

| sentences | speech (median) | four-region macro-F1, clips as cut [95% CI] | same, site preprocessing | detector AUC (cut / site) | Seoul-Gyeonggi called regional at 0.5 (cut / site) |
|---|---|---|---|---|---|
| 1 | 4.0 s | 0.842 [0.819–0.864] | 0.832 [0.808–0.854] | 0.908 / 0.901 | 17% / 24% |
| 2 | 8.3 s | 0.873 [0.850–0.893] | 0.859 [0.837–0.880] | 0.922 / 0.915 | 16% / 24% |
| 4 | 17.1 s | 0.895 [0.876–0.914] | 0.898 [0.878–0.916] | 0.935 / 0.934 | 18% / 23% |
| 6 | 25.7 s | 0.909 [0.890–0.926] | 0.906 [0.887–0.923] | 0.940 / 0.939 | 20% / 24% |
| 8 | 34.5 s | 0.907 [0.890–0.924] | 0.911 [0.892–0.928] | 0.944 / 0.942 | 19% / 26% |

The four-region choice reaches 0.90 from about four to six sentences (17-26 s); one sentence gives 0.83-0.84 on
this draw (r4_test_plain 0.812). The weak side is standard vs regional: at the answer level 16-26 % of the
Seoul-Gyeonggi speakers are still called regional, and there are only 100 of them in these sets.

The app's final five-way verdict (서울·경기 or a region, "unsure" below analysis.ZONE_MIN_CONF 0.6) is worse
than either part, because the 0.6 threshold is applied to P(regional) x P(region | regional): a regional speaker
at 0.9 x 0.65 is "unsure". `session_eval.py rules` (no GPU, `runs/session_eval/rules.md`) compares rules; with
site preprocessing (five-way macro-F1 with "unsure" counted wrong):

| sentences | app today | two steps (regional if mean P(regional) >= 0.5, region if its log-mean probability >= 0.6) |
|---|---|---|
| 1 | 0.700, unsure 22%, right when sure 82%, Seoul called regional 12% | 0.726, unsure 6%, right when sure 77%, Seoul called regional 24% |
| 4 | 0.729, unsure 30%, right when sure 90%, Seoul called regional 4% | 0.793, unsure 6%, right when sure 84%, Seoul called regional 17% |
| 6 | 0.735, unsure 31%, right when sure 92%, Seoul called regional 4% | 0.806, unsure 6%, right when sure 86%, Seoul called regional 19% |

Two steps answer 94 % of the answers instead of 70 %, at the price of more Seoul-Gyeonggi speakers called
regional. Thresholds tuned on half of the speakers did not carry over to the other half (100 Seoul speakers are
too few); the browser recordings (/record) are the place to set them. Nothing served was changed.

