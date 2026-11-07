# Gemma 4 12B dialect speech fine-tuning

LoRA fine-tuning of Gemma 4 12B Unified (`google/gemma-4-12B-it`, local snapshot in `MODEL_PATH` of `common.py`) on AI Hub Korean dialect speech. The model hears an utterance and answers in two lines:

```text
방언: <what was said, as spoken>
표준어: <the same sentence in standard Korean>
```

From stage 3 on (`--format region`), the answer starts with the speaker's accent zone, so the model also has to listen to accent and intonation, not only to the words:

```text
지역: <서울·경기 | 강원 | 충청 | 대구·경북 | 부산·울산·경남 | 전라 | 제주>
방언: <what was said, as spoken>
표준어: <the same sentence in standard Korean>
```

## How to run

Training runs in two stages. Gangwon finished downloading first, so it is trained first (started 2026-10-04 22:13); the other regions follow once their download is done.

```bash
cd training/gemma4-dialect
setsid nohup ./run_stage1_gangwon.sh > runs/stage1.log 2>&1 < /dev/null &
setsid nohup ./run_stage2_all.sh > runs/stage2.log 2>&1 < /dev/null &
```

Both can be started at once: stage 2 waits until the AI Hub download (118–122) and stage 1 have both finished. In each script, a step whose output already exists is skipped and training resumes from its last checkpoint, so after an interruption the same commands continue where they stopped.

| Stage | Step | Output |
| --- | --- | --- |
| 1 | Cut 25,000 + 300 Gangwon utterances | `data_118/` |
| 1 | Score the base model on Gangwon | `runs/base_eval_118/summary.json` |
| 1 | Train a LoRA adapter on Gangwon (1,563 steps, about 4.6 h) | `runs/lora_gangwon/final/` |
| 1 | Score it on Gangwon | `runs/lora_gangwon_eval/summary.json` |
| 2 | Cut 25,000 + 300 utterances each for Gyeongsang, Jeju, Chungcheong | `data_rest/` |
| 2 | Merge them with 7,500 Gangwon replay utterances (10%) | `data_all/` (82,500 train, 1,200 valid) |
| 2 | Score the base model on all four regions | `runs/base_eval/summary.json` |
| 2 | Continue the Gangwon adapter on `data_all` (5,157 steps, about 15 h) | `runs/lora_v1/final/` |
| 2 | Score it on all four regions and compare | `runs/lora_v1_eval/`, `runs/lora_v1_comparison.md` |

The replay keeps the second stage from overwriting what was learned on Gangwon; the final comparison includes Gangwon to check for that.

### Stage 3: accent zone (region format)

The first stage-3 run (`run_stage3_more.sh`, two-line answers on the 500,000 utterances of `data_more/`, 125,000 per region, none used in stages 1–2) was stopped on 2026-10-06 at step 4,949 of 31,239: its targets are text only, so it could not teach the model to hear accent or intonation. Its checkpoints stay in `runs/lora_v2/` but are not used.

`run_stage3_region.sh` replaces it. It continues `runs/lora_v1/final` with the `지역:` line in front of the answer:

| Step | Output |
| --- | --- |
| Label every utterance with its speaker's accent zone, from the birthplace and principal residence in the AI Hub label JSON (`prepare_origin.py`) | `data_more/train_origin.jsonl` (459,288 of 499,821 kept), `data_all/valid_origin.jsonl` |
| Add 22,000 unused Training-split utterances each of Seoul-Gyeonggi and Jeolla speakers, who are rare in `data_more` (about 2,700 and 1,600) | `data_zone_extra/` (44,000 clips, 55 h) |
| Build the intonation test set (see Evaluation) | `data_intonation/test.jsonl` |
| Train on the 503,288 utterances (31,456 steps, about 4 days) | `runs/lora_v2r/final/` |
| Score on the 1,199 validation utterances, compare with base and v1, then run the intonation test for base, v1 and v2r | `runs/lora_v2r_eval/`, `runs/lora_v2r_comparison.md`, `runs/intonation_comparison.md` |

```bash
cd training/gemma4-dialect && setsid nohup ./run_stage3_region.sh >> runs/stage3_region.log 2>&1 < /dev/null &
```

Since 2026-10-07 it runs under `keep_stage3_region.sh`, which starts the stage script again 5 minutes after it dies (it resumes from the newest checkpoint; the first run died on 2026-10-06 21:25 at step 2,495 under memory pressure) and gives up after 5 exits in a row without a new checkpoint:

```bash
cd training/gemma4-dialect && setsid nohup ./keep_stage3_region.sh > /dev/null 2>&1 < /dev/null &
kill -- -$(ps -o sid= -p $(pgrep -f 'bash ./keep_stage3_region.sh') | tr -d ' ')   # stop: wrapper, stage script and train.py
```

Why the speaker's zone and not the dataset: each AI Hub regional set also records speakers from elsewhere (the Chungcheong set has speakers born in Seoul and Daegu), and three of the four sets were recorded by the same company (솔트룩스; Gangwon by 인피닉). Labelling by speaker puts several zones inside each set, so the model cannot pass by recognising a recording setup. An utterance gets a zone only when its speaker's birthplace and principal residence fall in the same zone; Gyeongsang is split into its two main dialect areas, 부산·울산·경남 and 대구·경북.

Training data per zone: 강원 119,230, 제주 114,111, 충청 102,008, 부산·울산·경남 79,960, 대구·경북 39,633, 서울·경기 24,699, 전라 23,647. The Seoul-Gyeonggi and Jeolla utterances come from only about 440 and 230 recordings, so those two zones rest on fewer speakers than the others.

Progress: `tail -f runs/stage3_region.log`; training speed and ETA: `cat runs/lora_v2r/speed.json`.

`run_all.sh` is the earlier single-stage version (all four regions at once). It is kept for reference; do not run it alongside the stage scripts, since both use the GPU.

Outcome (2026-10-08): stopped after 6,500 steps. On the site's benchmark its CER was not better than v1's (mostly worse in a paired comparison), and on speech without dialect words its `지역:` line stayed near macro-F1 0.216 (7 zones): the zone came from the words, not the accent. The zone was therefore moved to a separate classifier on the Whisper encoder ([../accent-id](../accent-id)), and the later stages use the two-line format again.

### Stages 4–7 (2026-10-08 to 10-12): capacity helped, more of the same did not

Every stage continues the adapter that was served at the time, at lr 2e-5 with a cosine schedule. From stage 5 on, the previous adapter is merged into the weights (`train.py --merge-adapter`) and a new, larger LoRA is trained on top; `combine_adapters.py` then folds the stack into one adapter for the site, because the engine switches the adapter off for coaching and so cannot serve merged weights. Each `keep_stage*.sh` restarts its stage after a crash. CER as dialect / standard line.

| Stage | Script → output | Data (new utterances from `data_more` unless noted) | Setup | Result |
| --- | --- | --- | --- | --- |
| 4 | `run_stage4_v3.sh` → `runs/lora_v3` | `data_v3`: 64,000 (제주 25,600, 강원 16,000, 충청 16,000, 경상 6,400) | v1, r16, 4,000 steps planned | stopped at about 1,300 steps. On the 1,199 validation utterances checkpoint 500 improved Jeju (dialect 0.408 → 0.340) but lost Gyeongsang; on the independent test set no checkpoint differed from v1 |
| 5 | `run_stage5_r64.sh` → `runs/lora_r64`, folded with v1 into `runs/lora_r64c` (rank 80) | `data_r64`: 48,000 (12,000 per region) | v1 merged, new LoRA r64 / alpha 128, 3,000 steps (a first try at lr 1e-4 was stopped at 830 steps: validation loss 0.342 and 0.350 against v1's 0.281) | test, site decoding: 0.147 / 0.153 → 0.145 / 0.151 (p_better 0.98 / 0.96); served from 2026-10-11 00:52 |
| 6 | `run_stage6.sh` → `runs/lora_s6`, folded into `runs/lora_s6c` (rank 144) | `data_s6`: 64,000 (16,000 per region) | r64c merged, new LoRA r64 / alpha 128, 4,000 steps | test, site decoding: 0.145 / 0.151 → **0.140 / 0.147** (p_better 1.00 / 1.00); served since 2026-10-11 21:21 |
| 7 | `run_stage7.sh` → `runs/lora_s7` | `data_s7`: 308,363 that the served lineage has not seen (all of Jeju and Gangwon, part of Chungcheong and Gyeongsang), without clips of fewer than 2 or more than 8 syllables a second | s6c merged, new LoRA r128 / alpha 256, one pass (19,273 steps) with checks on 1,000 test clips at 5k, 10k and 15k steps | stopped before the first step (2026-10-12) |

Stage 6 first tested whether padding the cuts helps Jeju (`run_eval_pad.sh`, 0.3 s and 0.6 s more audio at both ends of the Jeju test clips): Jeju dialect CER 0.302 → 0.310 and 0.399, so it trained on clips cut as before. The output got longer: the added audio holds mostly the other speaker, not cut-off words.

## Results on the independent test set (2026-10-11)

`data_test/test.jsonl`: 1,000 utterances per region (80% with dialect words, 5.0 h) from AI Hub Validation recordings that neither `data_all/valid.jsonl` nor the site benchmark touched, at most 4 per recording (`run_eval_test.sh`). Corpus CER (all edits over all characters), 95% intervals by a blockwise bootstrap over recordings, p_better = probability of being better than the previous adapter in the paired bootstrap (`score_eval.py`). Decoded as the site does (`evaluate.py --guard`: the answer starts with `방언:`, a length-based token budget, a loop is cut after 8 repeats and the standard line generated again):

| Region | v1 dialect / standard | r64c | **s6c (served)** |
| --- | --- | --- | --- |
| 강원 | 0.163 / 0.166 | 0.161 / 0.163 | **0.151 / 0.156** |
| 경상 | 0.077 / 0.080 | 0.076 / 0.079 | **0.071 / 0.074** |
| 제주 | 0.308 / 0.334 | 0.302 / 0.329 | **0.295 / 0.322** |
| 충청 | 0.097 / 0.096 | 0.097 / 0.097 | **0.096 / 0.095** |
| all | 0.147 / 0.153 | 0.145 / 0.151 | **0.140 / 0.147** |

Without the guard the same v1 scores 0.191 / 0.164. Most of the difference comes from answers that leave the format (an empty dialect line, a thinking block, a repeated syllable), which the forced start prevents. Compare adapters with the guard, as the site serves them. The reports are in [results/eval_test_guard/](results/eval_test_guard/) (`report.txt`: v1 vs r64c, `report_s6c.txt`: s6c vs r64c), the plain-decoding ones in [results/eval_test/](results/eval_test/).

The 1,199 validation utterances alone were misleading once: stage 4's Jeju gain (p_better 0.98–1.00) and Gyeongsang loss (0.00) there both vanished on this set, so no decision rests on one set any more.

## Results after stage 2 (2026-10-05)

Final adapter `runs/lora_v1/final` (Gangwon stage, then all four regions), scored on 1,199 held-out utterances. Character error rate, lower is better:

| | dialect CER base | dialect CER tuned | standard CER base | standard CER tuned |
| --- | ---: | ---: | ---: | ---: |
| overall | 0.455 | 0.225 | 0.534 | 0.209 |
| 강원 | 0.426 | 0.216 | 0.514 | 0.188 |
| 경상 | 0.242 | 0.129 | 0.364 | 0.100 |
| 제주 | 0.860 | 0.425 | 0.873 | 0.424 |
| 충청 | 0.293 | 0.132 | 0.383 | 0.126 |

- 1,024 of 1,199 utterances improved on the standard-Korean line by more than 5 points of CER; 58 got worse.
- Gangwon after stage 1 alone: 0.208 dialect / 0.202 standard. After stage 2: 0.216 / 0.188, so the 10% replay kept the first region from being forgotten.
- Answers in the two-line format fell from 99.1% (base) to 97.0%; the failing 36 answers are worth a look before deployment.
- Jeju is still the weakest region (0.42 CER), but the base model barely understood it (0.86–0.87).

## Data

- **Regions:** Gangwon (118), Gyeongsang (119), Jeju (121) and Chungcheong (122), 25,000 training utterances each, about 100,000 in total (plus 7,500 Gangwon replay in stage 2). Jeolla (120) is left out: its labels are plain text without timestamps or standard-Korean forms, so its utterances cannot be cut out of the 10–20 minute recordings.
- **Sampling:** 80% of the chosen utterances contain at least one dialect word (`isDialect`), 20% are plain standard Korean, so the model also learns to leave standard speech unchanged. Utterances are 1–20 s long. Those with anonymization or noise tags (`#`, `@`, brackets) are skipped.
- **Validation:** 300 utterances per region from AI Hub's own Validation split, never used for training.

## Training setup

| Setting | Value |
| --- | --- |
| Method | LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections |
| Also trained | the audio projection layer (`embed_audio.embedding_projection`, about 2.5 M parameters) |
| Batch | 2 utterances × 8 accumulation steps = 16 per step (fastest split on the GB10, see below) |
| Optimizer | AdamW, lr 1e-4, cosine schedule, 3% warm-up, bf16, gradient checkpointing |
| Loss | answer tokens only; the prompt and audio are masked |

Pilot runs on 4,000 utterances measured these step times for 16 utterances per step: 2 × 8 took 10.6 s, 1 × 16 took 11.6 s, 4 × 4 took 12.0 s and 8 × 2 took 18.6 s (larger batches add padding). Peak memory was 27–42 GB. The 100-step 4 × 4 pilot lowered the loss from 0.70 to about 0.33. At 10.6 s per step, one epoch over 100,000 utterances (6,250 steps) takes about 18 hours.

The training sequence is the exact inference prompt (including the empty thinking block that the chat template adds with `add_generation_prompt=True`) followed by the answer and `<turn|>`, so training and inference see the same tokens.

## Evaluation

`evaluate.py` reports the character error rate (spaces and punctuation ignored) of the dialect transcript and of the standard-Korean line, overall, for utterances that contain dialect, and per region. It generates in batches of 16 with left padding. Compared with one-at-a-time decoding, 5 of 8 test outputs were identical and the rest differed by a word, which is bf16 numerical noise; base and adapter are scored the same way, so the comparison is fair.

### Error analysis (2026-10-12)

`error_analysis.py <run>` charges every character error of an evaluate.py run on data_test to one cause and
writes `<run>/error_analysis.{md,json}`. For lora_s6c (runs/eval_test_guard/s6c; dialect line CER 0.140,
standard 0.147):

| cause | share of the text | error rate inside it | share of the CER |
|---|---|---|---|
| ordinary words in the middle of a sentence | 68 % | 0.108 | 52 % |
| first and last word (and text added before or after) | 25 % | 0.192 | 34 % |
| dialect words (differ from the standard reference) | 6 % | 0.251 | 11 % |
| fillers (뭐~ 그~ 어~) and repetition loops | 1 % | | 3 % |

- Jeju is the outlier everywhere (middle words 0.21, first/last 0.41) and 58 % of its test clips are shorter
  than 3 s (CER 0.366; 3-6 s 0.274; 6 s+ 0.214). The other regions have 12-19 % short clips.
- 1.4 % of the clips do not hold the speech their label says (the answer matches neighbouring utterances of the
  same recording: label times off), 3.8 % in Jeju; without them the CER is 0.133 / 0.140.
- By loudness at the ends, 55 % of the Jeju clips start or end inside speech (4-10 % elsewhere), but intact Jeju
  clips are still at 0.269, and padding the cuts made it worse (runs/eval_pad).
- The conversion itself is nearly solved: when the dialect line is perfect, the standard line has CER 0.012.
- So even with perfect boundaries, dialect words and labels the CER would sit near the middle-word rate (0.10-
  0.11): a single-digit CER needs better recognition of ordinary speech, above all in Jeju and Gangwon.

### Intonation test

The CER above ignores punctuation, and the stage 1–2 targets are text only, so it cannot show whether the model hears intonation. `prepare_intonation.py` builds `data_intonation/test.jsonl` (4,410 clips, 3.2 h) from AI Hub's Validation split (never trained on, disjoint from `data_all/valid.jsonl`), using utterances whose words alone do not give the answer. "Zone" is the speaker's accent zone, as in stage 3.

| Subset | What it holds | What `score_intonation.py` reports |
| --- | --- | --- |
| `question` (1,465) | The same words said as a question and as a statement in the same regional set ("그래?" / "그래."), equal numbers of each, at most 600 per set | Balanced accuracy of a final "?" in the transcript; 50% means intonation is ignored |
| `region` (1,680) | The same plain sentence (two or more words, no dialect words) said in three or more zones, one utterance per zone | CER per zone, and the accent penalty: CER minus the other zones' average on the same sentence |
| `origin` (1,265) | Plain utterances (two or more words, no dialect words), 200 per zone (Jeolla 65), at most 5 per recording | With the region format, how often the `지역:` line names the zone, here and on `region`; 1 in 7 by chance |

```bash
./run_intonation_eval.sh   # base, v1 and v2r (once it exists); about an hour of GPU per model
```

Caveat: the labelers' use of "?" differs by region. Among utterances with a question word and a question-like ending, 54% are marked "?" in Gangwon, 25% in Gyeongsang, 20% in Jeju and 10% in Chungcheong, so some "statements" in the `question` subset are unmarked questions, and a model can score points by copying a region's labelling habit. Compare models on the same set rather than reading the numbers as absolute; Gyeongsang has the fewest pairs.

## Files

| File | Purpose |
| --- | --- |
| `common.py` | prompt, audio loading, batching, answer parsing, CER (the site's engine imports it) |
| `prepare_data.py` | sampling and audio cutting (`--exclude`, `--exclude-recordings`, `--max-per-recording`, `--pad`, `--only`) |
| `train.py` | LoRA training (`--max-steps 100` for a pilot run, `--init-adapter`, `--merge-adapter`, resumes from checkpoints) |
| `evaluate.py` | scoring (`--adapters` scores several on one loaded model, `--guard` decodes as the site does) |
| `score_eval.py` | corpus CER, jamo CER, near CER, recording-block bootstrap intervals and p_better |
| `combine_adapters.py` | folds a stack of LoRAs (earlier ones merged during training) into one adapter |
| `error_analysis.py` | CER broken down by cause, region, clip length and label check |
| `prepare_origin.py` | speaker accent zones for the `지역:` line, and the extra Seoul-Gyeonggi / Jeolla utterances |
| `prepare_intonation.py`, `score_intonation.py`, `run_intonation_eval.sh` | intonation test set and its scoring |
| `run_stage1_gangwon.sh`, `run_stage2_all.sh` | stages 1 and 2 |
| `run_stage3_more.sh`, `run_stage3_region.sh` | stage 3, two-line and region format |
| `run_stage4_v3.sh` … `run_stage7.sh` | stages 4–7 |
| `keep_stage*.sh` | restart a stage after a crash (it resumes from the newest checkpoint) |
| `run_eval_full.sh`, `run_eval_rest.sh` | v1 and the stage-4 checkpoints on the 1,199 validation utterances |
| `run_eval_test.sh`, `run_eval_guard.sh`, `run_eval_pad.sh` | the independent test set: plain, site decoding, padded Jeju clips |
| `run_all.sh` | the earlier single-stage pipeline (not used) |
| `align_jeolla.py`, `refine_jeolla.py`, `verify_jeolla.py` | Jeolla pilot (2026-10-09, not used): Whisper segment times + character alignment, phone CTC refinement, re-transcription check. The Whisper step took about 16 h per source file, and the data was deleted |
| `results/` | the small report files of `runs/` (same paths): summaries, comparison reports, error analysis, training settings and speed |

Not in Git: `data_*/` (manifests and cut clips of AI Hub audio), `runs/` (adapters, checkpoints, predictions, logs), `pilot_data/`.
