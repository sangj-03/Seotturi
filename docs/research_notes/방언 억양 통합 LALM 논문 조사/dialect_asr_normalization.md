# Prior work on "verbatim dialect transcript + standard-language conversion in one speech model" (for the Korean dialect coaching LALM)

Scope note: notes compiled 2026-10-14 for the project's dual-output model ("방언: <verbatim>" / "표준어: <standard>", plus a planned region/accent output). Verification tags:
- **[read]**: the number was read in the paper's own text (PDF/HTML) in this session.
- **[abstract]**: only the arXiv abstract or landing page was read.
- **[summary]**: the number comes from a search-engine digest or secondary page and was not checked against the paper. Treat it as unverified.
- **[builder]**: self-reported by the AI Hub dataset builder on the dataset page. Not peer-reviewed, and the test split and normalization are not documented.

---

## Q1. Korean dialect ASR and Korean dialect-to-standard conversion (speech or text), especially on AI Hub dialect corpora

### Takeaway
I found no peer-reviewed paper that takes Korean dialect *speech* to standard Korean (or to verbatim plus standard) end to end, and none that fine-tunes a speech LLM on AI Hub dialect speech. Published Korean work falls into three groups:
- AI Hub builder baselines, which report Conformer ASR CER of 5–6% and Transformer dialect MT BLEU of 0.79 (Jeju) and 0.90 (Gyeongsang).
- Text-only dialect↔standard MT: Kakao JIT, JKCS 2022, Applied Sciences 2025 and DIA-REFINE 2025.
- Speech dialect ID: XLS-R, F1 0.868.

Jeju is consistently the hardest region to *convert*, with the lowest MT BLEU of any region. It is also the easiest region to *identify*: text F1 0.98, speech recall 0.93.

### Cited Findings
**AI Hub builder-reported baselines [builder]**
- **한국어 방언 발화(경상도)** (dataSetSn=119): 3,000+ h, 2,000+ speakers aged 10s–60s, about 500k transcribed items. The JSON has `form` (dialect transcription), `standard_form`, `dialect_form`, and per-eojeol `eojeolList` entries with `eojeol` (dialect), `standard` and an `isDialect` boolean. Reported results: Conformer (CTC/attention) **CER 5.28%** against a 12% target, **WER 14.68%** against 24%; Transformer encoder-decoder MT **BLEU 0.9043** against 0.7. Built 2020, v1.0 released 2021-06-30. — [AI Hub 경상도](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=119)
- **한국어 방언 발화(제주도)** (dataSetSn=121): 3,000+ h, 2,000+ speakers, 500k+ texts. The same label schema, with dual notation such as "(집드레)/(집으로)". Transformer MT **BLEU 0.7879** against a 0.7 target; GPT-2 perplexity 21.94. **The page lists no ASR metric for Jeju.** Built 2020. — [AI Hub 제주도](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=121)
- **중·노년층 한국어 방언 데이터(강원도, 경상도)** (dataSetSn=71517): Gangwon 801.4 h with 142,374 label files; Gyeongsang 1,202.9 h with 236,563. Labels include a pronunciation transcription, a dialect-notation transcription and a standard-notation substitution (표준표기치환문). Conformer **CER 6.24%** against an 18% target, **sWER 12.52%** against 35%. Built 2022, released 2023-07-26. — [AI Hub 중·노년층 강원/경상](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71517)
- The AI Hub pages do not say which direction the "machine translation" baseline runs (dialect→standard or standard→dialect). — same pages

**Text dialect↔standard MT**
- Park, Choe, Ham (Kakao Brain), "Jejueo Datasets for Machine Translation and Speech Synthesis", LREC 2020 [read via arXiv HTML]:
  - **JIT parallel corpus:** 170,356 pairs (160,356 / 5,000 / 5,000).
  - **JSS speech set:** 10,000 utterances, about 13 h 47 m, single speaker.
  - **MT setup:** Transformer, SentencePiece BPE, 4k vocabulary best.
  - **Test BLEU:** copy baseline 24.45 (jje→kor) and 24.44 (kor→jje). **JIT jje→kor 67.70** (dev 69.35); kor→jje 43.31. Adding Korean Wikipedia "copy" pairs gave 67.94 / 44.19 (≤ about +0.9 BLEU).
  - **Why jje→kor is easier:** Jejueo speakers often use Korean words; 아버지 appears 530 times against 332 for 아방.
  - **TTS:** DCTTS trained on JSS; Hangul Jamo input gave the lowest MCD, 14.32.
  - — [arXiv 1911.12071](https://arxiv.org/html/1911.12071v1), [ACL Anthology](https://aclanthology.org/2020.lrec-1.318.pdf)
- 임상범, 박찬준, 양영욱, "언어적 특성과 서비스를 고려한 딥러닝 기반 한국어 방언 기계번역 연구", *Journal of the Korea Convergence Society* 13(2):21–29, 2022. Uses AI Hub dialect data by province, a Transformer with a copy mechanism, and compares one-to-one against many-to-one models. The single many-to-one model for all dialects "achieved significantly high performance" and the copy mechanism helped. **BLEU tables not retrieved** (PDF host refused the connection). [abstract] — [KoreaScience](https://koreascience.kr/article/JAKO202209542033704.page)
- Park, Yu, Kim (Chung-Ang Univ.), "Steering LLMs toward Korean Local Speech: Iterative Refinement Framework for Faithful Dialect Translation" (DIA-REFINE), arXiv 2511.06680 v3 (Mar 2026) [read]:
  - **Setup:** AI Hub dialect corpus, *standard→dialect* only, for Jeolla, Gyeongsang and Jeju. Gangwon and Chungcheong were excluded as "close to standard". 300 test sentences per dialect, each ≥30 characters.
  - **Text dialect classifier F1:** Jeju 0.9803, Jeolla 0.9228, Gyeongsang 0.9107, Standard 0.9346.
  - **BLEU bias:** a near-copy output scores **52.54 BLEU**, while a genuine dialect rendering scores **27.78**. HyperCLOVAX zero-shot gets BLEU 47.64 with TDR 0.03, meaning it output standard Korean.
  - **Proposed metrics:** Dialect Fidelity Score (DFS) and Target Dialect Ratio (TDR).
  - **Method:** in-context learning with 10 BM25-retrieved examples, then classifier-verified refinement with up to 2 retries.
  - — [arXiv 2511.06680](https://arxiv.org/html/2511.06680)
- Junho Park and Seong-Bae Park (Kyung Hee), "Low-Resourced Alphabet-Level Pivot-Based Neural Machine Translation for Translating Korean Dialects", *Applied Sciences* 15(17):9459, 2025. It describes the AI Hub corpus as standard–dialect pairs from five regions. **Numbers not retrieved** (HTTP 403). — [MDPI](https://www.mdpi.com/2076-3417/15/17/9459)
- Changwon National Univ., "복사 매커니즘을 이용한 한국어-제주어 기계번역": reports 68.54 for Jeju→Korean on its evaluation data. Whether that is BLEU is unconfirmed. [summary] — [PDF](https://air.changwon.ac.kr/wp-content/uploads/2022/03/Machine-Translation-of-the-Korean-Jeju-Language-using-the-Copy-Mechanism.pdf)
- "오픈소스 거대 언어모델을 활용한 제주 방언의 기계 번역 연구" (IEIE 2024): Polyglot fine-tuned on Jeju–standard pairs, 78% win rate. [summary] — [DBpia](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11891168)

**Dialect identification from speech**
- 이주영, 김선희, 정민화 (SNU), "wav2vec 2.0 XLS-R을 활용한 한국어 방언 자동 식별", *Phonetics and Speech Sciences* 17(3):83–91, 2025 [read]:
  - **Data:** AI Hub 중·노년층 방언 data, men aged 50+. Train: 3,126 speakers / 324,957 samples. Test: 180 speakers / 18,000 samples.
  - **Classes:** six (Central, which merges Chungcheong and Gangwon; Gyeongnam; Gyeongbuk; Jeonnam; Jeonbuk; Jeju). Only the first 5 s of each utterance is used.
  - **Model:** XLS-R-300m, first fine-tuned for Korean phoneme recognition (PER 3.88%), then for dialect classification.
  - **Results:** macro F1 **0.868** against 0.738 for the eGeMAPS Bi-LSTM baseline. Recall: Jeju 0.85→**0.93**, Gyeongbuk 0.66→0.88, Jeonnam 0.67→0.89, Gyeongnam 0.69→0.74 (lowest).
  - **Confusions:** every dialect is confused mostly with Central; Central and Jeonnam are confused with each other (shared intonation).
  - — [eksss.org](https://www.eksss.org/archive/view_article?pid=pss-17-3-83)

**Korean ASR general and adjacent work**
- 윤정현, Helin Yin, 구영현 (Sejong), "대규모 언어 모델을 사용한 한국어 음성 인식 오류 보정 연구", IEIE Summer Conference 2024, pp. 2494–2498. LLM post-correction gave a "13.7% reduction in WER and 6.2% reduction in CER". It is not stated whether these are relative, and neither the ASR model nor the LLM is named. No dialect data. [abstract] — [DBpia](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11891057)
- Whisper fine-tuned on about 1,000 h of KsponSpeech improved Korean CER, but **English performance deteriorated after fine-tuning**, which is a forgetting signal. *Phonetics and Speech Sciences* 15(3):75, 2023. [summary] — [eksss.org](https://www.eksss.org/archive/view_article?pid=pss-15-3-75)
- Korean speech-LLM benchmarks from 2026, KoALa-Bench and KVoiceBench/KOpenAudioBench/KMMAU, show no dialect coverage in their abstracts. [summary] — [arXiv 2604.19782](https://arxiv.org/html/2604.19782), [arXiv 2605.27984](https://arxiv.org/abs/2605.27984)
- Community project: Whisper fine-tuned for Jeju speech-to-text, plus a text MT model with `[제주]` / `[표준]` direction tokens. Self-reported BLEU 0.76 (Jeju→standard) and 0.5 (standard→Jeju). [summary, unverified student project] — [GitHub junhoeKu/Jeju-Translation](https://github.com/junhoeKu/Jeju-Translation)

### Inferences
- The project's line-level CER (0.140 dialect / 0.147 standard) **cannot be compared** with AI Hub's Conformer CER of 5.28–6.24%. Those are builder-internal splits with undocumented normalization, a possible speaker overlap, and a CTC/attention model trained on single regions. They are at most a loose sanity reference.
- Jeju has the lowest *text-only* AI Hub MT BLEU (0.79 against 0.90 for Gyeongsang). So at least part of the project's Jeju gap (CER about 0.30) is likely **lexical and morphological divergence in the dialect→standard mapping itself**, not acoustics alone.
- **Copy-bias warning (from DIA-REFINE and Kakao):** for Gangwon and Chungcheong, where dialect ≈ standard, simply copying the dialect line into the standard line already scores well. Report a per-region **"copy baseline" CER**, the CER between the gold dialect line and the gold standard line. It shows how much real conversion the model performs, and it upper-bounds what a lazy model gets.
- AI Hub's `standard_form` is an eojeol-aligned *substitution*, not a free translation. Our "standard line" task is therefore closer to **dialect normalization** (ArchiMob-style) than to Swiss-German-style speech translation. Character-level or lexicon-driven normalizers become usable baselines or second passes (see Kuparinen et al. in Q2).
- For the planned accent output: Jeju should be easy. Gyeongnam, and the Central-like regions (Gangwon, Chungcheong) against standard Korean, are the likely confusions.

### Gaps
- I found no Korean paper on end-to-end dialect speech → standard text, dual-output decoding, or speech-LLM fine-tuning on AI Hub dialect speech (searched arXiv, ACL, ISCA, KCI/KoreaScience/DBpia via web search). Non-indexed Korean conference proceedings (KCC/KSC, 한국음향학회) could not be searched directly.
- BLEU tables for 임상범 et al. (2022) and Park & Park (2025) were not retrieved.
- The AI Hub pages for Gangwon (118), Jeolla (120) and Chungcheong (122) were not fetched. The MT direction and the test-split definitions are unknown.

---

## Q2. Analogous work in other languages; which ones train one model for both verbatim and normalized output, and what they found

### Takeaway
Most dialect work commits to a single target:
- Swiss German and South Tyrolean are framed as *speech translation into Standard German*, reaching BLEU 64–86 after fine-tuning.
- Norway released *separate* verbatim and semantic Whisper checkpoints, not one model with task tokens.
- Chinese dialect corpora mostly transcribe into Mandarin characters.

The closest analogues to our dual-output design are these:
- **WenetSpeech-Wu:** one LALM does Wu ASR, Wu→Mandarin translation and speaker attributes. The joint model costs only +0.38 CER on ASR and reaches AST BLEU 53.1, against 33.3 for Qwen3-Omni.
- **Mitsumori et al. (IEEE SLT 2026):** tagged `<ASR>/<standard>/<en>` outputs. One intermediate step helps; two intermediate steps hurt.
- **Imaizumi et al. (Japanese ASR+DID):** emitting the dialect label *after* the transcript beats emitting it first.

### Cited Findings
**Swiss German → Standard German**
- Plüss et al., **SDS-200** (arXiv 2205.09501, LREC 2022): about 200 h, about 4,000 speakers, Swiss German speech with Standard German text. Baseline 30.3 WER / 53.1 BLEU; fine-tuned XLS-R **21.6 WER / 64.0 BLEU**. [abstract] — [arXiv](https://arxiv.org/abs/2205.09501)
- Plüss et al., **STT4SG-350** (arXiv 2305.18855, 2023): 343 h, 316 speakers, all dialect regions balanced. Speakers read Standard German sentences aloud in their dialect, and the test set uses the same sentences for every region. Baseline **BLEU 74.7**. [abstract] — [arXiv](https://arxiv.org/abs/2305.18855)
- Timmel, Paonessa, Vogel, Perruchoud, Kakooee, "Fine-tuning Whisper on Low-Resource Languages for Real-World Applications", arXiv 2412.15726 v3 (Apr 2025) [read]:
  - **Training data:** SDS-200 50 h, STT4SG-350 276 h, SPC 176 h, SRG pseudo-labeled TV 406 h.
  - **Zero-shot Whisper large-v2:** STT4SG-350 WER 22.41 / BLEU 64.13.
  - **Final model:** STT4SG-350 **WER 12.11 / BLEU 78.08**; SDS-200 16.70 / 72.69; SPC 20.98 / 68.34.
  - **Standard German kept:** Common Voice 13 German WER **6.42 against 6.53** for the original, because 125 h of CV-de was mixed in to "mitigate language forgetting".
  - **Segmentation forgetting:** training only on padded sentence-level clips gave the best STT4SG BLEU (77.38) but broke long-form segmentation (SubER 51.34 against 30.63 for the original).
  - — [arXiv 2412.15726](https://arxiv.org/pdf/2412.15726)
- Dolev, Lutz, Aepli, "Does Whisper Understand Swiss German? An Automatic, Qualitative and Human Evaluation", arXiv 2404.19310 [read via HTML; the VarDial 2024 venue comes from the ACL Anthology URL]:
  - **Whisper large-v3 zero-shot WER / BLEU:** STT4SG-350 0.230 / 63.1, SwissDial 0.230 / 61.0, SPC 0.295 / 57.0. A fine-tuned XLS-R reaches 0.153 / 72.2 on STT4SG-350.
  - **Human ratings:** meaning 4.36/5, fluency 4.39/5.
  - **Typical errors:** Whisper drops modal particles and conjunctions, renders tense inconsistently, and hallucinated in 4 of 16 whole conversations.
  - **Output targets:** lists three possible targets for Swiss German ASR: dialectal writing, normalized writing, and Standard German translation.
  - — [arXiv HTML](https://arxiv.org/html/2404.19310v2), [ACL Anthology](https://aclanthology.org/2024.vardial-1.3.pdf)
- Nigmatulina, Kew, Samardžić, "ASR for non-standardised languages with dialectal variation: the case of Swiss German", VarDial 2020 (ArchiMob, Kaldi, 14 varieties). Normalized output wins on plain WER (29.39%). Dialectal output wins only under the flexible FlexWER metric, and the normalized system is weaker at character level. [summary] — [ACL Anthology](https://aclanthology.org/2020.vardial-1.2)
- Akeret, "Subtitle-Aligned Fine-Tuning of Whisper for Swiss German ASR: Benchmark Contamination, Convention Mismatch, and an Honest Baseline at 25.6% WER (13.8% cWER)", arXiv 2606.07608 (May 2026) [read via HTML]:
  - **Data:** 1,367 h of broadcast speech with Standard German subtitles.
  - **Results:** Whisper large-v3 zero-shot 28.56% WER. LoRA (r = 160, **α = 32**, all attention and FFN layers in both encoder and decoder, lr 1e-5) reaches 25.32–26.28% WER; full fine-tuning reaches 25.60% WER / **13.8% cWER**.
  - **Convention mismatch:** about **11.3–11.9 points of WER are semantically correct stylistic variants**: tense, word order, number format, ß vs. ss.
  - **Contamination:** prior claims such as 17.5% WER were contaminated by test data.
  - — [arXiv 2606.07608](https://arxiv.org/html/2606.07608v1)
- **South Tyrolean → Standard German:** Ducceschi & Franzini, "Speech transcription from South Tyrolean Dialect to Standard German with Whisper", Interspeech 2025 [read]:
  - **GOLD data:** 6 h 52 m, about 4,000 sentences.
  - **Results:** Whisper large-v3 baseline WER 0.53 / BLEU 33.95 → fine-tuned with Optuna hyperparameter search **WER 0.12 / BLEU 86.18**.
  - **Pseudo-labels:** 118 h 47 m of podcast audio, labelled by Whisper large-v3 itself, stayed "suboptimal".
  - — [ISCA](https://www.isca-archive.org/interspeech_2025/ducceschi25_interspeech.pdf)

**Bavarian / Franconian / Alemannic**
- Blaschke, Winkler, Förster, Wenger-Glemser, Plank, "A Multi-Dialectal Dataset for German Dialect ASR and Dialect-to-Standard Speech Translation" (Betthupferl), Interspeech 2025, arXiv 2506.02894 [read]:
  - **Data:** 4 h of read dialect speech plus 0.5 h of Standard German, with **both dialectal and Standard German references**. The mean word-level distance between the two references is **57%**; character-level **24%**.
  - **Best model:** Whisper large-v3, dialect audio → Standard German: **WER 31 / CER 16 / BLEU 56**, against 9 / 3 / 88 on Standard German audio.
  - **Decoder type:** autoregressive (Whisper) outputs sit closer to the Standard German reference; **CTC models (MMS, XLS-R) sit closer to the dialect reference** (XLS-R 1b: CER 26 against the dialect reference, 33 against Standard).
  - **Transfer:** a Swiss-German-fine-tuned Whisper was *worse* on the other German dialects (except a marginal gain on Swabian).
  - **Human check:** the best model "sometimes normalizes grammatical differences, but often stays closer to the dialectal constructions".
  - — [arXiv 2506.02894](https://arxiv.org/pdf/2506.02894), [ISCA](https://www.isca-archive.org/interspeech_2025/blaschke25_interspeech.pdf)

**Norwegian dialects → Bokmål/Nynorsk**
- Kummervold, de la Rosa, Wetjen, Braaten, Solberg, "Whispering in Norwegian: Navigating Orthographic and Dialectic Challenges", Interspeech 2024, arXiv 2402.01917 [read]:
  - **Training data:** Stage 1, 22,184 h (NRK subtitles 16,518 h); Stage 2, 6,078 h of cleaned data to reduce hallucination.
  - **NRK subtitles are non-verbatim:** a word-for-word transcript would score about **30% WER** against them.
  - **Large-model WER:** Fleurs Bokmål 10.4 → **6.6**, NST 6.8 → **2.2**, Common Voice Nynorsk 30.0 → **12.6**.
  - **Design principle:** the written standard should not depend on the speaker's dialect.
  - **Data cleaning:** model outputs filter the data (≥80% fuzzy match on the first and last word; insertion and omission n-gram checks) rather than serve as targets.
  - **Proposal:** fine-tune LLMs to harmonize corpus style.
  - — [arXiv 2402.01917](https://arxiv.org/abs/2402.01917)
- **NB-Whisper verbatim and semantic variants:** separate checkpoints, made by fine-tuning the main model for about 200–250 more steps (the card is inconsistent on the number). The verbatim variant outputs lowercase text without punctuation and does not correct grammar. **No per-variant scores; paper "forthcoming".** [model card] — [HF nb-whisper-large-verbatim](https://huggingface.co/NbAiLab/nb-whisper-large-verbatim)
- De la Rosa et al., "Boosting Norwegian ASR", NoDaLiDa 2023: NPSC Bokmål WER 5.81%, Nynorsk 11.54%. [summary] — [arXiv 2307.01672](https://arxiv.org/abs/2307.01672)

**Dialect-to-standard text normalization**
- Kuparinen, Miletić, Scherrer, "Dialect-to-Standard Normalization: A Large-Scale Multilingual Evaluation", Findings of EMNLP 2023 (Finnish, Norwegian, Swiss German, Slovene). A **character-level Transformer on 3-word sliding windows** was best for Finnish, Swiss German and Slovene; **byT5 on full sentences** was best for Norwegian. [abstract] — [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.923/)

**Chinese dialects → Mandarin**
- Wang et al., "WenetSpeech-Wu: Datasets, Benchmarks, and Models for a Unified Chinese Wu Dialect Speech Processing Ecosystem", arXiv 2601.11027 (Jan 2026) [read]:
  - **Data:** about **8,000 h**, 8 Wu sub-dialects, 11 domains.
  - **How the Mandarin targets were made:** **lexicon-based mapping, then refinement with Qwen3-8B**.
  - **ASR CER on WS-Wu-Bench:** Qwen3-ASR **29.31**, Step-Audio2-mini 26.72, Whisper-medium 78.24, Gemini-2.5-Pro 89.99. Step-Audio2-mini **LoRA**-fine-tuned (MS-Swift; pretrained on the ASR-Mid subset, then SFT on ASR-High) reaches **12.85**.
  - **Unified multi-task model (Step-Audio2-Wu-Und):** ASR **13.23** (a slight drop from the ASR-only 12.85), Wu→Mandarin AST **BLEU 53.13** against 33.31 for Qwen3-Omni and 37.81 for Step-Audio2-mini, age accuracy 0.729 against 0.541. Qwen3-Omni scores 44.27 CER on ASR.
  - **Data handling:** quality tiers are assigned per task.
  - — [arXiv 2601.11027](https://arxiv.org/pdf/2601.11027)
- KeSpeech: about 1,542 h, 27,237 speakers, 34 cities, Mandarin plus 8 subdialects. [summary, via the Wenzhou paper] — [ACL Anthology](https://aclanthology.org/2025.loresmt-1.5.pdf)
- Gao, Tamura, Kato, "Wenzhou Dialect Speech to Mandarin Text Conversion", LoResMT 2025. The first public Wenzhou→Mandarin dataset; XLS-R-300M encoder with a text decoder. Word-usage differences remain the main weakness. Numbers not retrieved. [summary] — [ACL Anthology](https://aclanthology.org/2025.loresmt-1.5/)
- "Towards Comprehensive Semantic Speech Embeddings for Chinese Dialects" (arXiv 2601.07274) argues that dialect-speech → Mandarin-text speech LLMs are more practical than dialect-specific LLMs, because most dialects are mainly spoken. [summary] — [arXiv](https://arxiv.org/pdf/2601.07274)

**Japanese dialects**
- Imaizumi, Masumura, Shiota, Kiya, "End-to-end Japanese Multi-dialect Speech Recognition and Dialect Identification with Multi-task Learning", *APSIPA Transactions on Signal and Information Processing* 11(1), 2022 [read]:
  - **Data:** 6 dialects (Aomori, Hiroshima, Kumamoto, Nagoya, Sapporo, Sendai); 89,817 training and 7,450 test utterances, plus a standard Japanese corpus. Transformer encoder-decoder.
  - **Mixing data:** the standard-only model scores CER 52.5% on dialect speech. Mixed dialect+standard training gives 8.0%, but the standard test gets *worse*, 14.3 → 14.9.
  - **Label first (DID2ASR):** 8.4% CER, or 7.3% with the oracle label. Errors in the predicted label propagate into the transcript.
  - **Label after transcript (ASR2DID): 7.0% CER**, and dialect-ID accuracy **86.5%** against 83.1% for label-first.
  - **Soft dialect distribution (DID+ASR):** best on the standard test, and a 9.9% error reduction against the conventional multi-condition model on the multi-dialect ASR task.
  - — [PDF](https://pdfs.semanticscholar.org/f16c/94debe0f1a381f4586092cc66d215891ffa2.pdf), [DOI](https://doi.org/10.1561/116.00000045)
- Mitsumori, Mizumoto, Fujita, "Dialect-Robust Speech Language Models with Synthetic Pseudo-Dialect Augmentation", IEEE SLT 2026, arXiv 2610.09321 (7 Oct 2026) [read via HTML]:
  - **Model:** Whisper-large-v3 encoder + projector + Llama-3.1(-Swallow)-8B; **only the projector is trained** (18.4M parameters).
  - **Test sets:** CPJD (20 Japanese dialects), SwissDial (8), KeSpeech (8). Task: dialect → English speech translation, scored with BLEU.
  - **BLEU (JA / DE / ZH):**

    | Training setup | Output | JA | DE | ZH |
    |---|---|---|---|---|
    | No augmentation | English only | 24.21 | 22.37 | 11.61 |
    | Synthetic standard speech | English only | 25.38 | 31.57 | 11.72 |
    | Pseudo-dialect speech | English only | 26.24 | 32.47 | 11.67 |
    | Pseudo-dialect speech | `<ASR>` + English | 27.67 | 29.81 | 16.30 |
    | Pseudo-dialect speech | `<standard>` + English | **28.26** | 30.73 | **16.37** |
    | Pseudo-dialect speech | ASR + Standard + English | 24.95 | 29.92 | 13.24 |

  - **Findings:** the two-intermediate-step format was **worse in all three languages** (error propagation). Standard+English beat direct English in 19 of 20 Japanese dialects (9 significant). For German, the multi-task outputs did *not* help.
  - — [arXiv 2610.09321](https://arxiv.org/html/2610.09321v1)

**Arabic dialects**
- Talafha et al., "Casablanca: Data and Models for Multidialectal Arabic Speech Recognition", EMNLP 2024. A multidialect corpus with transcriptions, code-switching, dialect and gender labels. Zero-shot Whisper, SeamlessM4T and MMS give high WER. Whisper-v3 fine-tuned on all dialects reduces Algerian WER by 14.06 points against Whisper-Moroccan. [summary] — [arXiv 2410.04527](https://arxiv.org/pdf/2410.04527)
- Alhafni et al., "Exploiting Dialect Identification in Automatic Dialectal Text Normalization", arXiv 2407.03020. **Title only; not read.** — [arXiv](https://arxiv.org/pdf/2407.03020)

**Single model with selectable output form (not dialect)**
- Zhang et al., "Dual-Form ASR: Semantics-Aware Inverse Text Normalization for Chinese Speech Recognition", arXiv 2609.02901 (Jul 2026, submitted to SLT 2026). One prompt-conditioned model emits either the spoken form or the written form. Paired targets come from an LLM generate-and-judge workflow; training uses an ITN-weighted MWER loss. No numbers in the abstract. [abstract] — [arXiv](https://arxiv.org/abs/2609.02901)
- Li et al., "Multi-Dialect Speech Recognition with a Single Sequence-to-Sequence Model", ICASSP 2018 (7 English dialects). Dialect information enters either as a symbol in the output targets or as a one-hot vector into every layer. It beats dialect-specific LAS models by **3.1–16.5% relative**. [summary; pre-2022] — [IEEE](https://ieeexplore.ieee.org/iel7/8450881/8461260/08461886.pdf)

### Inferences
- **Evidence supports keeping the dual output, with one intermediate step.** In Mitsumori, "transcript then target" beat "target only": JA 27.67 vs 26.24, ZH 16.30 vs 11.67. WenetSpeech-Wu shows that adding a translation task costs only about 0.4 CER on transcription. Our "방언 → 표준어" order matches the helpful configuration. Adding a *third* text step, such as a gloss, would echo the losing ASR+Standard+English setup.
- **Order of the region/accent field:** Imaizumi's results favour putting it **after** the transcript (ASR2DID: better CER and better ID), or using a soft distribution, rather than a leading region tag that conditions decoding on a possibly wrong guess. Caveat: a leading tag is fine when the region is *given* (oracle 7.3% CER), e.g., as user-provided metadata.
- **Risk to the verbatim line:** autoregressive LM decoders drift toward the standard form (Betthupferl). An LLM-based LALM may "standardize" the 방언 line, especially for Gangwon and Chungcheong. Checks: the CER between the predicted dialect line and the *gold standard* line (if it is lower than against the gold dialect line, the model is normalizing), and the per-eojeol `isDialect` recall from AI Hub labels.
- **Mixing standard Korean speech** (e.g., KsponSpeech-style, where dialect line = standard line) has a direct precedent: Timmel's 125 h of CV-de kept Standard German WER flat. Imaizumi shows that naive mixing can hurt the standard test slightly unless dialect is modelled explicitly.
- **Valid variation inflates the metric.** In Swiss German, about 11–12 points of WER on the standard target were stylistic variation. AI Hub's standard line is eojeol-substitution, so it should vary less, but endings and spacing (e.g., -요 vs -습니다, spacing) likely inflate our 0.147. A small LLM-judge or "semantically correct" error class (cWER-style) would separate true errors.
- **Building standard targets:** WenetSpeech-Wu (lexicon mapping + LLM refinement) and Dual-Form ASR (LLM generate-and-judge) offer recipes for creating or cleaning standard targets if AI Hub's `standard_form` proves noisy.

### Gaps
- I found **no paper that trains one model with explicit task tokens to emit verbatim dialect *or* standard text and compares it against two separate models.** NB-Whisper used separate checkpoints and published no numbers. Dual-Form ASR is not about dialect and gives no numbers in its abstract.
- Low German (Plattdeutsch): nothing found.
- Dialectal Arabic → MSA output from a speech LLM: nothing found.
- Japanese dialect → standard Japanese *speech-to-text conversion*: not found. CPJD appears only as a test set.

---

## Q3. Do speech LLMs report dialect/accented ASR results? Evidence on fine-tuning them on low-resource dialect data (data size, LoRA, forgetting)

### Takeaway
Among the candidates, only **Qwen3-ASR** documents dialect robustness:
- 22 Chinese dialects; KeSpeech **CER 5.10** (1.7B); internal 22-dialect average **15.94**; 16-accent English **16.07**.
- Korean CER 2.57 on Fleurs.

Qwen3-Omni and Qwen2.5-Omni publish no dialect tables beyond Cantonese. I found no dialect results for Gemma 3n/4 or SALMONN.

Off-the-shelf LALMs collapse on an unseen dialect: Qwen3-ASR 29.31 and Qwen3-Omni 44.27 CER on Wu. A LoRA fine-tune of a 7B-class LALM on Wu data cut CER from 26.72 to 12.85.

Forgetting shows up in three ways: hallucination on silence, loss of segmentation, and loss of other languages. The documented fixes are mixing in standard/general data, a low LoRA scaling factor, and self-distillation.

### Cited Findings
**Qwen3-ASR Technical Report** (arXiv 2601.21337, Jan 2026) [read via HTML]
- **Coverage:** 30 languages + 22 Chinese dialects.
- **Architecture:** AuT encoder (300M for the 1.7B model, 180M for 0.6B) emitting 12.5 Hz tokens, then a Qwen3 LLM; built on Qwen3-Omni.
- **Training:** AuT pretraining on about 40M h of pseudo-labeled audio. The ASR SFT stage includes **context-biasing data**: "context tokens inside the system prompt as background knowledge". It also trains the model to transcribe rather than follow instructions in the prompt. Then GSPO reinforcement learning on about 50k utterances.
- **KeSpeech CER:** GPT-4o-Transcribe 26.87, Gemini-2.5-Pro 24.71, Doubao 5.27, Whisper-large-v3 28.79, Qwen3-ASR-0.6B 7.08, **1.7B 5.10**.
- **WenetSpeech-Chuan (easy–hard):** 1.7B 11.99–21.63, against 14.35–26.80 for Whisper.
- **Internal dialogue sets (Table 4):**

  | Test set | Qwen3-ASR-1.7B | Whisper-large-v3 | Gemini-2.5-Pro | Doubao |
  |---|---|---|---|---|
  | Accented English (16 accents) | **16.07** | 21.30 | 23.85 | – |
  | Chinese dialects (22) | **15.94** | 44.55 | 47.70 | 19.85 |

- **Language ID:** average accuracy 97.9 against 94.1 for Whisper.
- **Korean CER (1.7B / 0.6B):** Fleurs **2.57** / 3.72; Common Voice 5.88 / 8.48; MLC-SLM 8.61 / 10.31.
- The fine-tuning recipe is mentioned but its details (LoRA vs. full) are not in the report.
- — [arXiv 2601.21337](https://arxiv.org/html/2601.21337v1)

**Other LALMs**
- FireRedASR (arXiv 2501.14350) quotes average KeSpeech CERs from another report: **Qwen2-Audio-Instruct 9.9%**, Baichuan-omni 6.7%, Whisper-large-v3 44%. FireRedASR-LLM itself reaches 3.56% and AED 4.48%. [summary] — [arXiv 2501.14350](https://arxiv.org/pdf/2501.14350)
- Qwen3-Omni Technical Report (arXiv 2509.17765): Cantonese appears only inside the multilingual Fleurs average and the speech-translation tests; no dialect table. Qwen3.5-Omni reports Fleurs Cantonese 2.2%. [summary; not read] — [Qwen3-Omni](https://arxiv.org/pdf/2509.17765), [Qwen3.5-Omni](https://arxiv.org/pdf/2604.15804)
- WenetSpeech-Wu, zero-shot CER on WS-Wu-Bench: Qwen3-ASR 29.31, Step-Audio2-mini 26.72, Qwen3-Omni 44.27. After a **LoRA** fine-tune on about 8k h of Wu, Step-Audio2-mini reaches **12.85**. A Conformer trained from scratch reaches 15.14 and Whisper-medium-Wu 14.33. [read] — [arXiv 2601.11027](https://arxiv.org/pdf/2601.11027)
- Xu et al., "Leveraging LLM and Self-Supervised Training Models for Speech Recognition in Chinese Dialects: A Comparative Analysis" (arXiv 2505.21138; Interspeech 2025 per its ISCA URL). Data2vec2 pretrained on **300k h of unlabeled dialect and accented speech**, then **40k h of supervised** alignment; claims SOTA on KeSpeech; compares projectors and LLMs. Numbers not retrieved. [abstract] — [arXiv](https://arxiv.org/abs/2505.21138), [ISCA](https://www.isca-archive.org/interspeech_2025/xu25k_interspeech.pdf)
- Mitsumori et al. (SLT 2026): with a **frozen** encoder and LLM, training only an 18.4M-parameter projector already gives dialect gains. A Whisper+LLM cascade with LoRA on the LLM (41.9M trainable parameters) scored higher in absolute terms (29.78–30.44 BLEU, JA). [read] — [arXiv 2610.09321](https://arxiv.org/html/2610.09321v1)
- Ti-Audio (arXiv 2604.11110, Apr 2026): the first multi-dialect Tibetan speech LLM (Ü-Tsang, Amdo, Kham). It uses a Dynamic Q-Former adapter, "mutual assistance among related dialects" and **temperature-based dialect sampling**; claims SOTA on Tibetan ASR and speech translation. [abstract] — [arXiv](https://arxiv.org/abs/2604.11110)
- Per-region LoRA on **Qwen2.5-Omni-3B** for accented English: rank 32 on the LLM attention projections, one adapter per region "to avoid cross-region interference" (arXiv 2609.20828). [summary] — [arXiv](https://arxiv.org/pdf/2609.20828)
- Qwen2-Audio multilingual adaptation with a small LoRA (rank 16, α 32, attention projections only) "to avoid catastrophic forgetting" (ILT, arXiv 2507.08477). [summary] — [arXiv](https://arxiv.org/pdf/2507.08477)
- Community report: LoRA on **Qwen3-ASR-1.7B** for Vietnamese dialect speech cut WER from 11.56 to 8.72; full fine-tuning was only 0.13 better and took about twice as long. [summary; unverified GitHub] — [GitHub](https://github.com/duclvq/finetune_QwenASR_on_Vietnamese)
- Hu et al., "Chain-of-Thought Prompting for Speech Translation" (arXiv 2409.11538): with a Megatron-T5 speech LLM, **LoRA on the LLM beat full fine-tuning**. [abstract] — [arXiv](https://arxiv.org/abs/2409.11538)

**Forgetting and its mitigations**
- Akeret (2026): early LoRA runs on Whisper large-v3 made the model hallucinate on silence. The no-speech/EOS probability on silent input dropped **from above 90% to 8.8%**. Lower α scaling (α/r = 0.2) and same-model knowledge distillation fixed it; full fine-tuning showed none. The author calls α = 2r "too aggressive" for Whisper. Sequential per-corpus curricula forgot earlier phases. [read] — [arXiv 2606.07608](https://arxiv.org/html/2606.07608v1)
- Timmel (2025): sentence-only fine-tuning destroyed long-form segmentation. Adding 125 h of Standard German Common Voice kept standard WER at 6.42 against 6.53. [read] — [arXiv 2412.15726](https://arxiv.org/pdf/2412.15726)
- Blaschke (2025): a small Whisper fine-tuned on Standard German overfit and got worse on dialects. A Swiss-German fine-tune got worse on other German dialects. [read] — [arXiv 2506.02894](https://arxiv.org/pdf/2506.02894)
- WenetSpeech-Wu: multi-task fine-tuning cost 0.38 CER on ASR (12.85 → 13.23) while adding translation and paralinguistic tasks. [read] — [arXiv 2601.11027](https://arxiv.org/pdf/2601.11027)

**Region-conditioned adaptation**
- Mehralian, Farasyn, Breitbarth, Ghyselen, Van hamme, "GLoRIA: Gated Low-Rank Interpretable Adaptation for Dialectal ASR", ICASSP 2026, arXiv 2603.02464:
  - **Method:** each rank-1 LoRA component is gated by an MLP over location metadata (coordinates).
  - **Results:** on GCND it beats standard LoRA, geo-conditioned full fine-tuning, and both dialect-specific and unified full fine-tuning, while updating under 10% of parameters. It also generalizes to unseen dialects. No numbers in the abstract.
  - [abstract] — [arXiv](https://arxiv.org/abs/2603.02464)

### Inferences
- **Choosing a candidate.** Qwen3-ASR-1.7B has the strongest *documented* dialect and Korean ASR. Its 0.6B/1.7B LLMs are small, though, and its SFT deliberately trains it to *ignore instructions* and emit a fixed `language X <asr_text>` template. Producing "방언 / 표준어 / 지역" lines means fine-tuning that overrides this template; the dialect→standard conversion would then rest on a 1.7B LLM. Qwen3-Omni-30B-A3B and Qwen2.5-Omni-7B keep instruction following and a larger LLM, but they are much weaker zero-shot on unseen dialects (Qwen3-Omni: 44.27 CER on Wu) and need adaptation anyway. *Not tested on Korean dialects by anyone.*
- **Data scale.** Our roughly 100k+ AI Hub utterances (likely on the order of 100–300 h; the exact hours are the project's to check) is **small next to WenetSpeech-Wu (8k h) or Swiss work (500–1,400 h)**, but much larger than South Tyrolean (7 h), which still went from 0.53 to 0.12 WER. The AI Hub corpora hold 3,000+ h per region, so using more of the downloaded data is the most direct lever.
- **LoRA settings with precedent.** Speech-LLM papers use rank 16–32 on attention projections. For Whisper-style decoders, use a **low α/r** (0.2 worked; 2.0 caused hallucinations). Add a silence/no-speech regression check and a standard-Korean regression check (KsponSpeech or Fleurs-ko) to every run.
- **Temperature-based sampling** across regions (Ti-Audio) is a cheap fix if Jeju is under-represented in the training mix.

### Gaps
- No dialect or accent ASR results found for **Gemma 3n / Gemma 4**, **SALMONN**, **Qwen2.5-Omni** (general ASR only) or **Kimi-Audio**.
- No study found that measures LoRA vs. full fine-tuning *on an LALM for dialects* with general-capability forgetting metrics (e.g., instruction following, other languages) before and after.
- The Qwen3-ASR fine-tuning recipe (LoRA vs. full, hyperparameters) is not documented in the report.
- The Xu et al. (Interspeech 2025) numbers and its LLM-size comparison were not retrieved.

---

## Q4. Techniques for the hardest cases (very divergent, low-resource, short-clip dialects like Jeju; utterance-final endings)

### Takeaway
I found no Jeju-specific ASR paper and no paper on dialect *sentence-final endings*. Transferable techniques with evidence:
1. **Pseudo-dialect augmentation:** an LLM rewrites standard text into dialect, then a standard-language TTS voices it. Gains are small (+0.9 BLEU) and none for Chinese, because the accent is not captured.
2. **Lexicon or retrieval context** in the prompt: Qwen3-ASR supports it natively; DIA-REFINE retrieves examples with BM25.
3. **LLM generative error correction (GER)** over N-best lists. Gains are real on low-resource languages, but watch contamination and over-correction.
4. **Region-conditioned adapters** (GLoRIA, per-region LoRA).
5. **Lexicon + LLM construction and cleaning of targets.**

Jeju-specific resources also exist (JIT 170k pairs, JSS 13.8 h single-speaker speech, AI Hub Jeju 3,000+ h).

### Cited Findings
**Synthetic and pseudo-dialect data**
- Mitsumori et al. (SLT 2026) [read]:
  - **Method:** gpt-oss-120b rewrites standard transcripts into a random target dialect; standard TTS voices them (Tsukasa-Speech for Japanese; Qwen-TTS for German and Chinese, using the original speaker as reference).
  - **Gains over synthetic *standard* speech:** **+0.86 BLEU JA (p<0.001), +0.90 DE, none for ZH**.
  - **Stated limits:** the dialect accent is absent (a Tsugaru failure case); dialect words the LLM does not know cannot be learned; Chinese character readings are not reproduced; training time roughly doubles.
  - — [arXiv 2610.09321](https://arxiv.org/html/2610.09321v1)
- Kakao **JSS**: 10,000 single-speaker Jejueo utterances (13 h 47 m) with transcripts, used for DCTTS; Jamo input gave the best MCD, 14.32. [read] — [arXiv 1911.12071](https://arxiv.org/html/1911.12071v1), [GitHub kakaobrain/jejueo](https://github.com/kakaobrain/jejueo)
- Code-switching TTS augmentation lowered MER from 12.1% to 10.1% (DevMan) and from 17.8% to 16.0% (DevSGE). In Quechua, synthetic text plus synthetic speech improved WER by 8.73%. [summary] — [arXiv 2601.00935](https://arxiv.org/html/2601.00935), [arXiv 2204.00291](https://arxiv.org/abs/2204.00291)
- WenetSpeech-Wu cites DiaMoE-TTS, a unified IPA-based dialect TTS framework, as a dialect TTS baseline. Not read. — [arXiv 2601.11027](https://arxiv.org/pdf/2601.11027)
- South Tyrolean: 118 h of self-pseudo-labeled podcast audio (Whisper large-v3 translating) stayed "suboptimal" next to 7 h of gold data. [read] — [ISCA](https://www.isca-archive.org/interspeech_2025/ducceschi25_interspeech.pdf)

**Lexicon, retrieval and context biasing**
- Qwen3-ASR's SFT includes context-biasing data: arbitrary context text in the system prompt acts as background knowledge. No numbers reported. [read] — [arXiv 2601.21337](https://arxiv.org/html/2601.21337v1)
- DIA-REFINE (text): **10 BM25-retrieved example pairs** in context, plus classifier verification with up to 2 retries. TDR rises sharply; BLEU barely changes. [read] — [arXiv 2511.06680](https://arxiv.org/html/2511.06680)
- WenetSpeech-Wu: standard targets built by **lexicon-based mapping, then LLM (Qwen3-8B) refinement**. [read] — [arXiv 2601.11027](https://arxiv.org/pdf/2601.11027)
- Speech-LLM contextual biasing with "common word cues" (similar-sounding common words placed next to each bias word in the prompt) reduced B-WER by 11.3–16.3% against baseline. Speech LLMs otherwise tend to delete or substitute rare words. [summary, secondary review] — [review](https://www.themoonlight.io/en/review/contextual-biasing-for-asr-in-speech-llm-with-common-word-cues-and-bias-word-position-prediction)
- "How to Recognize New Words: A Comparison Between Context Biasing Methods and Speech LLMs" (arXiv 2608.05759): speech LLMs are strong on read speech but **generalize less to non-read speech**, and they are sensitive to the number and order of distractors in the bias list. [summary] — [arXiv](https://arxiv.org/pdf/2608.05759)
- Kakao JIT: adding Korean "copy" pairs (identity mappings) gave up to +0.9 BLEU. [read] — [arXiv 1911.12071](https://arxiv.org/html/1911.12071v1)

**LLM post-correction (GER)**
- West Frisian, contamination-aware study (arXiv 2605.19711): XLS-R 1-best **13.5% WER**, 5-best oracle **9.6%**, trigram LM 12.1%. The LLM can produce words absent from the N-best list. The LLM-correction WER itself was not captured. [summary] — [arXiv](https://arxiv.org/html/2605.19711)
- CantoASR (arXiv 2511.04139, Nov 2025), for low-resource Cantonese: forced-alignment acoustic features, a **LoRA-fine-tuned Whisper** for tone discrimination, and an **instruction-tuned Qwen-Audio for prosody-aware correction**. Reports "substantial CER gains over Whisper-Large-V3"; no numbers in the abstract. [abstract] — [arXiv](https://arxiv.org/abs/2511.04139)
- Sejong 2024 Korean LLM correction: WER −13.7%, CER −6.2%. [abstract] — [DBpia](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11891057)
- Over-correction controls: phonetic context with N-best lists (arXiv 2505.17410), and confidence filtering. [summary] — [arXiv 2505.17410](https://arxiv.org/abs/2505.17410)
- Chain-of-thought decoding for speech translation (Hu et al., arXiv 2409.11538), where the model first decodes the transcript and then translates with the transcript in the prompt: **+2.4 BLEU** on average over speech-only prompting, and **about +2 BLEU over predicting concatenated ASR+AST in one pass**, across 6 tasks. [abstract] — [arXiv](https://arxiv.org/abs/2409.11538)
- "Transcribe, Translate, and Optimize" (arXiv 2609.26536), Qwen2.5-Omni-3B with GRPO: transcription-first CoT beats direct speech translation by **+1.77 BLEU (CoVoST 2) and +0.83 (FLEURS)**. It also flags the mismatch between gold transcripts used in SFT and the model's own transcripts at inference. [summary] — [arXiv](https://arxiv.org/html/2609.26536)

**Region conditioning and routing**
- GLoRIA: location-gated LoRA (above). [abstract] — [arXiv 2603.02464](https://arxiv.org/abs/2603.02464)
- DialectMoE (CCL 2024): a dialect encoder's features drive a mixture-of-experts router. [summary] — [ACL Anthology](https://aclanthology.org/2024.ccl-1.89.pdf)
- Li et al., ICASSP 2018: dialect as an output symbol or as a one-hot input; 3.1–16.5% relative gain over per-dialect models. [summary] — [IEEE](https://ieeexplore.ieee.org/iel7/8450881/8461260/08461886.pdf)
- Imaizumi et al., 2022: an oracle dialect label as a prefix lowers CER from 8.4% to 7.3%. Predicted labels propagate errors. [read] — [PDF](https://pdfs.semanticscholar.org/f16c/94debe0f1a381f4586092cc66d215891ffa2.pdf)

**Data cleaning and evaluation**
- NB-Whisper: model-in-the-loop filtering (≥80% fuzzy match on the first and last word, n-gram insertion and omission checks, NER-count filter), followed by a 50k-step retrain on the cleaned data to cut hallucination. [read] — [arXiv 2402.01917](https://arxiv.org/abs/2402.01917)
- Evaluation tools: DIA-REFINE's DFS/TDR; Akeret's cWER (semantic-variant exclusion); Betthupferl's comparison against both dialect and standard references; the copy baseline (Kakao). [read] — links above

### Inferences (concrete techniques for the project, ordered by expected value per effort)
1. **Jeju lexicon from our own labels.** Mine the dialect→standard eojeol pairs where `isDialect=True` in AI Hub `eojeolList`, and add Kakao JIT (170k pairs). Use the result three ways:
   - (a) as **retrieved context** in the prompt for a second pass: first-pass dialect line → BM25 / edit-distance lookup → top-k (dialect, standard) entries in the system prompt → regenerate the 표준어 line. This mirrors Qwen3-ASR context biasing and DIA-REFINE retrieval.
   - (b) as **text-only training pairs** for the dialect→standard part, inserted as text-in/text-out turns so that the LLM learns the mapping without audio.
   - (c) as a **coverage diagnostic**: what share of Jeju test errors fall on eojeol whose form is in or out of the lexicon.
2. **Sentence-final endings.** These are a closed class: a few hundred surface forms per region, e.g., Jeju -수다 / -수꽈 / -쿠다 (illustrative; derive the real list from the labels). Build an ending inventory per region from `eojeolList`, report **ending-level accuracy** separately, and add these forms to the context list. No published method targets this; it is project-specific.
3. **Keep the dialect line before the standard line, and put the region tag last.** Supported by Mitsumori, Hu (CoT) and Imaizumi. If the user region is known (coaching app metadata), **feed it as an input prefix** instead (oracle-label gains in Imaizumi; GLoRIA-style gating).
4. **Pseudo-dialect speech for Jeju.** Expect small gains with standard-voice TTS (Mitsumori: about +0.9 BLEU, and nothing where the TTS cannot render dialect phonology). A better option is voice-cloning TTS conditioned on *real Jeju speakers* from AI Hub Jeju or JSS, so the acoustics carry the accent. Use Kakao JIT or LLM-rewritten Jeju text as the source.
5. **Region-balanced or temperature sampling, and region-conditioned LoRA,** if Jeju is under-represented (Ti-Audio, GLoRIA, per-region LoRA).
6. **N-best + LLM GER restricted to the 표준어 line,** with confidence filtering. Expect modest gains, and watch for over-correction toward "fluent standard Korean" that silently drops dialect content.
7. **Model-in-the-loop data cleaning** (NB-Whisper recipe) to remove misaligned AI Hub segments. Short Jeju clips with partial transcripts are a likely source of label noise; this is an inference, not measured.
8. **Evaluation hygiene:** per-region copy-baseline CER, dialect-line "standardization rate" (is the prediction closer to the gold standard line than to the gold dialect line?), ending-level accuracy, and an LLM or human semantic check on a sample of the 표준어 line.

### Gaps
- No Jeju (or any Korean dialect) speech-recognition paper with numbers was found. Jeju-specific error analyses of endings and short clips do not exist in the literature I could reach.
- No paper evaluates **dialect-lexicon biasing** of a speech LLM on dialect speech; the contextual-biasing evidence is on rare words and named entities only.
- No study of **TTS-synthesized Korean dialect speech** used to train ASR was found.
- LLM-GER numbers specific to dialect ASR were not found; the West Frisian LLM-correction result was not captured.
