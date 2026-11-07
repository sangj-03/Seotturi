# Shortcut learning and confounds in accent/dialect ID, and how to make a joint transcript + accent LALM judge accent from pronunciation/prosody

Scope note: notes compiled 2026-10-14 for the Korean dialect coaching project (AI Hub regional corpora, one dataset per region, no speaker IDs, birthplace-based labels). "Per search summary" = number taken from a search-engine digest of the paper, not read in the paper itself; treat as unverified. "Not fetched" = well-known reference cited from prior knowledge, URL not opened in this session.

## Q1. Lexical/content leakage in accent/dialect ID and how papers controlled it

### Takeaway
No accent-ID paper I found strips dialect *words* out and reports the before/after numbers, so our F1 0.216 result has no published equivalent. The closest evidence points the same way: content-oriented encoders (Whisper) score well when train and test content or speaking style match, then fall to near chance when style changes. Two kinds of system hold up better: ones that see only phoneme-aligned pronunciation units, with the transcript used only for forced alignment, and ones built on prosody/F0. Large audio LMs show a strong text-over-audio bias on a related paralinguistic task (emotion), the same pattern as our speech LLM that learned region from dialect words.

### Cited Findings
**Read vs. spontaneous transfer (the most direct content/style-control study, 2026)**
- Perez-Toro et al., "Do Speech Representations Preserve Regional Accent Across Read and Spontaneous Speech?" (arXiv 2610.06430, Oct 2026). Data: German RVG1-v3, 500 speakers, 9 regions, 498 speakers with both read speech (digits/commands/phonetically balanced sentences) and spontaneous speech (prompted monologue). Setup: 10 frozen representations (Whisper-large-v3, WavLM, XLSR, HuBERT, x-vector, ECAPA, PhonoQ, eGeMAPS, Mimi, Qwen2-Audio-7B-Instruct), mean-pooled per speaker, MLP probe, 5 speaker-independent 70:30 splits — [arXiv](https://arxiv.org/html/2610.06430)
  - Matched read speech: Whisper (layer 23) reaches 0.489 nine-way UAR (chance ~0.111) and 0.767 three-way UAR. Geolocation median error is 148 ± 16 km — [arXiv](https://arxiv.org/html/2610.06430)
  - Across speaking styles, Whisper collapses. Nine-way UAR falls to 0.11 (read→spontaneous) and 0.18 (spontaneous→read). Geolocation error rises to 363 km, worse than the 242 km you get by always predicting the mean location — [arXiv](https://arxiv.org/html/2610.06430)
  - Speaker embeddings (ECAPA) transfer better, at about 0.23–0.25 UAR in both directions. Transfer correlates with cross-style speaker retrieval (ρ=+0.74, p=0.014; exploratory, does not survive Holm correction) — [arXiv](https://arxiv.org/html/2610.06430)
  - Content control: sentence-disjoint evaluation lowers Whisper nine-way UAR from ~0.48 to 0.40, so part of the score comes from shared prompts. Duration matching leaves it unchanged (0.486→0.488) — [arXiv](https://arxiv.org/html/2610.06430)
**Phoneme-only / pronunciation-unit features**
- Qu, Todisco, Evans, "Phoneme-Aware Pronunciation Representations for L2-English L1-Background Accent Identification" (arXiv 2609.10466, Sep 2026). The transcript is used only for phoneme forced alignment, and no word- or sentence-level text reaches the classifier. Each unit pairs a frozen-encoder acoustic segment with its aligned phoneme. On L2-ARCTIC with 4-fold speaker-disjoint CV: 81.41% accuracy, 81.21% macro-F1, the best of the systems they compared. A Whisper-based ablation shows an extra gain from adding phoneme information — [arXiv](https://arxiv.org/abs/2609.10466)
- AESRC2020 challenge (Shi et al., ICASSP 2021, "The Accented English Speech Recognition Challenge 2020: Open Datasets, Tracks, Baselines, Results and Methods"). The accent-ID baseline improved from 47.8% (12-layer Transformer from scratch) and 54.1% (3-layer) to 76.1% when initialized from an ASR encoder. The winning accent-ID system was a TDNN on phonetic posteriorgrams (PPG); one summary gives 83.63%, but the sources conflict on which team ranked first, so treat that figure as unverified. All AESRC numbers are per search summary — [arXiv 2102.10233](https://arxiv.org/pdf/2102.10233); [baseline repo](https://github.com/R1ckShi/AESRC2020)
**Content-controlled vs. content-mismatched systems (older forensic work)**
- Brown (2015, PhD thesis, York), "Considering Accent Recognition Technology for Forensic Applications". Earlier ACCDIST-style systems needed the test speaker's content to match the training speakers' content exactly. The proposed Y-ACCDIST reaches 86.7% on a four-way geographically close accent task, against 47.5% for the best text-independent system. Pre-2019, per search summary — [thesis PDF](https://etheses.whiterose.ac.uk/id/eprint/20393/1/thesis1_FINAL_RESUB6.pdf)
- Brown & co. "Segmental Content Effects on Text-dependent Automatic Accent Recognition" (circa Odyssey 2018). Exists; details not read — [ResearchGate](https://www.researchgate.net/publication/325997656_Segmental_Content_Effects_on_Text-dependent_Automatic_Accent_Recognition)
**Human perception: dialect is identifiable without lexical cues**
- Listeners can identify a speaker's subregional Chinese dialect from single monosyllables, natural or with the tone swapped in. Vowels are the main cue and tone is secondary (Speech Communication). Per search summary — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167639320302417)
- Naive listeners identify German-Swiss dialects above chance from delexicalized, low-pass-filtered speech (prosody only). Per search summary; exact paper not pinned down — [Frontiers in Communication](https://www.frontiersin.org/journals/communication/articles/10.3389/fcomm.2021.691225/full)
**Korean dialect ID that explicitly targets prosody**
- Lee, Kim, Chung, "Korean Dialect Identification Based on Intonation Modeling" (O-COCOSDA 2021). A BiLSTM with attention over frame-level spectral and F0 features; the attention is meant to pick out dialect-rich segments — [ResearchGate](https://www.researchgate.net/publication/354720698_Korean_Dialect_Identification_Based_on_Intonation_Modeling)
- "Korean Dialect Identification Based on an Ensemble of Prosodic and Segmental Feature Learning for Forensic Speaker Profiling" (IEEE). An attention-BiLSTM on prosodic features combined with a DNN on segmental features reaches 61.28% F-measure on a public Korean conversational dataset, +25.87 points over its baseline. Which dataset it is (possibly AI Hub) is unverified — [IEEE Xplore](https://ieeexplore.ieee.org/document/9997971/)
**Benchmarks that don't control lexical content**
- Voxlect (Feng et al., USC, arXiv 2508.01691, Aug 2025) covers 11 language groups, 30 corpora and over 2M utterances. It does not separate lexical from acoustic cues. Korean is not included (Seoul/Gyeongsang/Jeolla listed as future work). Whisper-Large is best in 5 of 11 groups, e.g. Arabic macro-F1 0.923 — [arXiv](https://arxiv.org/html/2508.01691v1)
- Kukk & Alumäe, "Improving Language Identification of Accented Speech" (Interspeech 2022, pp. 1288–1292). Language-ID accuracy is inversely correlated with accent strength. Adding the output of a lexicon-free ASR for the target language cuts errors by 35–63% relative on non-native speech, without hurting native speech. Models trained on VoxLingua107. Takeaway for us: a classifier relies on whatever signal it is fed (here, linguistic/phonotactic hypotheses dominate over accent) — [ISCA](https://www.isca-archive.org/interspeech_2022/kukk22_interspeech.html)

### Inferences
- Our failure (speech LLM F1 0.216 on clips without dialect words; 14 of 15 Seoul speakers called regional) is the lexical version of the Whisper read/spontaneous collapse: the model learned cues that only exist in some content conditions. The Perez-Toro result also warns that a coaching app where users *read prompted sentences* is itself a style shift from AI Hub conversational speech. Whisper-type cues may not transfer even with no dialect words present. Evaluate on read prompts recorded by speakers from each region.
- The cleanest published way to block word-level leakage is the phoneme-aware approach: build accent features from phone-aligned acoustic segments (forced alignment against the *standard* pronunciation of the transcript), so the classifier never sees words. In a joint LALM, an analogue is an auxiliary accent target computed from phone-aligned encoder segments rather than from the decoder's text context.
- Adding prosody (F0 contour, intonation) as an explicit feature or auxiliary target has Korean precedent (the two K-DID papers) and fits Korean dialect phonology: Gyeongsang pitch accent and Jeolla/Chungcheong intonation.
- Our current mitigation (accent trained only on clips without dialect words) is consistent with the "lexical-free test set" idea. Add a sentence/prompt-disjoint split as well (Perez-Toro: 0.48→0.40 just from removing shared sentences).

### Gaps
- I found no accent-ID paper that adversarially removes linguistic content (e.g., GRL against a word/BPE predictor). GRL is documented only for speaker or domain removal (see Q2/Q3).
- I found no published "with vs. without dialect words" ablation for accent/dialect ID. Our experiment appears novel.
- The Qwen2-Audio-7B numbers inside Perez-Toro et al. were not retrieved.
- Older content-controlled corpora (ABI, Speech Accent Archive "Please call Stella") were not reviewed in this pass.

## Q2. Channel/dataset confounds in language/dialect/accent ID, and fixes

### Takeaway
"Each region recorded as a separate dataset" is the textbook setup for a channel confound. In 2012, on LDC Arabic telephone corpora, a dialect classifier trained on *silence segments only* matched one trained on speech. Channel equalization did not remove the cue, while a corpus recorded uniformly across dialects brought silence-only accuracy to chance. Modern self-supervised (SSL) dialect-ID models trained on YouTube-collected ADI-17 go from 92 macro-F1 in domain to about 1 on a different domain, and a model given MSA from broadcast TV used the channel as a shortcut. Our silenced-gap collapse (0.865→0.550) and noise-suppression drop (→0.693) have a close analogue in anti-spoofing, where silence duration alone gave 85% accuracy.

### Cited Findings
**Silence/channel as the label signal**
- Bořil, Sangwan, Hansen, "Arabic Dialect Identification – 'Is the Secret in the Silence?' and Other Observations" (Interspeech 2012). Read from the PDF — [ISCA](https://www.isca-archive.org/interspeech_2012/boril12_interspeech.html); [PDF](https://personal.utdallas.edu/~hynek/pdfs/BorilSangwanHansen-IS12-ArabicDID_updated2.pdf)
  - On LDC Arabic CTS corpora, each dialect was "acquired in separate efforts", and each has its own channel and noise characteristics.
  - In a 4-way task (chance 25%), GMM dialect ID on *silence-only* chunks averaged 83.3%, against 82.0% on speech chunks. Egyptian silence chunks were recognized at 100%.
  - Channel normalization (equalizing each corpus's long-term average spectrum estimated from silence) aligned the means. Dialect ID on normalized silence stayed about as high, and cepstral mean normalization gave comparable results. The authors attribute the remaining cues to non-stationary noise (electrical bursts, handset noise) that stationary-noise suppression handles poorly.
  - On an in-house Pan-Arabic corpus collected consistently across dialects, silence-only 4-way accuracy was 24.7%, i.e., chance.
- Müller et al., "Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?" (ASVspoof 2021 workshop, arXiv 2106.12914). In ASVspoof 2019 LA, bona fide clips have longer leading and trailing silence than spoofed clips. A model that sees only leading-silence duration reaches up to 85% accuracy (EER 15.1%). Trimming silence raises RawNet2's EER from 3.6% to 15.5%. They suggest trimming with librosa or a forced aligner such as MFA — [arXiv](https://arxiv.org/pdf/2106.12914); [ISCA](https://www.isca-archive.org/asvspoof_2021/muller21_asvspoof.html)
- "The Impact of Silence on Speech Anti-Spoofing" (arXiv 2309.11827, 2023). Even after retraining on silence-free data, countermeasures still have trouble with silence-free spoofs. Per search summary — [arXiv](https://arxiv.org/pdf/2309.11827)
**Arabic dialect ID under domain shift (SSL era)**
- Sullivan, Elmadany, Abdul-Mageed, "On the Robustness of Arabic Speech Dialect Identification" (arXiv 2306.03789, June 2023; venue reported as Interspeech 2023 by search summary, not confirmed from the PDF). Read from the PDF — [arXiv](https://arxiv.org/abs/2306.03789)
  - ADI-17 test macro-F1: HuBERT-17 92.12; XLS-R-300m 90.20; DKU ResNet 94.9 (reported).
  - On the newly collected YouTube Dramas set (7 dialects, 24.8 h), HuBERT-17 scores **0.91 macro-F1** and HuBERT-17MSA 5.09, although the latter scores 89.39 on ADI17+MSA dev. Self-training on silver-labeled YouTube data raises Dramas F1 to 39.40–45.10.
  - On ADI-5 (broadcast; MSA segments excluded), HuBERT-17 scores 80.36 and HuBERT-17MSA 37.17. The authors conclude the MSA-trained model "has used channel differences between the MSA (broadcast television) and ADI-17 DA (YouTube) as a shortcut."
  - They note that dialect-ID tasks can be "prone to models overfitting to channel information," citing Bořil 2012, and use quantized HuBERT k-means units (from the layer with the best phone purity) as a more robust representation.
  - Recommendations: dialect-ID benchmarks should include zero- or few-shot out-of-domain tests, and standard-variety (MSA) training data should match the recording conditions and topics of the dialect data.
- Shon, Ali, Glass et al., "ADI17: A Fine-Grained Arabic Dialect Identification Dataset" (ICASSP 2020). Data crawled from YouTube channels, capped at about 100 h per channel "to avoid the data being biased to a specific channel or genre". Splits are by YouTube ID only, so the same speaker or programme can appear across sets. MGB-3 test is used as the mismatched-domain condition — [OpenReview PDF](https://openreview.net/pdf?id=eKgTpFsG5f_)
- Shon, Ali, Glass, "Domain Attentive Fusion for End-to-end Dialect Identification with Unknown Target Domain" (ICASSP 2019). Several networks, each trained on one domain, are fused by an attention layer weighted on outputs or hidden activations. Reported to work on unseen target domains (broadcast vs. YouTube). Numbers not retrieved — [arXiv](https://arxiv.org/pdf/1812.01501)
**Language ID corpora built from metadata**
- VoxLingua107 (Valk & Alumäe, SLT 2021). Segments come from YouTube videos labeled by the language of the title and description. Only the dev set is human-verified. Labels are therefore tied to channel and video — [arXiv](https://arxiv.org/pdf/2011.12998)
**Microphone mismatch in regional accent**
- Perez-Toro et al. 2026. Under microphone mismatch, Whisper nine-way regional UAR drops from 0.36 to 0.27 and three-way from 0.68 to 0.60. "Channel characteristics contribute, although regional information remains detectable" — [arXiv](https://arxiv.org/html/2610.06430)
**Noise robustness tests in dialect benchmarks**
- Voxlect adds Gaussian noise at 25/15/5 dB SNR. Whisper-Large and MMS-LID-256 degrade similarly at moderate SNR, and MMS-LID drops more at 5 dB. Channel and cross-domain conditions are not tested — [arXiv](https://arxiv.org/html/2508.01691v1)
**Speech enhancement can remove what downstream models use**
- "VoiceID Loss: Speech Enhancement for Speaker Verification" (arXiv 1904.03601). Enhancement objectives make no guarantee for downstream tasks, and their artifacts can make speaker verification worse — [arXiv](https://arxiv.org/pdf/1904.03601)
- "When De-noising Hurts: A Systematic Study of Speech Enhancement Effects on Modern Medical ASR Systems" (arXiv 2512.17562). Enhanced audio gave higher ASR error than the noisy original in all 40 configurations tested. Per search summary — [arXiv](https://arxiv.org/html/2512.17562)
**Domain-adversarial training (gradient reversal, GRL) and its limits**
- Ganin et al., "Domain-Adversarial Training of Neural Networks" (JMLR 2016). The standard GRL recipe. Not fetched — [arXiv 1505.07818](https://arxiv.org/abs/1505.07818)
- REDAT (arXiv 2012.07353): domain-adversarial training for accent-invariant RNN-T ASR. Shows that gradient reversal is equivalent to minimizing the Jensen–Shannon divergence between domain-conditional output distributions — [arXiv](https://arxiv.org/pdf/2012.07353)
- "Domain Adversarial Training for German Accented Speech Recognition" (DAGA 2023). A reversal applied only to accent-related gradients improves seen accents — [PDF](https://pub.dega-akustik.de/DAGA_2023/data/articles/000561.pdf)
- Limitation: gradient reversal "often fails to guarantee" that the nuisance variable (speaker) is removed — Li et al., "Speaker-invariant Affective Representation Learning via Adversarial Training" — [arXiv 1911.01533](https://arxiv.org/pdf/1911.01533). SISER (arXiv 2609.02941) argues gradient reversal only penalizes discriminability without constraining the target distribution, and proposes entropy maximization over source domains instead — [arXiv](https://arxiv.org/pdf/2609.02941)
- Duroselle, Jouvet, Illina, "Unsupervised regularization of the embedding extractor for robust language identification" (Odyssey 2020): a channel/domain-robust LID regularizer, cited by Sullivan et al. Not fetched; numbers unknown — reference in [arXiv 2306.03789](https://arxiv.org/abs/2306.03789)

### Inferences
- **Diagnostic to run first (from Bořil 2012):** train or probe our accent classifier on *non-speech only*: VAD-detected leading/trailing silence and inter-word gaps from AI Hub clips. If region is predictable well above chance (1/k), the per-region corpora carry recording signatures, and our 0.865→0.550 collapse when gaps are silenced is that signature being removed. The prediction collapsing to Gyeongsang suggests the model has a "default" region it falls back to when the channel cue disappears.
- **Gradient reversal against "corpus ID" is not usable for us**, because corpus = region label: removing corpus identity removes the target. Adversaries must target nuisance attributes that vary *within* region, e.g. recording-environment or device metadata if AI Hub provides it, SNR or reverb bins, VAD gap statistics, pseudo-session clusters. Alternatively, add standard-Korean and regional speech from *other* corpora (broadcast, conversation, free-talk) so each region appears in more than one recording channel. This is the cross-corpus fix that Sullivan et al. and He & Beigi both recommend.
- **Make silence uninformative:** following Müller et al. and Bořil, VAD-trim edges and replace inter-word gaps and silences in *all* training clips with a shared, randomized noise floor (or digital zero plus random room tone). Treat this as a canonicalization step, not just one augmentation among eight, so no region has a characteristic silence. Apply the same canonicalization at inference.
- **Browser noise suppression:** treat WebRTC/RNNoise-style NS, AGC and Opus encoding as part of the training channel (run training audio through the same browser-style chain), since enhancement can strip cues the model depends on (VoiceID-loss finding). Calibrate on in-browser recordings, not on AI Hub dev.
- The self-training result (dramas F1 0.91→~43–45) suggests pseudo-labeling in-domain (browser) audio partly recovers performance. He & Beigi caution that confidence filtering alone is insufficient for pseudo-labels under domain shift.

### Gaps
- I found no recent (2019–2026) paper quantifying a channel confound in CORAAL dialect classification (the sites were recorded in different eras), and no paper quantifying speaker overlap in Common Voice accent splits. CommonAccent (Interspeech 2023, up to 95% English accent accuracy) was found, but its speaker-overlap handling was not checked — [arXiv 2305.18283](https://arxiv.org/pdf/2305.18283)
- I found no study measuring the effect of browser noise suppression or WebRTC processing on accent or dialect ID specifically.
- I found no dedicated study of codec or VAD-gated-silence augmentation for dialect ID. The evidence comes from anti-spoofing (silence) and the general enhancement literature.

## Q3. Speaker leakage and label noise from birthplace labels

### Takeaway
Two 2026 papers say outright that file-level splits with speaker overlap overstate accent-ID accuracy, and that evaluation needs speaker-disjoint and cross-corpus tests. Embeddings that transfer across speaking style correlate with speaker-identity retrieval. Label noise from where speakers come from is acknowledged everywhere but rarely modeled. The best-quantified case is Arabic YouTube data, where 74.71% of channel-country labels did not match the speaker's actual dialect. Fixes in the literature include geolocation-model filtering, self-training, and regression to coordinates in place of hard region labels.

### Cited Findings
- He & Beigi, "Revisiting Accent Classification with DistilHuBERT: Speaker-Disjoint Evaluation and Cross-Corpus Generalization" (arXiv 2610.13440, Oct 2026; ICNLSP 2026, Trento):
  - File-level evaluation with speaker overlap likely overstates performance.
  - Models trained only on external data do not reliably generalize to L2-ARCTIC.
  - Balanced external-data training was the most promising supervised setting.
  - Confidence-based filtering alone is not enough for pseudo-labeling under domain mismatch.
  - Recommendation: evaluate on both speaker-disjoint and cross-corpus tests.
  - Absolute numbers were not in the abstract — [arXiv](https://arxiv.org/abs/2610.13440)
- Qu et al. (2609.10466) use 4-fold speaker-disjoint CV, not a single split, because a single split can be too optimistic or too pessimistic — [arXiv](https://arxiv.org/abs/2609.10466)
- AESRC2020 baseline repo: per-accent dev accuracy tracked individual speakers, and the dev set has too few speakers for the absolute numbers to be statistically meaningful. Per search summary — [GitHub](https://github.com/R1ckShi/AESRC2020)
- ADI-17 splits by YouTube ID, so the same speaker or programme can appear across train and test — [OpenReview PDF](https://openreview.net/pdf?id=eKgTpFsG5f_)
- Perez-Toro et al. 2026 on demographic and speaker information in the encoders:
  - Whisper encodes sex at 0.99 balanced accuracy and age with 4.4-year MAE.
  - Age plus sex alone predict region at 0.141 nine-way UAR (near chance), and adding them to Whisper doesn't help.
  - They report geolocation regression with a great-circle loss as an alternative to hard labels: 148 km median error matched, 432 speakers with geocoded hometowns.
  - [arXiv](https://arxiv.org/html/2610.06430)
- **Quantified label noise:** Sullivan et al. 2023 had a native expert hand-check YouTube utterances labeled by the channel's country ("surrogate labels"). 74.71% did not belong to the labeled country's dialect, and among these mislabeled utterances the model found the correct dialect for only 11.49%. They also grouped silver labels by confidence: low < 54.24%, medium 54.24–87.84%, high > 87.84% — [arXiv](https://arxiv.org/abs/2306.03789)
- Dialect labels "often originate from self-reports, which may contain labeling noise" (Voxlect). Where a corpus has no default split, Voxlect holds out 20% of speakers for testing — [arXiv](https://arxiv.org/html/2508.01691v1)
- Common Voice contributors describe their accents in highly diverse free-text ways ("Common Voice and accent choice: data contributors self-describe their spoken accents in diverse ways", ACM EAAMO 2023) — [ACM](https://dl.acm.org/doi/fullHtml/10.1145/3617694.3623258)
- "Extracting accent features in spoken Brazilian Portuguese without sociolinguistic labels" (arXiv 2605.30457, 2026). Birthplace, current residence and family origin help narrow things down, but other factors make such labels unreliable, so the authors learn accent features without labels. Per search summary — [arXiv](https://arxiv.org/html/2605.30457v2)
- "Scalable Controllable Accented TTS" (arXiv 2508.07426) filters Common Voice accent labels with a speech-geolocation model. An utterance is kept only if its predicted location falls inside the target accent's bounding box, and the same model labels unlabeled clips. Per search summary — [arXiv](https://arxiv.org/pdf/2508.07426)
- Perceptual dialectology: listeners' region of origin and mobility change how they categorize dialects (Clopper & Pisoni line of work). This concerns listeners, not speaker labels — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3060789/)

### Inferences
- Without speaker IDs, splitting by recording/session (what we do now) is a reasonable proxy, but leakage can remain if one speaker contributed several sessions. A practical upgrade, not taken from any paper above: cluster ECAPA/x-vector speaker embeddings within each region to make pseudo-speaker IDs, then (a) split by cluster and (b) cap clips per cluster so no single voice dominates a region.
- The same pseudo-speaker clusters can feed a speaker adversary or a per-speaker-balanced loss. Remember the Q2 caveat that gradient reversal may only partly remove speaker information.
- For movers (label noise), the literature's tools translate as:
  - confidence-banded self-training (Sullivan);
  - filtering with an independent model, such as a geolocation-style model (Accented TTS) or a leave-one-region-out ensemble that flags clips whose region posterior strongly disagrees with the label;
  - soft or ordinal targets, e.g. geographic regression or neighbor-region label smoothing (Voxlect: errors follow geography).
- Because a birthplace label says "probably region X" rather than "clearly accented", some region-labeled clips will sound standard. Training a hard region label on them teaches the model to call standard speech regional. This links directly to Q4.

### Gaps
- I found no accent/dialect-ID paper that explicitly models migrants or "residence vs. origin" label noise with numbers, apart from the Arabic surrogate-label audit above.
- The absolute speaker-overlap vs. speaker-disjoint numbers in He & Beigi were not retrieved; only the abstract was available.
- No method for building speaker-disjoint splits when speaker IDs are missing was found in the dialect-ID literature.

## Q4. Preventing false "regional" calls on standard speakers

### Takeaway
I found no paper that targets false "regional" calls on standard-variety speakers directly. The closest evidence: regional corpora contain many standard-variety utterances labeled as regional (Arabic SYR/SAU sets with MSA), standard-variety detection fails out of domain (85.87% of MSA missed), and accent is graded with strength. Standard tools are prior/threshold adjustment (Bayes decision with chosen priors, as in NIST LRE detection scoring), post-hoc calibration on deployment-domain data, logit adjustment, and selective classification. These are general methods, not dialect-specific results.

### Cited Findings
- Djanibekov et al., "Dialectal Coverage And Generalization in Arabic Speech Recognition" (arXiv 2411.05872 v3, May 2025; venue not confirmed): "Many dialectal sets, such as SYR and SAU, contain a lot of MSA utterances that are incorrectly identified as dialectal." Their error figure shrinks the SAU→MSA confusion 5× for clarity, so it is the largest single confusion — [arXiv](https://arxiv.org/pdf/2411.05872)
- Sullivan et al. 2023: the model "is not able to recognize 85.87% of the MSA utterances" in the human-audited out-of-domain sample. They attribute this to MSA in different topics or recording conditions, and call for MSA data covering diverse recording conditions — [arXiv](https://arxiv.org/abs/2306.03789)
- Kukk & Alumäe 2022: LID accuracy is inversely correlated with accent strength. Accent is graded, so mildly accented speakers sit near the decision boundary — [ISCA](https://www.isca-archive.org/interspeech_2022/kukk22_interspeech.html)
- He & Beigi 2026: confidence-based filtering alone is insufficient under domain mismatch, so confidence scores calibrated on the training domain don't carry over to a new channel — [arXiv](https://arxiv.org/abs/2610.13440)
- NIST LRE22 frames language recognition as *detection*: each segment gets a 14-dimensional vector of natural-log likelihoods, and decisions use application priors and costs. Exact Cdet/P_target values were not checked (from memory, LRE17/22 average Cprimary over P_target = 0.5 and 0.1; unverified) — [arXiv 2302.14624](https://arxiv.org/pdf/2302.14624); [plan](https://lre.nist.gov/uassets/3)
- General selective-classification and abstention references:
  - The threshold trades coverage (fraction classified) against accuracy on the covered subset — [search-summary source](https://www.researchgate.net/publication/353492014_The_Art_of_Abstention_Selective_Prediction_and_Error_Regularization_for_Natural_Language_Processing)
  - Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks" (NeurIPS 2017). Not fetched — [arXiv 1705.08500](https://arxiv.org/abs/1705.08500)
  - Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017; temperature scaling). Not fetched — [arXiv 1706.04599](https://arxiv.org/abs/1706.04599)
  - Menon et al., "Long-tail learning via logit adjustment" (ICLR 2021; adds or subtracts τ·log prior to logits). Not fetched — [arXiv 2007.07314](https://arxiv.org/abs/2007.07314)
  - "Know Your Limits: A Survey of Abstention in Large Language Models" (TACL 2025): calibration of confidence is a separate step before threshold-based abstention — [ACL Anthology](https://aclanthology.org/2025.tacl-1.26.pdf)

### Inferences
- Frame the coaching decision as **detection of "regional" against a "standard" null**, with an explicit asymmetric cost. Calling a Seoul speaker regional is worse for user trust than missing a mild accent.
  - Pick the operating point on *in-browser* Seoul recordings so the false-regional rate stays at or below a target, e.g. 5%. Our current 14–23% shows the AI Hub-dev threshold does not carry over.
  - Report results as a detection-error tradeoff (DET) curve and a coverage–risk curve, not just accuracy.
- Logit adjustment toward the standard class is the principled version of our "bias toward standard". Temperature scaling, fitted on browser-domain held-out data, should come before both the bias and the abstain threshold.
- Clean the "standard inside regional corpora" problem at the label level, as Djanibekov et al. saw with MSA in dialect sets:
  - run a standard-vs-regional detector over regional-corpus clips and relabel or drop confidently standard-sounding ones;
  - or train with a "standard-sounding" auxiliary label instead of forcing every region-corpus clip to its region.
- In a generative LALM, abstention can be an explicit output token ("판단 보류" / uncertain) trained on low-confidence or ambiguous clips. Alternatively, take the accent decision from the label-token probabilities with a calibrated threshold rather than greedy decoding (see Q5).

### Gaps
- I found no paper on false-positive dialect/accent detection for standard speakers, or on cost-sensitive dialect ID.
- I found no speech-specific selective-classification study for accent/dialect.
- The exact NIST LRE22 cost parameters were not verified.

## Q5. LLM-style generative classifiers: shortcut learning when the label is generated with the transcript, tag order, task mixing

### Takeaway
No published ablation directly compares tag-first and tag-last accent/dialect output in a speech LLM, so we would need to run one ourselves. The evidence that does exist points one way:
- Audio LLMs, including Qwen2.5-Omni and Qwen3-Omni, fall back on lexical content for paralinguistic judgments. Neutral words *lower* audio-only accuracy, and word-free items are near chance.
- Joint ASR + dialect-tag models put the tag first in the sequence (Whisper/ESPnet convention), where it can't condition on generated words. Letting the model infer the dialect tag beat forcing the gold tag for ASR, because the labels are coarse.
Tag-first ordering plus lexical-free training targets is therefore the safer default for our joint LALM.

### Cited Findings
**Lexical dominance in audio LLMs (emotion as proxy for paralinguistic accent)**
- Chen et al., "Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance" (arXiv 2510.10444, Oct 2025). LISTEN benchmark, 7,939 questions, 4 conditions. Models: Gemini 2.5 Flash and Pro, Qwen2.5-Omni-7B, Qwen3-Omni-30B, Baichuan-Omni-1.5 — [arXiv](https://arxiv.org/html/2510.10444v1)
  - Neutral-text items: text-only accuracy is 96.6% (Gemini 2.5 Pro) and 85.4% (Qwen2.5-Omni-7B), but audio-only accuracy is only about 34–35% for the best models.
  - **Qwen2.5-Omni-7B drops from 34.0% (audio-only) to 19.8% (text+audio)**, i.e., neutral words bias the prediction.
  - Paralinguistic items with no words: Qwen2.5-Omni-7B and Baichuan 22.7%, Qwen3-Omni-30B 21.0%, Gemini 2.5 Pro 15.7% (uniform chance 10%).
  - Emotion-mismatched (sarcasm) items: Gemini 2.5 Pro scores 31.8% (text), 36.9% (audio), 42.6% (text+audio).
- Corrêa et al., "Evaluating Emotion Recognition in Spoken Language Models on Emotionally Incongruent Speech" (arXiv 2510.25054; submitted to ICASSP 2026). Four spoken LMs "rely predominantly on textual semantics rather than speech emotion"; the EMIS dataset is released. Per search summary, audio-label accuracy is near chance on incongruent items, while agreement with the text often exceeds 95% for explicit sentences — [arXiv](https://arxiv.org/abs/2510.25054)
- "When Tone and Words Disagree…" (arXiv 2601.04564): encoders pretrained on speech–text alignment (Whisper, CLAP) default to literal sentiment, while SSL encoders (HuBERT, WavLM) mix phonetic and prosodic information. The follow-up TWIN-SER benchmark (arXiv 2609.04236) reports Whisper at 47.26% weighted accuracy and attributes the errors to reliance on semantics. Per search summary — [arXiv 2601.04564](https://arxiv.org/html/2601.04564v1); [arXiv 2609.04236](https://arxiv.org/html/2609.04236)
**Dialect/language tags inside the decoder sequence**
- Djanibekov et al. (arXiv 2411.05872): dialect tokens are added to the vocabulary and *prepended*, as `<S> DIALECT T1 … Tn </S>`. Read from the PDF — [arXiv](https://arxiv.org/pdf/2411.05872)
  - Average WER, read from the last row of Table 7: no dialect ID 32.63, dialect *forcing* (gold tag) 34.09, dialect *inference* (model predicts the tag) 31.45.
  - Dialect-ID accuracy from the inferred token is about 90%.
  - Their explanation: "the model learns dialectal patterns that do not perfectly align with the dialect ID", since labels are coarse country-level approximations.
  - Low-resource KUW gets predicted as its closest high-resource neighbor, SAU.
  - Caveat: one sentence in the paper says "on average, dialect forcing leads to the lowest WER", which contradicts both the table and the earlier text. Internal inconsistency.
- Lonergan et al., "Low-resource speech recognition and dialect identification of Irish in a multi-task framework" (Odyssey 2024). Hybrid CTC/attention with intermediate CTC (InterCTC), Conformer/E-Branchformer encoders. Joint ASR+DID improves DID by 10.8% relative over ECAPA-TDNN, and WER approaches a dedicated TDNN-HMM. A search summary says a dialect tag is prepended to the target text; the abstract does not confirm this — [arXiv](https://arxiv.org/abs/2405.01293)
- Convention: Whisper predicts the language token *before* the transcript (Radford et al. 2022, not fetched — [arXiv 2212.04356](https://arxiv.org/abs/2212.04356)). The joint LID+ASR line of Watanabe et al. (ASRU 2017) also inserts the language tag at the start (not fetched).
- A Google multi-dialect LAS system (patents US 11238845 / 11900915 / 12254865): joint dialect ID alone left about 5% dialect prediction error, and feeding dialect vectors to the decoder improved WER by 3.1–16.5%. Patent text, per search summary — [USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11238845)
- "Teaching LLMs to Hear Who Spoke What: Metadata-Supervised Pretraining for Encoder-Free Speech-LLMs" (Shi, Fan, Sivasankaran, Deng, Li; arXiv 2610.01695, Oct 2026). Speech attributes (speaker, emotion, …) are supervised alongside content. A search summary says the fixed order is attributes first (gender, emotion, language, speaker, then content), but no order ablation was visible in the abstract — [arXiv](https://arxiv.org/abs/2610.01695)
- MMGER (arXiv 2405.03152): frozen Qwen-7B with accent embeddings for joint accent recognition + ASR on KeSpeech. Per search summary, accent accuracy is about 90% on dev but about 77% on test; the reason for the gap was not checked — [arXiv](https://arxiv.org/pdf/2405.03152)
- "Decoupling and Interacting Multi-Task Learning Network for Joint Speech and Accent Recognition" (arXiv 2311.07062). Exists; numbers not retrieved — [arXiv](https://arxiv.org/pdf/2311.07062)
- AudioBench (NAACL 2025) builds its accent task from VoxCeleb1 metadata: predicting the speaker's likely origin, which is nationality used as a stand-in for accent — [ACL Anthology](https://aclanthology.org/2025.naacl-long.218.pdf)

### Inferences
- **Tag-first vs. tag-last.** With tag-last (`transcript … <accent=X>`), the accent token attends to the model's own generated words. Dialect words such as "-예", "머꼬" or "거시기" then become a direct, trivially learnable path to the label, which is our F1 0.216 failure mode. Tag-first (`<accent=X> transcript …`) removes that explicit text path. The label must then come from the audio, plus whatever lexical information the encoder carries implicitly, so tag-first is necessary but not sufficient. Recommended ablation set, using our existing counterfactual tests:
  - tag-first
  - tag-last
  - accent-only prompt (separate task)
  - all three evaluated on dialect-word-free clips, silenced-gap clips, browser-NS clips and Seoul browser speakers.
- **Decouple the supervision sources.** Train the accent token only on clips without dialect words (our existing filter). On clips *with* dialect words, mask the accent target or replace it with a separate "dialect vocabulary present" field. That way lexical cues are reported as lexical, never as pronunciation. LISTEN-style results suggest this matters for the Qwen-Omni family in particular, which leans toward text.
- **Read the label as a score, not a greedy token.** Constrain decoding at the tag position to the set of accent tokens, take the softmax over them, and apply the calibrated threshold, standard-class prior and abstain rule from Q4. This keeps the existing abstention design inside the LALM.
- **Don't force gold dialect tags for ASR.** Djanibekov et al. found forcing the coarse gold tag hurt WER (34.09 vs. 31.45 inferred). Birthplace-labeled region tags are also coarse and noisy (Q3), so during ASR training let the model infer the tag, or drop it, rather than teacher-forcing it on every clip.
- **Mix in counterfactual training data**, mirroring the incongruent-emotion benchmarks:
  - standard-Korean transcripts read with regional prosody (if TTS or voice conversion allows);
  - regional-speaker clips whose dialect words were normalized in the transcript target;
  - silence-canonicalized and browser-processed versions of each clip with the *same* accent label.
  These make words and channel uninformative about the label.
- **Optional hybrid.** If generated tags stay lexically biased, attach a small discriminative accent head to pooled audio-encoder states, optionally with phone-aligned segments as in Q1. Use it as the accent score, and let the LALM handle the transcript and the explanation text.

### Gaps
- I found no paper that ablates tag-first vs. tag-last (or transcript-then-label chain-of-thought) for accent, dialect or any paralinguistic label in a speech LLM.
- I found no paper that measures lexical leakage in Qwen2.5-Omni, Qwen3-Omni or Qwen3-ASR for *accent* specifically. The evidence is for emotion only.
- Task-mixing ratio ablations (ASR vs. accent vs. paralinguistic tasks) for accent in LALMs were not found.
- The Qwen2-Audio accent numbers in Perez-Toro et al. and AudioBench were not retrieved.
