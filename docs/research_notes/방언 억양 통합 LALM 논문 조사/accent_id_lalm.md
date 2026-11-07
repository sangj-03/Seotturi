# 음성 기반 모델·LALM의 억양/지역 방언 식별, 그리고 ASR + 억양 식별을 한 모델에서 하는 방법

조사일: 2026-10-14. 2021~2026년 자료를 우선했다. 원문(HTML/PDF 본문)에서 직접 확인한 수치는 그대로 적었다. 검색 요약이나 초록에서만 본 것은 "(미검증: …)"으로 표시했다.
용어: LALM = large audio language model. AR = accent recognition. DID = dialect identification. RP = representation probe(인코더 은닉 상태 위 분류기). TP = transcript probe.

---

## 1. LALM과 음성 인코더는 억양·방언 식별 벤치마크에서 얼마나 하는가? 음향을 쓰는가, 어휘에 기대는가? (Qwen2-Audio / Qwen2.5-Omni / Qwen3-Omni / Gemma 결과)

### Takeaway
제로샷 LALM은 지금까지 나온 억양·방언 식별 벤치마크에서 모두 우연 수준이거나 그보다 조금 나은 정도다. 같은 데이터에서 텍스트만 쓰는 cascade(Whisper→Llama3)가 Qwen2-Audio를 이겼고, Whisper 인코더 위에 얹은 작은 probe가 Qwen3-Omni를 크게 앞섰다(L1-arc F1 84.3 대 31.8). 또 LALM은 특정 라벨 하나로 몰아주는 강한 사전 편향(attractor)을 보인다. 서울 화자 15명 중 14명을 지역 방언으로 판정한 Gemma 실패와 같은 양상이다. 정보는 인코더에 있지만 LLM의 "읽어내기(readout)"에서 무너진다는 증거가 있다.

### Cited Findings
**벤치마크 수치(제로샷 LALM)**
- **AudioBench** (Bin Wang 외, NAACL 2025). VoxCeleb1-Accent 과제(화자 국적 메타데이터로 "억양으로 보아 어디 출신인가"를 묻는 문항, 4.8k개, model-as-judge 점수, 높을수록 좋음) 결과는 다음과 같다. SALMONN **28.06**, Qwen-Audio-Chat **45.70**, WavLLM **37.65**, Qwen2-Audio-Instruct **29.19**, 텍스트 전사만 쓰는 cascade Whisper-large-v3 + Llama-3-8B-Instruct **39.33**. 오디오를 듣지 않는 cascade가 Qwen2-Audio와 SALMONN보다 높았다. 저자들은 일반론으로 "paralinguistic 과제에서 AudioLLM이 대체로 cascade보다 낫지만 늘 만족스럽지는 않다"고 썼다. 같은 표에서 VoxCeleb1-Gender는 Qwen2-Audio 99.12, cascade 53.41이었다. 즉 성별처럼 순수 음향 과제에서는 오디오 모델이 확실히 앞서는데 억양에서는 그렇지 않다 — [AudioBench, NAACL 2025](https://aclanthology.org/2025.naacl-long.218.pdf)
- **Dynamic-SUPERB Phase-2** (ICLR 2025). Accent Classification(AccentDB Extended)을 LLM-judge 정확도(LLM-C)로 쟀다. GPT-4o judge 표(Table 5) 기준으로 Whisper-LLaMA 17.50%, LTU-AS 5.50%, SALMONN-7B 3.00%, SALMONN-13B 4.50%, Qwen-Audio-Chat 26.50%, **Qwen2-Audio-7B-Instruct 14.00%**, WavLLM 7.00%, MU-LLaMA 27.00%, GAMA-IT 4.00%였다. LLaMA-3.1-8B judge 표(Table 6)에서는 Qwen2-Audio가 14.00%로 같았고 MU-LLaMA 37.00%, Qwen-Audio-Chat 24.00%였다. 30%를 넘는 모델이 사실상 없다 — [Dynamic-SUPERB Phase-2](https://arxiv.org/pdf/2411.05361)
- **MMSU** (ICLR 2026). 47개 과제 중 "Accent Identification"은 영어 억양 13종(Singapore, Hong Kong, Australia, India, Kenya, Nigeria, US, South Africa, UK, Philippines, Ireland, Canada, NZ) 4지선다이며 Perception/Phonology로 분류된다. 과제별 억양 점수는 본문에서 찾지 못했다. 집계 점수는 다음과 같다. Perception-Phonology에서 Qwen2.5-Omni-7B 37.33, Qwen2.5-Omni-3B 38.73, Qwen2-Audio-Instruct 32.87, Kimi-Audio 42.30, GPT-4o-Audio 41.56, Gemini-1.5-Pro 53.60. 무작위는 25.70, 사람은 94.32. Perception-Paralinguistics에서 Qwen2.5-Omni-7B 39.35, GPT-4o-Audio 21.44, 무작위 26.10. 오류 분석에서 억양은 "lacks intonation knowledge of different English accents"인 지식 부족 오류의 예시로 쓰였다. 결론에는 "performance in specialized domains is constrained by insufficient domain-specific knowledge (e.g., accent)"라고 적혀 있다 — [MMSU](https://arxiv.org/pdf/2506.04779)
- **PRiSM** (ACL 2026, Bharadwaj 외). Qwen3-Omni를 다룬 가장 직접적인 자료다. 제로샷 결과는 다음과 같다.
  - **Qwen3-Omni-Instruct**: L1-eda(EdAcc L1 분류) F1 **18.2**, L1-arc(CMU ARCTIC + L2-ARCTIC) F1 **31.8**, GEO-v(인도 힌디 벨트 방언 지리 추정, Recall@1) **5.3**.
  - **Gemini 2.5 Flash**: 32.7 / 43.9 / 6.5.
  - naive baseline: 6.3 / 14.3 / 3.3.
  - 인코더 RP(최종층 + attention pooling + MLP): **Whisper-small 24.3 / 84.3 / 35.0**, WavLM-base 14.4 / 58.3 / 23.5.

  GEO-v에서 Qwen3-Omni는 "predicts New Delhi for nearly all inputs"였다. 저자 결론은 LALM이 "strongly biased toward higher-resourced dialects"이고 "lack the fine-grained acoustic perception"이라는 것이다. Gemini는 Romance 억양으로 쏠렸다. Slavic/Balkan 화자의 25.8%, South Asian 화자의 28.5%를 Romance로 오분류했다. thinking 모드를 켜면 F1이 32.7에서 24.9로 오히려 떨어졌다. 오답 추론문의 87%가 "Spanish/Italian/Portuguese"를, 65%가 "syllable-timed rhythm"을 언급했다 — [PRiSM](https://arxiv.org/html/2601.14046v2)
- **Vox-Profile** (Feng 외, USC/JHU, arXiv 2505.14648, 2025). LALM 비교는 없는 인코더 벤치마크다. 11개 영어 억양 데이터셋을 썼고, broad(3클래스: 북미/영국제도/기타)와 narrow(16클래스) 두 수준으로 평가했다. 결과는 다음과 같다.

  | 인코더 | broad 정확도 | broad macro-F1 | narrow 정확도 | narrow macro-F1 |
  |---|---|---|---|---|
  | ECAPA-TDNN | 74.6 | 0.638 | 42.3 | 0.363 |
  | HuBERT-Large | 85.4 | 0.841 | 49.1 | 0.443 |
  | WavLM-Large | 92.9 | 0.884 | 65.7 | 0.617 |
  | Whisper-Tiny | 89.1 | 0.831 | 54.1 | 0.484 |
  | Whisper-Small | 90.9 | 0.863 | 70.4 | 0.663 |
  | **Whisper-Large** | 92.8 | **0.889** | 75.8 | **0.724** |

  British Isles 세부 억양에서는 CommonAccent F1 0.581 대비 Vox-Profile 0.770이었다. 사람 평가자는 ParaSpeechCaps의 억양 기술을 Vox-Profile 예측보다 더 정확하다고 판단했다. 저자들도 억양 예측을 "challenging"이라고 썼다 — [Vox-Profile](https://arxiv.org/html/2505.14648v1)
- **CommonAccent** (Zuluaga-Gomez 외, Interspeech 2023). XLSR/ECAPA를 Common Voice 영어 억양으로 fine-tune해 최고 약 95% 정확도를 냈다. XLSR 임베딩은 음운 유사성과 지리적 근접성에 따라 군집했다(England/Wales/Scotland) — [CommonAccent](https://arxiv.org/pdf/2305.18283)

**음향 대 어휘 의존(억양 외 paralinguistic 과제 포함)**
- **LISTEN** (Chen 외, EACL 2026). 감정 과제로 어휘와 음향 단서를 분리해 쟀다. 억양 과제는 아니다.
  - Qwen2.5-Omni-7B, Neutral-Text 조건: 텍스트만 85.4, 오디오만 **34.0**, 텍스트+오디오 **19.8**. 전사를 같이 주면 오히려 더 나빠졌다.
  - Qwen3-Omni-30B, 같은 조건: 85.4 / 29.3 / 25.3.
  - 어휘가 없는 Paralinguistic 조건(우연 12.5): Qwen2.5-Omni-7B **22.7**, Qwen3-Omni-30B **21.0**, Gemini 2.5 Pro 15.7.
  - 결론은 "current LALMs largely 'transcribe' rather than 'listen'", "In paralinguistic settings, performance approaches chance" — [LISTEN, EACL 2026](https://arxiv.org/html/2510.10444v2)
- **Where Does the Sound Go?** (Jo 외, SNU/KAIST/NAVER Cloud, arXiv 2609.05871, 2026). LLM으로 Qwen3.5-4B를 썼다. 인코더·projector·LM 각 층에 probe를 붙여 정보가 어디서 사라지는지 추적했다.
  - Whisper-Small 인코더 기준 linear probe 매크로 평균은 **83.3**인데, 같은 정보로 LLM이 객관식에 답하면 **38.1**이었다(Whisper-Tiny는 79.5 대 36.4).
  - 인코더에서 projector로 갈 때 probe 회수율은 오히려 오르고, LM 깊이를 거치면서도 대체로 유지된다. 실패는 출력 readout 단계에 있다.
  - 선택지 글자에 해당하는 출력 행(W_U)만 fine-tune해도 MCQA가 평균 **30~42pp** 회복됐다. 반면 LM LoRA는 "limited, encoder-dependent" 효과에 그쳤다.

  단, 이 논문은 억양을 probe하지 않았다(감정·소리·음악 과제) — [Where Does the Sound Go?](https://arxiv.org/html/2609.05871v1)
- **Activation Steering for Accent Adaptation in LALMs** (Sun 외, arXiv 2603.05813, 2026). Qwen2-Audio-7B 인코더(Whisper식 32층)를 분석해 "accent information concentrates in a narrow band of middle encoder layers"라고 결론지었다(층별 내용은 3번 질문 참조) — [Activation Steering, arXiv 2603.05813](https://arxiv.org/pdf/2603.05813)

**모델별 검색 결과**
- Qwen2-Audio: AudioBench 29.19, Dynamic-SUPERB 14.00%, MMSU Phonology 32.87.
- Qwen2.5-Omni-7B: MMSU Phonology 37.33 / Para 39.35, LISTEN Para 22.7.
- Qwen3-Omni: PRiSM 제로샷 L1 F1 18.2/31.8, 방언 지리 추정 5.3. LISTEN Para 21.0.
- Gemma(오디오 입력 Gemma 3n/4 계열): 억양·방언 벤치마크 결과를 하나도 찾지 못했다(Gaps 참조).
- 출처: 위 각 항목.

### Inferences
- 제로샷 LALM에게 억양을 바로 묻는 방식은 쓸 수 없다. AudioBench, Dynamic-SUPERB, MMSU, PRiSM 네 벤치마크가 같은 방향을 가리킨다. Qwen2.5-Omni, Qwen3-Omni, Qwen3-ASR 중 무엇을 고르든 지도 학습(SFT)이 전제다.
- 텍스트만 쓰는 cascade가 Qwen2-Audio보다 높았던 것(39.33 대 29.19)은 LALM의 "억양" 답이 어휘·주제·사전 확률에서 상당 부분 나온다는 정황 증거다. LISTEN에서 전사를 같이 주면 성능이 오히려 떨어진 것도 같은 방향이다. 우리 테스트 조건(방언 어휘가 없는 클립)은 바로 이 지름길을 막는 조건이다. 그래서 Gemma 방식이 방언 단어 없는 클립에서 F1 0.216으로 무너진 것과 들어맞는다.
- PRiSM의 "New Delhi" 쏠림과 Gemini의 Romance 쏠림은 LALM이 특정 라벨로 붕괴하는 경향을 보여준다. Gemma가 서울 화자 14/15를 지역으로 판정한 것도 같은 현상(라벨 사전 편향)으로 보인다. 그렇다면 5번 질문의 사전 보정(contextual/batch calibration, 클래스 균형)이 직접적인 처방이다.
- "Where Does the Sound Go?"에 따르면 정보는 LLM 안에 남아 있고 readout이 문제다. 따라서 라벨 토큰의 logit을 직접 읽고 해당 출력 행이나 작은 head를 학습하면 생성 방식보다 확률과 정확도가 모두 나을 가능성이 높다.

### Gaps
- MMSU의 억양 과제별 모델 점수 표는 본문 추출에서 찾지 못했다(부록 그림에 있을 가능성이 있다).
- AIR-Bench, SAKURA, VoxEval, SpeechCaps 계열에 억양 과제가 있는지는 이번에 확인하지 못했다. SAKURA는 성별·언어·감정·동물 4개 속성으로 알고 있으나 이번 세션에서 검증하지 않았다. VoxEval은 지식 QA 벤치마크로 알고 있으나 역시 미검증이다.
- Gemma 계열 오디오 모델의 억양·방언 결과는 찾지 못했다.
- 한국어 억양/방언을 다룬 LALM 벤치마크는 찾지 못했다.
- "같은 문장을 다른 억양으로" 통제한 LALM 억양 실험(어휘 통제)은 찾지 못했다. LISTEN은 감정 과제, PRiSM은 추론문 분석으로 간접 증거만 준다.

---

## 2. ASR과 억양/방언 식별을 한 모델에서: 태그를 먼저 생성하는 방식 vs multi-task head. 어느 쪽이 억양 정확도가 높고, ASR을 해치는가?

### Takeaway
같은 데이터(AESRC2020, 8개 억양)에서 직접 비교한 연구가 있다. decoder 출력 앞에 억양 태그를 붙이는 방식(STJR)은 ASR WER을 유지했다(6.6 대 기준 6.7). 인코더 위에 억양 head를 둔 multi-task(MTJR)는 억양 정확도가 3pt 더 높았지만(75.2 대 72.2) WER이 약간 나빠졌다(7.1). 태그는 뒤보다 앞에 두는 편이 낫다는 팀 보고가 있다. 모델이 직접 추론한 태그는 ASR에 해가 없거나 도움이 됐지만, 태그를 강제하거나 틀린 태그가 들어가면 ASR이 손상됐다. 두 방식 모두 ASR로 사전학습된 인코더가 억양 정확도의 핵심이었다.

### Cited Findings
**AESRC2020** (Accented English Speech Recognition Challenge, Interspeech 2020 챌린지. 개요 논문 Shi 외는 ICASSP 2021)
- 데이터는 Accent160이다. 억양 8종 × 20시간 = 160시간이며, 테스트에는 학습에 없던 억양 2종(Spanish, Canadian)이 포함됐다 — [AESRC 개요](https://ar5iv.labs.arxiv.org/html/2102.10233)
- AR 트랙 기준선: 인코더를 ASR 모델로 초기화하면 AR이 "significantly improved"였다. 최고 기준선 ASR-init-Transformer-12L은 CV set **76.1%**였다. 6층·12층 인코더를 처음부터 학습하면 과적합했다 — [AESRC 개요](https://ar5iv.labs.arxiv.org/html/2102.10233)
- 상위 팀은 다음과 같다.
  - 1위 S2: PPG 특징 + TTS 증강, TTS로 "improved absolutely by 10%".
  - 2위 E2: Transformer ASR이 억양 태그를 전사와 함께 출력했다. "태그를 앞에 두는 것이 끝에 두는 것보다 낫다(outperforms tagging at the end)".
  - ASR 트랙 T2: AR multi-task 인코더로 WER 3% 상대 감소.
  - 최저 WER은 Q2의 4.06%.
  - 개요 논문 결론: "Explicit accent related optimizations were rarely used" — [AESRC 개요](https://ar5iv.labs.arxiv.org/html/2102.10233)
- AR 3위 팀(Accent Recognition with Hybrid Phonetic Features)은 ASR 보조 과제와 음소 특징을 함께 써서 기준선 대비 상대 7.28% 개선을 얻었다. ASR 보조 과제가 없으면 AR이 학습 데이터의 화자 라벨에 과적합했다 — [arXiv 2105.01920](https://ar5iv.labs.arxiv.org/html/2105.01920)

**Zhang 외, "E2E-based Multi-task Learning Approach to Joint Speech and Accent Recognition"** (arXiv 2106.08211, 2021, 신장대·NTU. 게재처는 미검증). AESRC 8개 억양(US, UK, CHN, IND, JPN, KR, PT, RU)을 썼다.
- 단일 과제 AR: ASR 사전학습이 없으면 최고 **32.6%**(Transformer-12L)였다. ASR로 사전학습한 뒤에는 66.6%, Librispeech까지 넣어 사전학습하면 **69.8%**로 두 배 이상이 됐다.
- **태그 생성(STJR) vs multi-task head(MTJR)**, 테스트 셋 기준:

  | 방식 | 시스템 | WER | ACC |
  |---|---|---|---|
  | Transformer ASR(기준) | M2 | 6.7 | – |
  | STJR(억양 라벨을 ASR 출력 앞이나 뒤에 붙임) | M5 | 6.6 | 72.2 |
  | MTJR(인코더 pooling + 억양 softmax head, λ=0.1) | M8 | 7.1 | 75.2 |

  사전학습 없이 비교하면 MTJR(M6: WER 9.5, ACC 71.0)이 STJR(M3: 10.6, 69.7)보다 둘 다 좋았다. 억양 loss 가중 λ는 작게(0.1) 줄 때 가장 좋았다.
- 태그 위치: "output a accent label before actual ASR output or append an accent label after … We have not observed much WER difference".
- 공유 층 수(12층 인코더 중 몇 층을 공유하는가):

  | 공유 층 수 | ACC (dev / test) | WER (dev / test) |
  |---|---|---|
  | 3층 | 64.8 / 54.6 | 거의 일정(8.6~8.7 / 9.9~10.1) |
  | 6층 | 78.3 / 66.0 | |
  | **9층** | **80.2 / 71.4** | |
  | 12층 | 77.1 / 70.8 | |

  즉 9층까지만 공유하고 억양 head를 붙이는 부분 공유가 가장 좋았다 — [Zhang 외 2021](https://arxiv.org/pdf/2106.08211)

**DIMNet** (Shao, Guo, Yan, Hu, Xie, IEEE/ACM TASLP). AR과 ASR을 서로 다른 단위(AR은 세밀한 단위, ASR은 거친 단위)로 분리했다. CTC 정렬 텍스트를 AR에 주고, 억양 임베딩을 ASR 인코더와 decoder에 넣는다. 공개 기준선 대비 영어에서 AR 상대 +21.45%, WER 상대 −32.33%였고, 중국어에서 AR 상대 +28.53%, WER 상대 −14.55%였다. 절대 수치와 데이터셋 이름은 초록에 없다(미검증: 영어는 AESRC, 중국어는 KeSpeech로 추정) — [DIMNet, arXiv 2311.07062](https://arxiv.org/abs/2311.07062)

**아랍어: 방언 토큰을 decoder 맨 앞에 붙인 연구** (Djanibekov 외, MBZUAI/STC, arXiv 2411.05872, 2024). ArTST(SpeechT5 계열) 기반이고 12개 방언 약 1,501시간을 썼다. 형식은 `<S> DIALECT T1 … Tn </S>`이다.
- 방언 식별 정확도는 "around 90%"였다.
- 매크로 평균 WER: 방언 ID 없음 32.63%, **방언 태그 강제 34.09%(악화)**, **모델이 직접 추론 31.45%(개선)** — [Arabic dialect tokens](https://arxiv.org/html/2411.05872v1)

**인도어 8개 언어** (Ghosh 외, IISc, arXiv 2607.02862, 2026). 평균 수치는 다음과 같다.

| 시스템 | DID 정확도 | CER | WER |
|---|---|---|---|
| Base-ASR(방언 정보 없음) | – | 4.81 | 18.38 |
| 방언 ID 접두 ASR-DID | 79.19 | 4.73 | 18.17 |
| 제안(bottleneck + CTC 임베딩 RoBERTa 융합, 텍스트 앞 방언 ID 제거) | **81.63** | **4.65** | **17.73** |

방언 ID가 틀린 발화에서는 기존 방식이 CER 6.01 / WER 21.85, 제안 방식이 5.68 / 20.72였다. 저자들의 비판은 두 가지다. 접두 방언 토큰이 틀리면 ASR에 "false context"를 학습시키고, DID 블록에서 ASR로 gradient가 흐르지 않는다. 선행 연구도 정리했다. 일본어 Imaizumi 외는 DID는 좋아졌으나 최고 DID 모델이 ASR을 악화시켰고, 아일랜드어 Lonergan 외는 intermediate CTC로 DID를 개선했으나 ASR이 "slight degradation"을 보였다 — [Indian joint DID+ASR](https://arxiv.org/html/2607.02862)

**일본어 Imaizumi, Masumura, Shiota, Kiya** (APSIPA Trans. SIP 2022). 6개 방언 + 표준어로 DID→ASR, ASR→DID, 동시 3가지 배치를 비교했고 단일 과제보다 둘 다 좋았다고 보고했다(수치 미확인) — [APSIPA Trans 2022](https://www.nowpublishers.com/article/Details/SIP-2021-0045)

**아일랜드어** (Lonergan 외, arXiv 2405.01293). 저자원 ASR + 방언 식별 multi-task 연구다(수치 미확인) — [arXiv 2405.01293](https://arxiv.org/pdf/2405.01293)

**대규모 모델에서 "태그 먼저" 형식을 쓰는 사례**
- **Qwen3-ASR** (기술 보고서, arXiv 2601.21337, 2026)
  - 출력 형식: 언어 태그 뒤에 전사가 온다("language English" + `<asr_text>` + 전사). 음성이 없으면 "None"을 출력한다. 별도 LID head는 없고 생성으로 LID를 수행한다.
  - 구조: AuT 인코더(1.7B 모델은 300M, 0.6B 모델은 180M, 12.5Hz) + Qwen3-1.7B/0.6B.
  - 학습 단계: AuT 사전학습 → Omni 사전학습 → ASR SFT → GSPO 강화학습. 각 단계에서 인코더를 동결했는지는 밝히지 않았다.
  - LID 정확도 평균: Whisper-large-v3 94.1, Qwen3-ASR-0.6B 96.8, **Qwen3-ASR-1.7B 97.9**. 남은 오류는 주로 ms/id 혼동이다.
  - 지원 범위: 52개 언어·방언, 그중 중국 방언 22종(안후이·둥베이·쓰촨 등 18종 + 광둥어 2종 + 우어 + 민난어).
  - 방언 CER(1.7B): KeSpeech 5.10, WenetSpeech-Chuan easy/hard 11.99/21.63.
  - **한국어 지원**(1.7B): CommonVoice-ko 5.88, Fleurs-ko 2.57.
  - 16개 억양 영어 대화(Dialog-Accented English) WER 16.07로 비교 시스템 중 최저.
  - 출처: [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337)
- **Dolphin** (Dataocean AI + 칭화대, arXiv 2503.20212). 언어 토큰과 지역 토큰 2단계(예: `<zh><CN>`)를 쓴다. 동방 40개 언어 + 중국 방언 22종, 21만 시간 이상, CTC-Attention(E-Branchformer)이며 언어·지역을 강제할 수도 자동 감지할 수도 있다 — [Dolphin README](https://huggingface.co/DataoceanAI/dolphin-small/blob/refs%2Fpr%2F6/README.md). 후속 Dolphin-CN-Dialect는 방언 전용 토큰 80개를 두어 방언 인식 정확도를 원래 Dolphin 대비 38% 개선했다고 한다(미검증: 2차 요약) — [emergentmind 2605.08961](https://www.emergentmind.com/papers/2605.08961)
- **KeSpeech** (NeurIPS 2021 Datasets & Benchmarks). 표준 중국어 + 하위 방언 8종, 1,542시간, 화자 27,237명이다.
  - 하위 방언 식별(6클래스, 표준어 제외) 정확도: Kaldi x-vector 56.34%, **ResNet-34 61.13%**, ECAPA-TDNN 60.77%. 저자들은 "most of the subdialects are mixed up"이라고 썼다. 북방 하위 방언끼리는 너무 비슷해 구분이 어렵다.
  - ASR: 전체 데이터로 학습하면 CER이 표준어 6.1, 하위 방언 9.6~15.9였다. 표준어 데이터가 하위 방언 ASR을 돕는다 — [KeSpeech 논문](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/0336dcbab05b9d5ad24f4333c7658a0e-Paper-round2.pdf)

### Inferences
- **태그를 먼저 생성하는 방식은 ASR을 거의 해치지 않는다.** STJR은 기준과 WER이 같았고(6.6 대 6.7), 아랍어에서 추론 태그는 오히려 WER을 1.2pt 낮췄다. 억양 정확도만 보면 인코더 위 분류 head(MTJR)가 3pt 정도 낫다. 따라서 "한 모델" 요구와 정확도를 함께 맞추려면 이렇게 할 수 있다. (a) 생성 출력은 Qwen3-ASR 형식처럼 `지역: X` + 전사(태그 먼저)로 둔다. (b) 학습 때는 인코더 중상위층 pooling 위에 보조 억양 head를 λ≈0.1로 같이 학습한다. (c) 추론 확률은 태그 위치의 라벨 토큰 logit이나 보조 head 중 보정이 좋은 쪽을 쓴다. 이 하이브리드 구성은 근거 논문들을 조합한 추론이다.
- 태그를 앞에 두는 것이 "억양 판단이 전사 텍스트에 끌려가는 것"을 줄이는지는 직접 증거가 없다. 다만 태그를 뒤에 두면 모델이 방금 생성한 전사(방언 어휘)를 조건으로 판단하게 된다. 이는 우리가 피하려는 어휘 지름길을 더 쉽게 만든다. AESRC E2 팀이 앞쪽 태그가 낫다고 보고한 것과도 방향이 맞는다.
- 태그를 강제하면(사용자가 지역을 지정) ASR이 나빠질 수 있다(아랍어 +1.5pt). 틀린 태그는 ASR 오류를 키운다(인도어). 코칭 앱에서 정답 지역을 강제 입력하는 기능은 피하고 모델 추론에 맡기는 것이 안전하다.
- ASR 사전학습이 억양 정확도를 두 배로 만든다(32.6→69.8). ASR이 이미 학습된 LALM(Qwen3-ASR, Qwen2.5-Omni)에서 출발하는 것은 유리한 조건이다.
- Qwen3-ASR-1.7B는 이미 "언어 태그 → 전사" 템플릿을 쓰고 한국어를 지원한다. 그래서 "language Korean" 자리를 "language Korean-Gyeongsang" 같은 세분 태그로 바꾸는 설계가 가장 손이 덜 간다. 1.7B 크기라 인코더까지 포함한 전체 fine-tune도 현실적이다(추론).

### Gaps
- AESRC 상위 팀 표(Table 2/3)의 팀별 AR 정확도와 WER 수치는 추출하지 못했다.
- Imaizumi 외(일본어)와 Lonergan 외(아일랜드어)의 수치는 확인하지 못했다.
- LALM(LLM 디코더) 위에서 "방언 태그 + 전사"를 학습해 태그 정확도와 WER을 보고한 논문은 찾지 못했다. 위 결과는 모두 Transformer/Conformer AED나 CTC 기반이다.
- Qwen3-ASR의 LID 태그가 중국 방언을 별도 라벨로 출력하는지(예: "language Sichuanese")는 보고서에서 확인하지 못했다.
- 한국어 joint ASR + 방언 식별 논문은 찾지 못했다.

---

## 3. 어느 인코더 층에 억양 정보가 있나? 오디오 인코더를 fine-tune하는 것과 LLM만 학습하는 것은 차이가 큰가?

### Takeaway
Whisper-large-v3 계열 32층 인코더(Qwen2-Audio 인코더)에서 억양 민감도는 0~14층에서 낮다. 15층 부근에서 오르기 시작해 21층 이후 최대가 된다. 조정(steering)이 가장 잘 먹히는 구간은 15~19층이다. 우리 probe가 쓴 14층은 이 구간의 바로 앞이다. ASR 인코더를 multi-task로 쓸 때도 12층 중 9층까지 공유한 지점이 최적이었다. 즉 중상위층이 유리하다. LLM에 LoRA만 거는 방식은 음향 과제에서 효과가 제한적이라는 증거가 있다. 인코더 중간층 정보를 LLM에 직접 주거나 인코더 상위층을 학습하는 쪽이 근거에 더 맞는다.

### Cited Findings
- **Qwen2-Audio-7B 인코더**(Whisper식 32층), VCTK 억양 5종과 L2-ARCTIC 3종으로 층별 accent sensitivity를 쟀다.
  - VCTK: "low sensitivity in early layers (0–14), an emerging peak near layer 15, and a sharp rise to global maximums from layer 21 onward".
  - L2-ARCTIC(비원어민): 초기층에서도 피크가 있다. 저수준 음향 일탈이 크기 때문이다.
  - mean-shift steering: 중간층(15~19)에서 WER 감소가 가장 컸다. VCTK 일부 억양에서 최대 약 30% ΔWER, L2-ARCTIC에서 약 5% ΔWER이었다. 마지막 층(31)에 steering하면 WER이 약 4.0 올라 심하게 악화됐다 — [Activation Steering](https://arxiv.org/pdf/2603.05813)
- **Zhang 외 2021**(ASR + AR joint): 12층 인코더 중 공유 층에 따른 테스트 ACC는 3층 54.6, 6층 66.0, **9층 71.4**, 12층 70.8이었다. 공유 층 수가 바뀌어도 WER은 거의 그대로였다 — [Zhang 외 2021](https://arxiv.org/pdf/2106.08211)
- **Prasad & Jyothi, "How Accents Confound"** (ACL 2020, DeepSpeech2). 영어 억양 7종에서 gradient, 정보량, 분류기 세 가지 probe 모두 첫 번째 recurrent 층에 억양 정보가 가장 많았다 — [ACL 2020](https://aclanthology.org/2020.acl-main.345)
- Wav2Vec2-base 후속 연구(arXiv 2603.03359)는 억양 정보가 이른 층(3층)에 몰려 있다고 보고했다. 그 부분공간을 제거해도 억양 간 성능 격차는 줄지 않았다(미검증: 초록만 확인) — [arXiv 2603.03359](https://arxiv.org/abs/2603.03359v1)
- 초분절(성조·강세·pitch accent) 층별 probe(Interspeech 2024): wav2vec 2.0은 중간층에서 피크가 났고, HuBERT/WavLM은 마지막 층까지 유지됐다. 구 단위 pitch accent는 어휘 강세나 성조보다 약하게 부호화됐다 — [Mandarin/English suprasegmentals](https://arxiv.org/html/2408.13678)
- HuBERT 이산 토큰에서는 억양 복원력이 중초기층에서 피크가 났다(미검증: 검색 요약) — [arXiv 2601.19786](https://arxiv.org/pdf/2601.19786)
- **Vox-Profile**은 conv층과 모든 Transformer 층의 가중 평균(weighted sum)을 1D conv와 시간 평균에 넣고 FC로 분류한다(Pepino 외 구조). 이 설정에서 Whisper-Large narrow F1이 0.724였다. 억양 과제에서 인코더를 동결했는지는 명시되지 않았다(다른 속성은 동결) — [Vox-Profile](https://arxiv.org/html/2505.14648v1)
- **PRiSM**: Whisper-small 최종층 RP가 L1-arc F1 84.3으로 WavLM-base RP 58.3과 음소 인식 특화 모델들보다 높았다 — [PRiSM](https://arxiv.org/html/2601.14046v2)
- **ASR 사전학습·초기화 효과**: AESRC에서 ASR 초기화가 AR을 "significantly improved"했다([AESRC](https://ar5iv.labs.arxiv.org/html/2102.10233)). Zhang 외에서는 32.6에서 69.8로 올랐다([Zhang 외](https://arxiv.org/pdf/2106.08211)). 한국어 XLS-R에서는 음소 인식 fine-tune을 먼저 한 효과가 미미했다(F1 0.865 → 0.868)([Lee 외 2025](https://www.eksss.org/archive/view_article?pid=pss-17-3-83)).
- **LLM만 학습할 때의 한계**: "Where Does the Sound Go?"에서 LM LoRA는 "limited, encoder-dependent gains"였다. Whisper-Tiny에서는 captioning CIDEr가 0.685에서 0.476으로 오히려 떨어졌다. 출력 행(readout)만 학습해도 30~42pp가 회복됐다 — [Where Does the Sound Go?](https://arxiv.org/html/2609.05871v1)
- **인코더 해동의 위험**: Granite-speech로 화자 귀속 ASR을 한 연구는 인코더를 동결하고 LLM에 LoRA를 걸었다. 원래 인코더를 해동하면 ASR이 악화돼서, 대신 중간 인코더 층 출력을 추가로 주는 방식을 썼다고 한다(미검증: 검색 요약에만 나오고 초록에서는 확인 안 됨) — [arXiv 2604.11269](https://arxiv.org/pdf/2604.11269)
- **인코더가 없는(encoder-free) 음성 LLM**: Mel 특징을 가벼운 embedding으로 LLM에 바로 넣는 구조다. 화자·감정 같은 메타데이터로 감독하는 사전학습(MSP)을 따로 해야 화자 구분과 paralinguistic 능력이 생긴다고 보고했다(정량 수치는 초록에 없음) — [Teaching LLMs to Hear Who Spoke What, arXiv 2610.01695](https://arxiv.org/abs/2610.01695)

### Inferences
- 우리 linear probe(Whisper-large-v3 14층 0.44)가 약했던 이유 중 하나는 층 선택일 수 있다. Qwen2-Audio 인코더는 Whisper-large-v3에서 출발한 32층 구조로 알려져 있고, 같은 구조에서 억양 민감도는 15층 이후, 특히 21층 이후 높다. 다음 실험은 두 가지다. (1) 18~28층 단일층 probe 스윕. (2) Vox-Profile처럼 전 층 가중합. 현재 쓰는 0.81 분류기가 어느 층을 쓰는지도 같은 기준으로 점검할 만하다.
- Gemma 4 실패(인코더 없음 + LLM 층 LoRA만)는 문헌과 맞는다. 인코더 없는 구조는 paralinguistic 능력을 따로 사전학습해야 하고, LM LoRA만으로는 음향 과제 이득이 제한적이다. LALM 후보(Qwen3-Omni, Qwen2.5-Omni, Qwen3-ASR)에서는 다음 순서로 시도하는 것이 근거에 맞다. (a) 오디오 인코더 상위 절반(예: 16~31층)에 LoRA를 걸거나 해동한다. (b) 인코더 중간층(예: 20층 부근) 출력을 pooling해 보조 억양 head를 학습한다. (c) 라벨 토큰의 출력 행과 head를 학습한다. 인코더 전체를 해동하면 ASR이 나빠질 위험이 있으니 ASR CER을 함께 감시해야 한다.
- ASR 초기화가 억양 정확도를 크게 올린다는 결과는 ASR이 이미 강한 Qwen3-ASR에서 시작하는 선택을 지지한다. 단, Qwen3-Omni와 Qwen3-ASR의 AuT 인코더에서 억양이 어느 층에 있는지는 측정된 바가 없어 직접 probe해야 한다.

### Gaps
- Whisper-large-v3 자체(또는 AuT 인코더)에서 억양 분류 정확도를 층별로 잰 표는 찾지 못했다. steering 논문은 민감도 그래프만 있고 수치표가 없다.
- 오디오 인코더 fine-tune과 LLM-only LoRA를 억양 과제에서 직접 비교한 ablation은 찾지 못했다. 위 결론은 감정·소리 과제와 ASR 연구에서 끌어온 간접 증거다.
- 한국어 방언에 대한 층별 분석은 없다.

---

## 4. 한국어 지역 방언 식별 연구(AI Hub 방언, KsponSpeech 등)와 수치, 표준어 대 방언 판별

### Takeaway
한국어 방언 식별 최고 보고치는 XLS-R-300M fine-tune의 6개 방언권 F1 0.868이다(Lee·Kim·Chung, 말소리와 음성과학 2025). 다만 데이터는 50세 이상 남성 방언 화자의 첫 5초이고, 표준어 클래스가 없으며 어휘 통제도 없다. 이전 전통 ML·RNN 연구는 대개 F1 0.6 안팎이었다. LALM을 쓴 한국어 방언 식별, 한국어 ASR + 방언 식별 joint 모델, "방언 어휘 없는" 조건을 통제한 연구는 찾지 못했다.

### Cited Findings
- **이주영·김선희·정민화, 「wav2vec 2.0 XLS-R을 활용한 한국어 방언 자동 식별」** (Automatic Korean dialect identification using wav2vec 2.0 XLS-R), 말소리와 음성과학(Phonetics and Speech Sciences) 17권 3호, pp.83–91, 2025, DOI 10.13064/KSSS.2025.17.3.083.
  - 데이터: AI Hub 「중·노년층 한국어 방언 데이터」(강원·경상, 충청·전라·제주). 남성 50세 이상만 썼다(NORM 기준). 학습 3,126명 324,957개, 테스트 180명 18,000개(방언당 30명). 화자가 겹치지 않게 다시 나눴다.
  - 6개 방언권(이기갑 2005 3차 구획): 중부, 경남, 경북, 전남, 전북, 제주.
  - 모델: wav2vec2-xls-r-300m. 한국어 낭독 음소 인식(CTC, PER 3.88%)으로 먼저 fine-tune한 뒤 6-way 분류 fine-tune을 했다. 입력은 첫 5초다.
  - 결과(F1): XLS-R(음소 FT 포함) **0.868**, XLS-R(음소 FT 없음) 0.865, eGeMAPS LLD + Bi-LSTM 0.738, 정규화 MFCC + attention Bi-LSTM 0.647, 평균 MFCC + DNN 0.438.
  - 클래스 가중치가 없으면 F1이 **0.45~0.55**로 떨어지고 중부로 쏠렸다.
  - 클래스별 재현율(Bi-LSTM → XLS-R): 중부 0.70→0.91, 경남 0.69→0.74, 경북 0.66→0.88, 전남 0.67→0.89, 전북 0.86→0.86, 제주 0.85→0.93.
  - 혼동: 모든 방언이 중부와 자주 혼동됐다. 중부와 전남은 억양이 비슷해 서로 헷갈렸다.
  - 기존 한국어 6개 지역 시스템은 정확도·F1이 "약 60%" 수준이었다고 정리했다.
  - 어휘 통제나 표준어 클래스는 없다 — [Phonetics Speech Sci. 2025](https://www.eksss.org/archive/view_article?pid=pss-17-3-83)
- **김영국·김명호, 「음향 특성에 따른 한국어 방언 분류 모델의 성능 비교」**, 한국컴퓨터정보학회논문지 26권 10호, pp.37–43, 2021. AI Hub 「한국어 대화 음성」을 쓰고 **경기/서울, 강원, 충청, 전라, 경상** 5클래스로 나눠 서울을 한 클래스로 포함했다. ML·DL 모델 8종을 비교했고, MFCC를 정규화하면 최고 모델 정확도가 +1.07%p, F1이 +2.04%p 올랐다. 모델별 절대 수치는 이번에 확인하지 못했다 — [KCI JKSCI 2021](https://journal.kci.go.kr/jksci/archive/articleView?artiId=ART002768401)
- **Lee, Kim, Chung (2022), "Korean Dialect Identification Based on an Ensemble of Prosodic and Segmental Feature Learning for Forensic Speaker Profiling"**, O-COCOSDA 2022(하노이). 운율 + 분절음 특징 앙상블(attention BiLSTM + DNN)로 F-measure **61.28%**를 얻어 기준선 대비 약 26pt 높았다(미검증: 검색 요약에만 있음) — [IEEE Xplore 9997971](https://ieeexplore.ieee.org/document/9997971/). 서지 정보는 [KoreaScience 인용](https://www.koreascience.or.kr/article/JAKO202322149985872.page?&lang=ko)에서 확인했다. 같은 그룹의 2021년 억양(intonation) 모델링 연구도 O-COCOSDA 2021에 있다(같은 출처).
- **말 속도·휴지 기반 방언 분류**(말소리와 음성과학 15권 2호): 경상도 대 충청도 2지역만 분류했다 — [Phonetics Speech Sci. 15(2)](https://www.eksss.org/archive/view_article?pid=pss-15-2-43). 이주영 외 2025 논문은 이 연구를 Na & Lee(2023, AI Hub 노년층 자유대화, SVM/RF/LightGBM)로 인용한다.
- 이주영 외 2025가 인용한 기타 연구도 있다. Lee 외(2019, MFCC + F0, 성별·연령·방언), Lee 외(2021, 억양 특징 attention Bi-LSTM), Lee 외(2022, AI Hub 자유대화 일반, 운율 + 분절음 융합) — [Phonetics Speech Sci. 2025](https://www.eksss.org/archive/view_article?pid=pss-17-3-83)

### Inferences
- 우리 결과(4개 지역 macro-F1 0.81, 표준 대 지역 AUC 0.885, 방언 어휘 없는 클립)는 문헌 최고치(6개 방언권 F1 0.868)보다 숫자는 낮다. 하지만 조건이 더 어렵다. 어휘 통제, 표준어 포함, 화자 연령·성별 제한 없음이 그 차이다. 문헌의 어떤 결과도 "방언 단어 없는 클립"으로 평가하지 않았기 때문에, 우리 수치가 한국어에서 사실상 첫 어휘 통제 결과일 가능성이 있다(확인된 범위 안에서).
- 중부(서울·경기 포함)가 모든 방언의 흡수점이 되는 혼동 패턴과 클래스 불균형 민감도(가중치 없으면 F1 0.45~0.55)는 LALM SFT에서도 그대로 나타날 것이다. 지역 태그 토큰의 loss 가중이나 균형 샘플링, 그리고 5번 질문의 사전 보정이 필요하다.
- 문헌의 학습 입력은 첫 5초였다. 짧은 구간으로도 0.87이 나오므로 LALM 학습 클립 길이를 줄이는 것도 가능해 보인다(추론).

### Gaps
- 한국어 "표준어 대 방언" 이진 판별의 AUC나 정확도를 보고한 논문은 찾지 못했다(김영국·김명호 2021은 서울을 한 클래스로 넣었지만 수치를 확보하지 못함).
- KsponSpeech로 방언 식별을 한 논문, LALM이나 Whisper로 한국어 방언을 식별한 논문, 한국어 ASR + 방언 식별 joint 논문은 찾지 못했다.
- KCI/학위논문(석사) 중 더 최근 연구가 있을 수 있으나 이번 검색에서는 나오지 않았다.

---

## 5. 생성형 LALM에서 보정된 클래스 확률을 얻는 방법(라벨 토큰 logit, 언어화된 confidence, 보정 기법)

### Takeaway
분류 확률은 문자열을 생성시켜 파싱하지 말고, 태그 위치에서 K개 지역 라벨의 첫 토큰 logit만 모아 softmax하는 방식이 표준이다(Whisper의 언어 토큰 LID와 같은 원리). 생 softmax는 보정이 나쁘고 라벨 사전 편향이 크다. 그래서 dev 셋 temperature scaling과 batch/contextual calibration(표준어 쪽 편향 제거)을 같이 쓰는 것이 안전하다. 언어화된 confidence(예: "확신도 80%")는 텍스트 LLM에서 토큰 확률보다 보정이 좋다는 보고가 있다. 하지만 SFT한 분류 모델이나 연속 점수(억양 강도)에는 맞지 않다. 오디오 LLM에서는 semantic entropy와 P(True)가 토큰 수준 방법보다 낫다는 2026년 결과가 있지만 억양 과제 자료는 없다.

### Cited Findings
- **라벨 토큰 logit → softmax**: Odyssey 2024 감정 인식 챌린지의 Qwen-Audio식 시스템은 감정 토큰 8개의 logit만 뽑아 softmax해 클래스 확률을 냈다 — [Odyssey 2024 SER system](https://oa.upm.es/81516/1/Odyssey2024_challenge_v2.pdf)
- 자유 생성은 유효 라벨에 대응한다는 보장이 없다. 시퀀스 log-prob은 근사 confidence일 뿐이다(41개 open-weight LLM 제로샷 의도 분류 평가) — [arXiv 2607.27421](https://arxiv.org/pdf/2607.27421)
- **Whisper LID**: 첫 디코딩 위치에서 언어 토큰들의 확률로 언어를 고른다(이번 세션에서 원문 미확인. Whisper 논문의 일반 사실) — [Whisper, arXiv 2212.04356](https://arxiv.org/abs/2212.04356)
- **Contextual calibration** ("Calibrate Before Use", Zhao 외, ICML 2021): "N/A" 같은 내용 없는 입력으로 라벨 사전 편향을 추정하고 나눠서 제거한다(이번 세션에서 원문 미확인) — [arXiv 2102.09690](https://arxiv.org/abs/2102.09690)
- **Batch Calibration** (Zhou 외, Google/Cambridge, ICLR 2024): 내용 없는 토큰으로 편향을 추정하는 방식의 결함을 지적했다. 대신 추론 배치 자체의 평균 라벨 확률로 편향을 추정해 제거한다. "zero-shot, inference-only, and incurs negligible additional costs"이고, 소량 라벨 데이터로 편향을 학습하는 few-shot 변형도 있다. PaLM 2와 CLIP으로 10개 이상 과제에서 기존 보정법보다 좋았다 — [Batch Calibration, arXiv 2309.17249](https://arxiv.org/pdf/2309.17249v3)
- **언어화된 confidence**: Tian 외(2023)는 RLHF 모델에서 출력 토큰으로 말한 confidence가 조건부 토큰 확률보다 보정이 좋은 경우가 많다고 보고했다(이번 세션에서는 2차 인용으로 확인) — [Tian 외 2023, arXiv 2305.14975](https://arxiv.org/abs/2305.14975). 범주형 라벨에 대한 언어화된 확률 분포에는 "re-softmax" 문제가 있어, 되돌린 언어화 확률에 temperature scaling을 하자는 제안이 있다 — [Calibrating Verbalized Probabilities for LLMs](https://www.researchgate.net/publication/384771081_Calibrating_Verbalized_Probabilities_for_Large_Language_Models). ConfTuner는 tokenized Brier score로 언어화된 confidence를 직접 fine-tune한다 — [ConfTuner, arXiv 2508.18847](https://arxiv.org/pdf/2508.18847)
- **오디오 LLM 불확실성** (Kuan, Huang, Hung-yi Lee, "Walking Through Uncertainty", arXiv 2604.25591, 2026, 작성 중 원고): predictive entropy, length-normalized entropy, semantic entropy, discrete semantic entropy, P(True)를 MMAU/MMAR/MMSU/SAKURA에서 비교했다. 일반 오디오 추론에서는 semantic 수준과 검증 기반 방법(semantic entropy, P(True))이 토큰 수준 기준선보다 "consistently outperform"이었다. 환각·답 없는 질문 같은 신뢰성 벤치마크에서는 최적 방법이 모델과 벤치마크에 따라 달랐다 — [arXiv 2604.25591](https://arxiv.org/abs/2604.25591). SAKURA 감정 범주 ECE가 0.227(7B), 0.181(3B)로 다른 범주보다 높다는 수치도 있다(미검증: 검색 요약, 모델명 미확인).
- **readout 행 학습**: 선택지 토큰 출력 행만 fine-tune해도 정확도가 30~42pp 회복됐다(Where Does the Sound Go?). 라벨 토큰 logit을 확률로 쓸 경우 그 행을 학습 대상으로 삼을 근거가 된다 — [arXiv 2609.05871](https://arxiv.org/html/2609.05871v1)
- **억양 강도를 연속 점수로 쓰는 선례**:
  - Vox-Profile narrow 억양 분류기(16클래스 posterior)를 억양 강도 지표로 쓴 L2 억양 음성 생성 연구가 있다(미검증: 검색 요약) — [arXiv 2603.07550](https://arxiv.org/pdf/2603.07550)
  - Idiap의 음소 확률 시퀀스 비교 기반 accentedness는 ISLE 코퍼스에서 사람 평가와 상관 0.68을 보였다(미검증: 검색 요약) — [Idiap publication 3140](https://publications.idiap.ch/publications/show/3140)
  - 자기지도 표현 거리가 미국 영어 청자의 억양 강도 판단을 잘 예측한다는 probe 연구 — [arXiv 2506.17542](https://arxiv.org/html/2506.17542v1)

### Inferences
- **권장 확률 산출 절차**(근거 결합 추론):
  1. 지역 라벨을 서로 다른 첫 토큰을 가진 단일 토큰이나 짧은 고정 문자열로 설계한다. 예: `지역: 서울경기|강원|충청|경상|제주|전라`. 서브워드가 겹치면 첫 토큰 확률이 왜곡된다.
  2. 태그 위치에서 K개 라벨 첫 토큰의 logit만 모아 softmax한다. 이것을 P(서울경기)로 쓴다.
  3. 서울 화자 dev 셋에서 temperature T를 맞춘다. 서울 화자 15명 중 14명을 지역으로 판정한 것과 같은 사전 편향은 batch calibration으로 빼는 방식을 시도한다. 배치 평균 확률이나 표준어 dev 셋 평균으로 나눈 뒤 재정규화한다.
  4. 현재 쓰는 0.81 분류기의 posterior를 SFT 때 지역 태그 위치의 soft target(KL 증류)으로 쓰면, 생성 태그의 확률이 이미 보정된 분류기 분포를 따라가도록 학습할 수 있다.
- 언어화된 confidence("80%")는 출력 하나가 0~100 사이의 거친 값이고 SFT 데이터에 정답 confidence가 없다. 그래서 연속 억양 강도 지표로는 토큰 logit 방식보다 불리하다. P(True)나 semantic entropy는 여러 번 샘플링해야 하므로 비용이 커서 실시간 코칭 앱에는 맞지 않는다.
- 보정 품질은 ECE와 Brier, 그리고 표준 대 지역 AUC(현재 0.885)를 같이 보고해야 기존 분류기와 공정하게 비교할 수 있다.

### Gaps
- 오디오 LLM에서 억양·방언 라벨 확률의 보정(ECE)을 잰 논문은 찾지 못했다.
- 라벨 토큰 logit, 언어화된 confidence, 보조 head 확률을 같은 오디오 분류 과제에서 직접 비교한 연구도 찾지 못했다.
- Calibrate Before Use와 Tian 외 2023은 이번 세션에서 원문을 열지 않았다(잘 알려진 논문이라 URL과 요지만 기재).
- "Walking Through Uncertainty"의 모델별 ECE 표는 확인하지 못했다.
