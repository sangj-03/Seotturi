# 공개 LALM 미세조정 레시피, 다중 과제 설계, 증류, 영상(유튜브) 기반 방언 학습 데이터 구축

조사일: 2026-10-14. 대상: 하나의 LALM이 1~20초 음성에서 [억양 5종 분류 + 방언 그대로 전사 + 표준어 변환]을 함께 내도록 미세조정하는 방법, 그리고 유튜브 같은 영상에서 방언 학습 데이터를 만드는 방법.
표기: "(검색 요약)"은 원문을 직접 열지 못하고 검색 결과 요약으로만 확인한 내용이다. "(미검증)"은 출처끼리 어긋나거나 확인하지 못한 내용이다.

---

## 1. Qwen2-Audio / Qwen2.5-Omni / Qwen3-Omni / Qwen3-ASR 미세조정 레시피와 결과 (학습 모듈, LoRA rank, 데이터 양, 메모리, 30B MoE의 QLoRA 가능성, 일반 능력 망각)

### Takeaway
공개 레시피는 대부분 "오디오 인코더는 고정하고 LLM에 LoRA(r=16~64, α=32~128)를 붙이는" 방식이다. 이 방식으로 지역 억양·저자원 언어에서 수천~1만 발화만으로 WER을 40~60% 줄인 사례가 있다(Qwen2.5-Omni-3B, Qwen2-Audio QLoRA, Qwen3-ASR). Qwen3-Omni-30B-A3B는 공식 가이드가 8×H100 전체 미세조정뿐이다. 다만 단일 GPU에서 QLoRA(NF4, r=16)로 학습한 논문이 있고, 같은 GB10 칩에서 30B-A3B MoE를 4-bit QLoRA로 약 35 GB에 학습한 보고도 있다. 그래서 DGX Spark에서는 QLoRA만 현실적이다(bf16 로드는 전문가 병합 단계에서 OOM 위험).

### Cited Findings

**Qwen3-ASR (1.7B / 0.6B)**
- 구조: Qwen3-ASR-1.7B는 Qwen3-1.7B + projector + AuT 인코더(300M, hidden 1024)이고, 0.6B는 Qwen3-0.6B + AuT 인코더(180M, hidden 896)이다. AuT는 Fbank를 8배 다운샘플해 12.5 Hz 토큰을 낸다. — [Qwen3-ASR Technical Report, arXiv 2601.21337](https://arxiv.org/abs/2601.21337)
- 학습 4단계: (1) AuT 사전학습(중국어·영어 위주 의사 라벨 ASR 약 4,000만 시간) (2) Omni 사전학습(3조 토큰) (3) ASR SFT (4) GSPO 강화학습(약 5만 발화: 중·영 35%, 다국어 35%, 기능성 30%). 강화학습은 "잡음 강건성, 전사 안정성"에 핵심이었다고 한다. — [arXiv 2601.21337](https://arxiv.org/abs/2601.21337)
- **중요**: SFT 단계에서 일부러 "자연어 지시를 따르지 않는 ASR 전용 모델"로 만들었다(지시 주입 방지 목적). 출력 형식은 `language Korean<asr_text>…` 또는 `language None<asr_text>`이다. 시스템 프롬프트의 context 토큰은 배경 지식(용어 편향)으로만 쓴다. — [arXiv 2601.21337](https://arxiv.org/abs/2601.21337)
- 한국어 성능(1.7B / 0.6B, 오류율 %): CommonVoice ko 5.88 / 8.48, Fleurs ko 2.57 / 3.72, MLC-SLM ko 8.61 / 10.31. 한국어는 ASR 30개 언어와 ForcedAligner 11개 언어(한국어 AAS 37.2 ms)에 들어간다. 중국 방언 22종은 지원하지만 한국 방언은 언급이 없다. — [arXiv 2601.21337 부록 표](https://arxiv.org/abs/2601.21337); [QwenLM/Qwen3-ASR README](https://github.com/QwenLM/Qwen3-ASR)
- 공식 미세조정: `finetuning/qwen3_asr_sft.py`. JSONL 형식은 `{"audio": wav, "text": "language Korean<asr_text>…"}`이다. 예시 하이퍼파라미터는 lr 2e-5, batch 32, grad_acc 4, epoch 1이다. 문서에 LoRA, 고정 모듈, GPU 메모리 설명은 없어서 전체 파라미터 SFT로 보인다(명시는 안 됨). — [Qwen3-ASR finetuning README](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning)
- 커뮤니티 한국어 미세조정 TeamUNIVA/qwen3_asr_1.7b_ko_beta(데이터·방법 비공개): 한국어 합계 CER 0.0686 → 0.0537, KsponSpeech 0.0865 → 0.0670, clova_call 0.0448 → 0.0293. 영어 LibriSpeech WER 0.0250 → 0.0246으로 영어 망각은 없었다. — [HF 모델 카드](https://huggingface.co/TeamUNIVA/qwen3_asr_1.7b_ko_beta)
- MLC-SLM 2026 대회 시스템(Qwen3-ASR-1.7B): (a) 전체 SFT 4 epoch, lr 2e-5 (b) 텍스트 디코더 q/k/v/o 28층 전체(112개 모듈)에 LoRA r=64, α=128, dropout 0.05를 붙이고 **오디오 projector(proj1/proj2)만 추가로 학습**, 5 epoch, lr 5e-5 (c) GRPO lr 1e-6. 한국어 tcpCER은 기본 27.17 → 25.14(Whisper-large-v3 31.58)였다. 전체 평균은 30.53 → 24.3(전체 SFT) → 23.78(+합성 음성 LoRA)이었다. — [Diarization-Guided Qwen-ASR Adaptation, arXiv 2607.08208](https://arxiv.org/abs/2607.08208)

**Qwen3-Omni-30B-A3B**
- 변형: Instruct(thinker+talker), Thinking(thinker만, CoT), Captioner(Instruct를 오디오 캡션용으로 미세조정; 방법은 비공개). 음성 입력 언어에 한국어가 들어 있다. README는 "19개 언어"라고 하지만 목록에는 18개만 있다(미검증 불일치). — [QwenLM/Qwen3-Omni README](https://github.com/QwenLM/Qwen3-Omni); [Qwen3-Omni Technical Report, arXiv 2509.17765](https://arxiv.org/abs/2509.17765)
- 추론 메모리 하한(bf16, FlashAttention2, 15초 영상 기준): Instruct 78.85 GB, Thinking 68.74 GB. `disable_talker()`를 쓰면 약 10 GB를 아낀다. vLLM serve는 thinker만 지원한다. 저장소에는 미세조정 안내가 없다. — [Qwen3-Omni README](https://github.com/QwenLM/Qwen3-Omni)
- ASR(WER): Fleurs 19개 언어 평균은 Qwen3-Omni-30B-A3B 5.33, Qwen2.5-Omni 14.04다. Librispeech clean/other는 1.22/2.48이다. 기술 보고서는 단일 모달 모델 대비 "성능 저하가 없다(without any degradation)"고 주장한다. — [README](https://github.com/QwenLM/Qwen3-Omni); [arXiv 2509.17765](https://arxiv.org/abs/2509.17765)
- NVIDIA NeMo AutoModel 공식 레시피는 **전체 미세조정**이다. 비전 타워는 고정하고 오디오 타워와 LLM은 학습한다(`freeze_audio_tower: false`). 8×H100 80GB, FSDP2, ep_size=8, GPU당 피크 약 36~40 GB, lr 2e-5, global batch 64, 1 epoch(AMI 약 6.9만 클립, 약 22분)로 돌렸다. AMI IHM WER 15.81% → 8.31%였다. PEFT 구성은 "의도적으로 생략"했고 thinker만 학습한다(talker/code2wav는 그대로 복사). 같은 문서에서 Qwen2.5-Omni-3B의 매크로 WER은 8.97% → 6.55%였다. — [NeMo AutoModel: Fine-Tune Qwen3-Omni for ASR](https://docs.nvidia.com/nemo/automodel/latest/recipes-e2e-examples/qwen3-omni-asr)
- **단일 GPU QLoRA 사례**: Qwen3-Omni-30B-A3B-Thinking을 4-bit NF4(bf16 compute)로 올렸다. LoRA r=16, α=32를 LM projection에 붙이고 **오디오 타워는 bf16으로 고정**했다. "All runs use a single GPU"이고 3 epoch, 단계별 lr은 2e-4 / 5e-5다. 영역은 오디오 이벤트 시간 지정(ASR 아님)이다. — [Auto-AEG, arXiv 2607.04383](https://arxiv.org/abs/2607.04383)
- 같은 GB10(sm_121, aarch64, 약 122 GB 통합 메모리)에서 Nemotron-3-Nano-30B-A3B(MoE)를 다룬 보고: **bf16 로드 단계에서 전문가 병합(torch.stack) 때문에 116 GiB에서 OOM**이 났고 장비가 멈췄다. 4-bit QLoRA(`device_map: 0`, `low_cpu_mem_usage`)로 바꾸자 로드 피크 약 20 GB, 학습 약 35 GB로 128개 전문가 전체 LoRA를 얻었다. CUDA 13, NGC PyTorch 2.10, transformers 5.5.0 환경이다. 실패한 뒤에도 메모리가 남으므로 재부팅을 권한다. 다른 모델 사례이고 일화성 보고다. — [NVIDIA 개발자 포럼](https://forums.developer.nvidia.com/t/fine-tuning-nemotron-3-nano-30b-a3b-on-asus-ascent-gx10-gb10-dgx-spark-cuda-oom-at-load-box-freezes-4-bit-qlora-fix/372434)
- bitsandbytes CUDA 백엔드는 Linux aarch64를 지원한다고 명시돼 있다. GB10/sm_121 공식 호환 성명은 찾지 못했다(검색 요약). — [HF Transformers bitsandbytes 문서](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- LLaMA-Factory는 `qwen3_omni`(30B), `qwen2_omni`(3B/7B), `qwen2_audio`(7B), `gemma3n` 템플릿을 지원한다. 일반 추정 메모리(MoE 별도 표 없음)는 30B 기준 LoRA 16-bit 64 GB, QLoRA 8-bit 40 GB, QLoRA 4-bit 24 GB다. 7B는 각각 16 / 10 / 6 GB다. — [hiyouga/LlamaFactory README](https://github.com/hiyouga/LlamaFactory)
- ms-swift는 Qwen3-Omni를 지원 목록에 올려 두었다. "vit/aligner/llm 독립 제어", BNB/AWQ/GPTQ 양자화 모델 학습(QLoRA), Megatron-SWIFT의 MoE LoRA(EP 병렬)를 지원한다. Qwen3-Omni 오디오 전용 메모리 수치는 공개돼 있지 않다. — [modelscope/ms-swift README](https://github.com/modelscope/ms-swift)
- ms-swift에서 `q_proj|k_proj` 같은 정규식 target은 오디오 인코더 층까지 잡는다. 따라서 `--target_regex`로 모듈 경로를 명시해야 한다는 FAQ·커뮤니티 안내가 있다(검색 요약). Qwen3-Omni의 audio-in-video SFT/GRPO가 vLLM 0.12 이상에서 크게 느려진다는 이슈도 열려 있다. — [ms-swift FAQ](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Frequently-asked-questions.md); [ms-swift issue #7108](https://github.com/modelscope/ms-swift/issues/7108)

**Qwen2.5-Omni (3B/7B)**
- **억양별 LoRA 사례(가장 유사)**: Qwen2.5-Omni-3B LLM의 attention projection에 LoRA r=32, α=32를 붙이고 나머지는 모두 고정했다. 지역(인도·인도네시아·중남미 영어)마다 9k 발화로 2 epoch 학습했다. AdamW-8bit, lr 5e-5, batch 4, max len 2048, Unsloth를 썼고 A100-40GB 1장에서 adapter 하나에 약 6시간 걸렸다. vLLM으로 서빙해 P95 약 800 ms였다. **한 번의 forward로 verbatim 전사와 교정 전사를 함께 출력(dual-output)**한다. 미세조정 전 대비 WER이 40~46% 줄었고(예: India 11.02 → 5.95), Qwen3-Omni-30B zero-shot(7.13)보다 좋았다. 정답 전사는 Gemini 2.5 Pro로 만들고 750개를 사람이 검증했다. — [Beyond WER, arXiv 2609.20828](https://arxiv.org/abs/2609.20828)
- LLaMA-Factory로 Qwen2.5-Omni-7B에 LoRA를 붙이고 오디오 타워를 고정한 연구 사례가 있다(검색 요약). ms-swift로 Qwen2.5-Omni의 오디오 인코더 + projector + LLM 모두에 LoRA r=8, α=32를 붙인 오디오 QA 사례(lr 1e-4, batch 1, grad_acc 8)도 있다(검색 요약). — [arXiv 2607.25355](https://arxiv.org/pdf/2607.25355); [arXiv 2608.23092](https://arxiv.org/pdf/2608.23092)

**Qwen2-Audio-7B**
- CantoASR: Qwen2-Audio-7B-Instruct를 4-bit NF4(bitsandbytes)로 올리고 attention과 MLP에 LoRA r=16, α=32, dropout 0.05(학습 파라미터 약 0.6~0.8%)를 붙였다. 8-bit paged AdamW, lr 1e-4, warmup 200, 유효 batch 8, 3 epoch로 학습했다. zero-shot 광동어 CER은 Qwen2-Audio 14.5로 SALMONN-7B 36.7, DeSTA2.5 28.5, Phi-4-MM 30.8보다 크게 낮았다. — [CantoASR, arXiv 2511.04139](https://arxiv.org/abs/2511.04139)
- 다국어 적응(MLC-SLM)에서 "망각을 피하려고" q/k/v에만 LoRA r=16, α=32를 쓴 사례가 있다(검색 요약). — [ILT, arXiv 2507.08477](https://arxiv.org/abs/2507.08477)
- LoASR-Bench(CommonVoice 16.1, 25개 저자원 언어, HF Transformers, 10 epoch, batch 8, lr 5e-5, fp16)의 한국어(학습 2시간) WER: Whisper-L 0.17, Qwen2-Audio 기본 0.34, **Qwen2-Audio 미세조정 0.12**, Qwen3-Omni 0.09. Qwen3-Omni 열이 미세조정 결과인지는 표에서 분명하지 않다. — [LoASR-Bench, arXiv 2603.20042](https://arxiv.org/abs/2603.20042)

**vLLM LoRA 서빙**
- 2026-10-14에 확인한 vLLM latest 문서의 LoRA 열: Qwen2_5OmniThinker ✅, Qwen3ASR ✅, Gemma4 ✅, Qwen2Audio ✗, Whisper ✗. Qwen3OmniMoeThinker는 **생성 표에서 ✅**, 전사 표에서는 ✗다. 사용자가 적어 둔 "0.26은 Qwen3-Omni LoRA 미지원"과 다르다. 버전 차이로 보이며 미검증이다. — [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

**망각(일반 능력 유지)**
- VoiceBench 종합: Qwen2.5-7B-Instruct(텍스트) 82.81 → Qwen2.5-Omni 텍스트 입력 78.60 → 음성 입력 75.08. Qwen2-Audio는 텍스트 64.54 → 음성 55.35다. 단순 CE로 미세조정하면 음성 입력 점수가 71.25로 **떨어졌다**. 텍스트 LLM 교사로 KD(λ=0.5, τ=2, 약 6만 샘플, 2 epoch, lr 5e-6)를 하자 77.19로 올랐다. — [Cross-Modal Knowledge Distillation for Speech LLMs, arXiv 2509.14930](https://arxiv.org/abs/2509.14930)
- DeSTA2.5-Audio: 기존 LALM이 원래 LLM 능력을 잃는 문제를 백본 LLM이 학습 목표를 스스로 생성하게 해서 줄였다고 주장한다(약 500만 샘플, 50개 데이터셋, TASLP 2026). — [arXiv 2507.02768](https://arxiv.org/abs/2507.02768)

### Inferences
- DGX Spark(128 GB)에서 Qwen3-Omni-30B-A3B를 bf16 LoRA로 학습하는 것은 이론상 가능 범위(LLaMA-Factory 추정 64 GB)다. 그러나 MoE 전문가 병합 시 순간 메모리가 약 2배로 뛴 GB10 사례를 보면 4-bit QLoRA + thinker만 + 오디오 타워 고정이 가장 안전하다. 학습한 LoRA를 vLLM에서 쓰려면 설치 버전의 Qwen3-Omni LoRA 지원을 먼저 확인해야 한다. 지원하지 않으면 bf16 base에 병합한 뒤 서빙해야 한다(NF4로 학습한 adapter를 bf16에 병합할 때의 손실은 이번에 확인하지 못했다).
- Qwen2.5-Omni-3B/7B는 vLLM LoRA를 지원하고, "억양별 LoRA + verbatim/교정 이중 출력" 공개 선례(2609.20828)가 있다. 그래서 이 프로젝트의 [방언 전사 + 표준어 변환] 출력 구조와 가장 가까운 검증된 경로다.
- Qwen3-ASR은 한국어 기본 성능이 가장 좋다(Fleurs ko 2.57). 하지만 지시를 따르지 않도록 SFT된 ASR 전용 모델이다. 억양 라벨과 표준어 변환을 붙이려면 출력 형식 자체(예: `<asr_text>` 뒤에 사용자 정의 태그)를 SFT로 새로 가르쳐야 한다. 이런 다중 출력 공개 사례는 찾지 못했다.
- 공개 사례의 공통 설정은 LLM에 LoRA r 16~64, α 32~128, lr 5e-5~2e-4(QLoRA는 1e-4~2e-4), 1~3 epoch다. 오디오 인코더는 대부분 고정하고, 필요할 때만 projector나 오디오 인코더 상위 층을 함께 연다.

### Gaps
- Qwen3-Omni를 한국어(또는 방언) ASR로 LoRA/QLoRA 미세조정해 정량 결과를 낸 공개 사례는 찾지 못했다. ms-swift 공식 Qwen3-Omni 오디오 예제와 메모리 수치도 확인하지 못했다(커뮤니티 저장소 HardikSailor/qwen3_omni_speechllm은 404).
- QLoRA로 학습한 Qwen3-Omni adapter를 bf16 vLLM에 올렸을 때 정확도가 얼마나 차이 나는지는 자료가 없다.
- transformers 최신 버전에서 Qwen3-Omni MoE 전문가가 fused 파라미터로 구현돼 PEFT LoRA를 전문가에 붙일 수 없는지는 확인하지 못했다.
- Qwen3-ASR 공식 SFT 스크립트의 LoRA 옵션 유무와 필요 메모리는 문서에 없다.

---

## 2. 음성 LLM의 다중 과제 미세조정 (SALMONN, LTU/LTU-AS, Qwen-Audio 계층 태그, WavLLM, Typhoon-Audio, DeSTA): 과제 혼합, 프롬프트/태그, 비율, 분류 과제가 ASR을 해치는가

### Takeaway
다중 과제를 섞으면 ASR이 소폭 나빠지는 사례가 반복해서 보고된다. Typhoon-Audio에서는 QA·번역을 추가하자 태국어 WER이 16.80 → 19.84로 올랐다. E2E 억양+ASR 공동학습도 ASR이 약간 나빠졌다. SALMONN은 ASR/캡션처럼 결정적인 과제가 지배하는 "task overfitting"을 보고했다. 해법은 세 가지다. Qwen-Audio식 명시적 과제 태그로 간섭을 분리하거나, 대상 언어 비중을 확보하거나(10%만 넣어도 큰 개선), 한 응답 안에 여러 출력을 고정 형식으로 함께 내게 하는 것이다(2609.20828의 dual-output).

### Cited Findings
- **Qwen-Audio 계층 태그**(6단계, 순서대로): ① `<|startoftranscripts|>`(전사·번역) 또는 `<|startofanalysis|>` ② 오디오 언어 태그(8개 언어, 비음성은 `<|unknown|>`) ③ 과제 태그 `<|transcribe|>`/`<|translate|>`/`<|caption|>`/`<|analysis|>`/`<|question-answer|>` ④ 출력 텍스트 언어 태그 ⑤ `<|timestamps|>`/`<|notimestamps|>` ⑥ 출력 지시. 30개 이상 과제를 다루며 공통 태그는 공유를, 특정 태그는 간섭 회피를 노린다. 사전학습은 LLM을 고정하고 인코더(Whisper-large-v2 초기화, 약 640M)만 학습했다. SFT는 인코더를 고정하고 LLM만 학습했으며 지시 데이터는 2만 개다. 단어 단위 타임스탬프(SRWT) 과제를 넣으면 ASR이 좋아졌다(LibriSpeech test-clean 2.22 → 2.04, AISHELL1 test 1.71 → 1.29). — [Qwen-Audio, arXiv 2311.07919](https://arxiv.org/abs/2311.07919)
- **SALMONN**: Whisper-large-v2와 BEATs 인코더, Vicuna-13B는 고정하고 window-level Q-Former(창 0.33초, 30초에 88토큰)와 LoRA(r=8, scaling 4.0, q·v, 약 33M = 0.24%)만 학습했다. 지시 튜닝은 12개 과제, 약 4,400시간, 약 230만 샘플이다(ASR 960+220 h, 화자검증 1,200 h, 감정인식 5 h 등). "task over-fitting"(지시를 무시하고 ASR/캡션식으로 답함)의 원인은 ASR·AAC의 결정적인 출력이다. 해법으로 LoRA scaling을 추론 때 약 2.0으로 낮추거나, 그렇게 생성한 긴 이야기 12개로 12 step "activation tuning"을 했다. — [SALMONN, arXiv 2310.13289](https://arxiv.org/abs/2310.13289)
- **Typhoon-Audio**(Interspeech 2025): SALMONN 구조에 Typhoon-1.5-8B, 태국어 미세조정 Whisper-large-v3 + BEATs를 썼다. 1단계는 adapter만 학습했다(약 182만 예: ASR 169만, 캡션 13.5만). 2단계 SFT는 adapter와 LoRA(r=8, α=32)를 학습했고 64만 예의 비율은 Speech IF 약 32%, Speech QA 약 27%, **ASR 약 16%**, Audio QA 약 11%, 캡션 약 8%, 번역 약 6%다. 영어로만 SFT하면 태국어 ASR WER이 80.86으로 붕괴했고, 태국어를 10%만 섞자 16.80이 됐다. QA·번역·MCQ를 추가할수록 16.80 → 18.33 → 19.84로 나빠졌다. "ASR을 시켜도 명령에 응답해 버리는" 현상이 있었다. 사전학습 모델(SFT 전)의 ASR은 13.52로 가장 좋았다. — [Typhoon-Audio, arXiv 2409.10999](https://arxiv.org/abs/2409.10999)
- **LTU-AS**(ASRU 2023): Whisper + TLTR(시간·층 Transformer)로 인식하고 LLaMA + LoRA로 추론한다. Open-ASQA는 960만 개(13개 데이터셋, 그중 690만 개는 GPT가 생성한 개방형 QA)이고 지시 수행률은 95% 이상이다. — [LTU-AS, arXiv 2309.14405](https://arxiv.org/abs/2309.14405)
- **WavLLM**(EMNLP 2024): Whisper(내용)와 WavLM(화자·음향) 이중 인코더를 쓴다. 1단계는 ASR·번역·화자검증·감정 등 기초 과제 혼합, 2단계는 복합 과제로 학습한다. 2단계에 prompt-aware LoRA weight adapter를 넣는다. — [WavLLM, arXiv 2404.00656](https://arxiv.org/abs/2404.00656)
- **DeSTA2.5-Audio**: 과제에 무관한 DeSTA-AQA5M(약 500만, 50개 데이터셋)으로 학습한다. 응답을 백본 LLM이 스스로 생성해 분포 이동과 망각을 줄인다고 주장한다. — [arXiv 2507.02768](https://arxiv.org/abs/2507.02768)
- **억양 분류 + ASR 공동학습(E2E, 비LLM)**: 억양 인식은 크게 좋아졌지만 ASR WER은 단일 과제보다 "약간 나빴다". 억양 손실이 작은데도 ASR을 해쳐 손실 가중치 λ를 조정했다. — [E2E-based Multi-task Learning for Joint Speech and Accent Recognition, arXiv 2106.08211](https://arxiv.org/abs/2106.08211)
- **한 응답에 다중 출력**: Qwen2.5-Omni-3B 억양별 LoRA가 한 번의 forward로 verbatim 전사와 교정 전사를 함께 낸다. 그 상태에서도 WER은 40~46% 줄었다. — [arXiv 2609.20828](https://arxiv.org/abs/2609.20828)
- 저자원 언어에서는 프롬프트에 언어 이름을 명시하는 "language-aware" 지시(`Transcribe the [Language] audio into text.`)가 일부 언어에서 도움이 됐다. — [LoASR-Bench, arXiv 2603.20042](https://arxiv.org/abs/2603.20042)

### Inferences
- 이 프로젝트의 세 출력(억양·방언 전사·표준어)은 모두 같은 음성에 대한 것이다. 그래서 과제를 따로 섞기보다 "하나의 고정 프롬프트 → 하나의 구조화 응답"(예: `억양: 경상\n전사: …\n표준어: …`)으로 묶으면 Typhoon식 과제 경쟁과 SALMONN식 task overfitting을 상당 부분 피할 수 있다. 선례는 2609.20828의 dual-output이다.
- 출력 순서는 전사를 먼저, 억양 라벨을 나중에 두는 쪽이 합리적일 수 있다. 전사를 먼저 생성하면 억양 판단이 어휘 단서(어미·어휘)를 조건으로 쓸 수 있기 때문이다. 이 순서를 비교한 실험은 찾지 못했으므로 추론이다.
- 억양 라벨은 출력 토큰이 1~2개뿐이라 손실 기여가 작다. 그래도 E2E 연구에서는 ASR을 약간 해쳤다. 억양 정확도와 CER을 모두 검증 세트로 추적해 균형(데이터 비율 또는 손실 가중)을 잡아야 한다.
- 일반 대화·지시 능력을 유지하려면 표준어 ASR과 일반 지시 데이터를 일부 섞거나(Typhoon의 10% 사례), DeSTA·교차모달 KD처럼 원 모델이 생성한 응답을 섞는 방법이 근거가 있다.

### Gaps
- LALM에서 [ASR + 억양 분류]를 함께 학습했을 때 ASR이 얼마나 나빠지는지 직접 측정한 논문은 찾지 못했다.
- 출력 순서(라벨 먼저 vs 전사 먼저)나 과제 태그와 자연어 지시를 비교한 음성 LLM 실험도 찾지 못했다.

---

## 3. 전문 교사(ASR 교사 + 분류 교사)에서 단일 LALM으로 지식 증류, 의사 라벨 품질 필터링, 클래스 soft-label 증류

### Takeaway
가장 잘 검증된 방식은 "교사가 만든 hard pseudo-label + 품질 필터"다. Distil-Whisper는 WER 10% 임계값으로 데이터의 45%를 버려 평균 WER을 13.4 → 11.4로 낮췄다. 정답이 없으면 교사 두 개의 일치도(pWER)로 거르는 uDistil-Whisper 방식이 있다. LALM 쪽 soft-label(KL) 증류는 텍스트 LLM 교사 → 음성 LLM 학생 사례(λ=0.5, τ=2)가 있다. 그러나 분류기 교사의 클래스 확률을 생성형 LALM에 증류한 직접 사례는 찾지 못했다.

### Cited Findings
- **Distil-Whisper**: 정답 전사와 Whisper 의사 라벨을 각각 정규화한 뒤 WER을 계산하고, WER이 λ를 넘는 샘플을 버린다. 결과 표: λ=100(필터 없음) → 평균 WER 13.4, λ=40 → 12.0(11.9% 제거), λ=20 → 11.7(24.2%), **λ=10 → 11.4(45.4% 제거)**, λ=5 → 11.4(60.3%). 임계값이 높으면 오전사·환각이 들어오고, 낮으면 쉬운 샘플만 남는다. 학습 데이터는 9개 공개 데이터셋 21,170시간이다(부록에는 21,770시간으로 적혀 있어 불일치). — [Distil-Whisper, arXiv 2311.00430](https://arxiv.org/abs/2311.00430)
- **uDistil-Whisper**: 정답 없이 proxy ASR 모델과의 불일치(pWER), 교사 logit 등으로 거르는 label-free 필터다. 최상 학생이 교사보다 WER 5~7점 낫다고 주장한다(저자원 환경). — [uDistil-Whisper, arXiv 2407.01257](https://arxiv.org/abs/2407.01257)
- **CantoASR(ASR 교사 + LALM 협업)**: Whisper의 문자 단위 신뢰도 0.7 이상은 그대로 두고, 미만인 부분만 미세조정한 Qwen2-Audio가 교정한다. N-best를 넣을수록 좋아졌다(1-best 11.52 → 10-best 9.52, lattice 제약 8.04). 평균 CER은 ASR만 21.24, ASR 미세조정 15.64, +LLM 미세조정 11.55, 최종 11.19였다. — [CantoASR, arXiv 2511.04139](https://arxiv.org/abs/2511.04139)
- **LALM KL 증류 손실 형태**: L = CE(y 또는 교사 출력 ŷ) + λτ²·KL(교사‖학생), λ=0.5, τ=2. 교사 출력을 hard target으로 쓰는 "Teacher CE"와 KL을 함께 써야 효과가 났다. 단순 CE는 오히려 성능을 떨어뜨렸다(75.08 → 71.25). — [arXiv 2509.14930](https://arxiv.org/abs/2509.14930)
- **Raon-Speech(KRAFTON)**: 2단계 종단 사전학습에서 L_total = L_CE + λ·L_KD를 썼다. 교사는 원래 텍스트 LLM이고 학생이 출력 분포를 맞추게 했다(λ 값은 비공개). — [Raon-Speech, arXiv 2605.23912](https://arxiv.org/abs/2605.23912); [alphaXiv 개요](https://www.alphaxiv.org/overview/2605.23912)
- 감정 인식에서 LALM 교사와 학생의 오디오 logit에 temperature softmax를 씌우고 KL을 쓴 증류(PL-Distill)가 있다(검색 요약). 비LLM SER에서는 교사의 감정 확률분포를 soft label로 KL + hard CE를 함께 쓴 사례가 있다. — [arXiv 2602.01547](https://arxiv.org/abs/2602.01547); [arXiv 2312.13556](https://arxiv.org/abs/2312.13556)
- VoxEmo(35개 감정 코퍼스)는 분포 인식 soft-label 평가를 제안했다. zero-shot 음성 LLM은 hard 정확도가 지도학습 모델보다 낮지만 사람 라벨 분포와는 더 잘 맞았다(검색 요약). — [VoxEmo, arXiv 2603.08936](https://arxiv.org/abs/2603.08936)
- 소규모 데이터에서는 사람이 검증한 정답으로 교사 품질을 보정하는 방식이 쓰였다(Gemini 2.5 Pro 참조 전사 + 750개 사람 검증). — [arXiv 2609.20828](https://arxiv.org/abs/2609.20828)

### Inferences
- 이 프로젝트에 적용하면 다음 순서가 된다. (1) 정답 전사가 있는 AI Hub 데이터: 교사(Gemma 4 LoRA)가 아니라 정답으로 학습한다. (2) 정답이 없는 영상 데이터: Gemma 4 LoRA와 다른 ASR(예: Qwen3-ASR 미세조정본)의 CER 일치도(pWER)로 거르고, Distil-Whisper처럼 10% 안팎을 출발점으로 잡는다. (3) 억양 교사(Whisper 인코더 + head)의 softmax 확률이 높은(예: 최대 확률 상위) 샘플만 억양 라벨로 쓴다.
- 생성형 LALM에서 억양 soft-label 증류는 이렇게 할 수 있다. 억양 이름 토큰(예: "경상" 첫 토큰) 위치의 학생 logit을 5개 후보 토큰으로 제한하고, 교사 분포와 KL을 계산하는 보조 손실을 추가한다(λτ² 형태는 2509.14930을 따름). 표준 SFT 프레임워크(ms-swift, LLaMA-Factory)는 이를 기본 지원하지 않으므로 사용자 정의 loss가 필요하다. 근거 있는 선례가 없는 추론이다.
- 표준어 변환 출력은 정답이 없으면 텍스트 LLM 교사(예: 방언 → 표준어 변환 프롬프트)로 만든다. 이때 교사 결과를 다시 방언 전사와 비교하는 일관성 검사(어순·내용 보존)로 걸러야 한다.

### Gaps
- 분류기 교사의 soft label을 생성형 LALM의 라벨 토큰에 증류한 논문은 찾지 못했다.
- 다중 교사(ASR 교사 + 분류 교사)에서 하나의 LALM으로 증류한 음성 사례도 찾지 못했다.
- 방언 전사에 맞는 pWER 임계값은 근거가 없다. Distil-Whisper의 10%는 영어 WER 기준이다.

---

## 4. 영상(유튜브)에서 ASR/방언 데이터 만들기: YODAS, GigaSpeech, Emilia, 한국어 사례, 자막 정렬·필터링, Whisper 의사 라벨, 화자 분리, 품질 이득 vs 잡음, 법·라이선스(한국·EU), 영상 프레임의 효용

### Takeaway
대규모 영상 코퍼스는 모두 "CC 라이선스 필터 → (자막 있으면) CTC 정렬 점수 / (없으면) Whisper 의사 라벨 → 언어 판별 → 음질(DNSMOS) → 화자 분리" 단계를 거친다. 필터 후 남는 비율은 30~45%로 작다. YODAS 원본을 거르지 않고 쓰면 반복 환각으로 WER이 100%를 넘었다. 방언에서는 Whisper가 방언을 표준어로 "번역"해 버리는 문제(스위스 독일어)가 확인됐다. 그래서 일반 Whisper 의사 라벨은 방언 verbatim 전사의 정답으로 쓸 수 없다. 영상 프레임은 깨끗한 음성에서 이득이 거의 없고(Whisper-Flamingo clean 1.0 vs 1.1) 잡음 환경에서만 크다(babble 0 dB에서 WER 11.7 → 5.6). 법적으로는 YouTube 약관이 크롤링·다운로드를 제한한다. 한국은 TDM 면책 조항 없이 공정이용(사안별 판단)에 의존하고, 2026년 5월 정부 안내서가 연구 목적을 유리한 요소로 보았다.

### Cited Findings

**대규모 영상 기반 코퍼스와 파이프라인**
- **YODAS**: 수동 자막 86,400시간, 자동 자막 335,845시간, 비라벨 144,174시간(2023년 7월 기준 56만 시간 이상)이다. 키워드(위키 덤프)와 채널 기반 크롤러를 쓰고, YouTube 필터로 자막이 있고 CC 라이선스인 영상만 받았다. 사전학습 음향 모델의 CTC 손실 점수가 2.0보다 나쁜 발화는 제외했다. 영어 학습 시 수동 자막 대비 자동 자막 CER은 14.9 vs 32.3(자동 자막의 deletion 26.6)이었다. "considerable noise"가 있고 웃음·음악 같은 자막 표기가 섞이며, 정렬 오류의 주원인은 음악·배경 잡음이다. 라이선스는 CC BY 3.0이다. 한국어 시간 수는 원 논문에 없다. — [YODAS, arXiv 2406.00899](https://arxiv.org/abs/2406.00899)
- **OWSM v4의 YODAS 정제**: 370k h(149개 언어) → OWSM-CTC v3.2로 CTC segmentation 재분할(30초 이하, 비음성 제거) 후 345k h → 언어 판별(텍스트 fastText와 음성 ECAPA-TDNN이 모두 원래 라벨과 일치) 후 284k h → 언어별 CTC 점수 하위 10% 분위 제거(긴 녹음 안에서 하나라도 걸리면 녹음 전체 제거) 후 **166k h(75개 언어)**. **한국어는 필터 전(θ=0) 17.5k h, θ=0.10에서 10.9k h**다. FLEURS 한국어 CER은 OWSM v3.1 20.09 → v4 6.74(MMS 18.30)로 좋아졌다. CTC 필터 없이 학습하면 "몇 토큰 반복에 갇혀" WER이 100%를 넘었다. — [OWSM v4, arXiv 2506.00338](https://arxiv.org/abs/2506.00338)
- Granary(유럽 25개 언어)는 YODAS 주석을 YouTube 자막이라 신뢰하기 어렵다고 보았다. 언어 판별 오류로 일부 언어(불가리아어·우크라이나어)는 20%만 남았다. — [Granary, arXiv 2505.13404](https://arxiv.org/abs/2505.13404)
- **GigaSpeech**: 오디오북·팟캐스트·YouTube 40,000시간 중 10,000시간을 고품질 라벨로 썼다. 새 강제정렬·분할 파이프라인을 쓰고, 검증 단계에서 10,000시간 세트는 WER 4% 이하, 작은 세트는 0%로 제한했다. — [GigaSpeech, arXiv 2106.06909](https://arxiv.org/abs/2106.06909)
- **Emilia-Pipe**(자막 없는 in-the-wild 영상·팟캐스트용): ① 24 kHz mono, -20 dBFS로 표준화 ② UVR-MDX-Net Inst 3로 음원 분리 ③ pyannote/speaker-diarization-3.1로 화자 분리 ④ Silero-VAD로 3~30초 분할 ⑤ faster-whisper의 Whisper medium으로 ASR ⑥ 필터: Whisper 언어 신뢰도 80% 미만 제거, **DNSMOS P.835 OVRL 3.0 초과만 유지**, 평균 음소(문자) 길이 IQR 1.5배 밖 제거. 처리 속도는 8×RTX 4090에서 원시 음성 2.5시간/분이다. 평가에서 약 600 h 중 176.22 h(29.43%)가 남았다. 원본 Emilia 101,654 h 중 **한국어는 약 0.2k h(README 기준 217 h)**뿐이다. — [Emilia, arXiv 2407.05361](https://arxiv.org/abs/2407.05361)
- Emilia-Large(216k h 이상) 평가에서는 666.94 h 원시 데이터의 DNSMOS가 2.52 → 음원 분리 후 2.88 → 최종 3.26이었다. 남은 양은 258.44 h(38.75%)였다. 화자 분리 단계에서 클립 431,876개, 515.50 h(77.29%)가 됐다. — [Emilia-Large, arXiv 2501.15907](https://arxiv.org/abs/2501.15907)

**한국어 영상 기반 코퍼스**
- **Pansori-TEDxKR**: 2010~2014년 한국 TEDx 강연에서 만든 약 3시간, 화자 41명 코퍼스다. 커뮤니티 번역가가 손으로 만든 자막이 있는 강연만 썼다. 자막 경계로 분할한 뒤 도구 보조 수동 정렬을 하고 상용 인식기로 검증했다. 라이선스는 CC BY-NC-ND 4.0이다. — [OpenSLR 58](https://openslr.org/58/); [arXiv 1812.09798](https://arxiv.org/abs/1812.09798)
- **KMSAV**(ETRI Journal 46(1), 2024): 사람이 전사한 한국어 시청각 데이터 약 150시간과, 전사되지 않은 CC 라이선스 YouTube 영상 약 2,000시간이다. 미세조정 후 CER은 ASR 11.1%, AVSR 18.9%였다(검색 요약; 본문 접근 403). — [KMSAV DOI 10.4218/etrij.2023-0352](https://doi.org/10.4218/etrij.2023-0352); [ETRI KSP PDF](https://ksp.etri.re.kr/ksp/article/file/69217.pdf)
- OLKAVS: 스튜디오에서 9개 시점으로 촬영한 한국어 시청각 1,150시간, 화자 1,107명이다(검색 요약). — [OLKAVS, arXiv 2301.06375](https://arxiv.org/abs/2301.06375)
- 일본 TV 자막 코퍼스 연구는 자막 타임스탬프가 "항상 제때가 아니다"라며 별도 정렬이 필요하다고 지적했다. — [arXiv 2103.14736](https://arxiv.org/abs/2103.14736)

**방언과 의사 라벨의 위험**
- Whisper는 스위스 독일어 음성을 방언 전사가 아니라 **표준 독일어 번역**으로 출력하는 경향이 있다(VarDial 2024). — [Does Whisper Understand Swiss German?, arXiv 2404.19310](https://arxiv.org/abs/2404.19310)

**영상 프레임(AVSR)의 효용**
- Whisper-Flamingo(LRS3): clean에서 오디오만 1.0% / 0.68%, AV 1.1% / 0.76%로 이득이 없었다. babble 잡음 0 dB에서는 오디오만 11.7% / 11.1% → AV 5.6%로 49.5% 상대 감소였다. — [Whisper-Flamingo, arXiv 2406.10082](https://arxiv.org/abs/2406.10082)
- KMSAV 한국어에서도 AVSR CER(18.9%)이 ASR(11.1%)보다 나빴다(검색 요약). — [KMSAV](https://doi.org/10.4218/etrij.2023-0352)

**법·라이선스**
- YouTube 약관은 제3자 크롤링을 금지하고, 업로드 콘텐츠에 대해 Google에 (ML 포함 가능한) 이용 라이선스를 준다. 이 약관이 크리에이터의 CC 라이선스보다 우선할 수 있다. 음성·영상 학습 데이터의 주요 출처가 YouTube라는 점도 지적됐다. — [Bridging the Data Provenance Gap Across Text, Speech and Video, arXiv 2412.17847](https://arxiv.org/abs/2412.17847)
- YODAS는 CC 라이선스 영상만 모았다고 밝힌다(CC BY 3.0으로 배포). — [arXiv 2406.00899](https://arxiv.org/abs/2406.00899)
- 한국 저작권법: TDM 면책 개정안은 2024년 5월 21대 국회 임기 만료로 폐기됐다. 공정이용 항변이 AI 학습에 적용되는지는 불확실하다. — [Legal 500 South Korea TMT](https://legal500.de/c/south-korea/tmt-local-firms/in-focus)
- 2026년 5월 문체부와 저작권 당국은 생성형 AI 학습의 공정이용 적용 안내서를 냈다(PDF 파일명 "Guide_on_Applicability_of_the_Fair_Use_Doctrine_to_Training_of_GenAI_Models"). 안내서 요지는 다음과 같다. 학습은 복제를 수반할 수 있어 원칙상 사전 허락이 필요하다. 공정이용은 4요소(목적·성격, 저작물 성격, 이용량, 시장 영향)로 사안별로 판단하며 최종 판단은 법원이 한다. "연구·분석용" 모델이 원본을 재현하지 않으면 인정 가능성이 높다. — [Asia IP Law, 2026-05-20](https://asiaiplaw.com/section/cover-story/south-korea-issues-guidance-on-copyright-and-ai-training-data)
- 개인정보: 개인정보보호위원회는 2024년 7월 「AI 개발·서비스를 위한 공개된 개인정보 처리 안내서」를 냈다. 2025년 8월에는 생성형 AI 개인정보 처리 가이드라인을 냈고, PIPA 제15조 제1항 제6호(정당한 이익)를 공개 데이터 학습의 근거로 제시하며 목적 정당성·필요성·이익형량을 요구했다. 법적 구속력은 없고 최종 확정 여부는 미검증이다(2차 출처). — [PPC Land](https://ppc.land/south-korea-establishes-ai-privacy-framework-with-new-guidelines/); [Baker McKenzie](https://connectontech.bakermckenzie.com/south-korea-sets-ai-standard-pipcs-guidelines-for-generative-ai-present-obligations-opportunity/)
- EU DSM 지침: 제3조는 연구기관·문화유산기관의 과학 연구 목적 TDM 면책이며 권리자가 opt-out할 수 없다. 제4조는 상업 포함 일반 TDM이고 권리자가 "기계 판독 가능한 방식 등 적절한 방법"으로 유보하면 쓸 수 없다. — [Kluwer Copyright Blog](https://legalblogs.wolterskluwer.com/copyright-blog/the-new-copyright-directive-text-and-data-mining-articles-3-and-4/)
- EU AI Act 제53조 제1항 (c)는 범용 AI 제공자에게 저작권 정책과 DSM 제4조 제3항 유보 준수를, (d)는 학습 데이터 요약 공개를 요구한다. 2025년 8월 2일부터 적용되며 Code of Practice는 2025년 7월 10일에 공개됐다. — [Technollama](https://technollama.co.uk/the-eu-ai-act-and-copyright); [Kluwer: Copyright, the AI Act and extraterritoriality](https://legalblogs.wolterskluwer.com/copyright-blog/copyright-the-ai-act-and-extraterritoriality/)

### Inferences
- 방언 영상 데이터에서 "자동 자막 또는 Whisper 의사 라벨 = 정답"으로 쓰면 방언 어휘·어미가 표준어로 바뀌어 있을 위험이 크다(스위스 독일어 사례, YODAS 자동 자막 CER 32.3). 현실적인 쓰임새는 세 가지다. (a) 사람이 손으로 만든 자막이 있는 영상만 쓰거나 (b) 이미 방언으로 학습된 교사(Gemma 4 LoRA)로 라벨을 만들고 두 번째 모델과의 일치도로 거르거나 (c) 전사 라벨 없이 억양 분류용(지역 라벨은 채널·출연자 출신 정보로)으로만 쓴다.
- 영상의 지역 라벨(억양)은 화자 단위로 붙는다. 그래서 pyannote 화자 분리 + 화자 임베딩으로 출연자와 분리된 구간을 구분해야 한다. 아니면 진행자(표준어)와 방언 화자가 섞여 억양 라벨 잡음이 생긴다.
- 품질 필터 기준점: CTC/정렬 점수 하위 10% 제거(OWSM v4), DNSMOS OVRL 3.0 초과(Emilia), 언어 신뢰도 80% 이상(Emilia), 의사 라벨 일치 WER 10% 이하(Distil-Whisper). 공개 파이프라인에서 남는 비율은 30~45%다. 그러니 원시 영상 시간을 목표 시간의 2.5~3배 이상 모아야 한다.
- 이 앱의 입력은 마이크 음성이다. 영상 프레임(입술)은 clean 조건에서 이득이 없고 코칭 앱 입력에는 영상이 없으므로, 영상은 오디오 출처로만 쓰는 것이 맞다.
- 법적으로 가장 위험이 적은 경로는 다음과 같다. CC 라이선스 영상만(YODAS 방식) 쓰고, 연구 목적·비공개 학습·원본 재배포 금지를 지키고, 개인정보(얼굴·이름·음성) 최소화와 가명화를 하는 것이다. 다만 YouTube 약관의 다운로드 금지는 CC 라이선스와 별개의 계약 문제로 남는다. 상업 서비스로 이어지면 한국 공정이용 판단이 불리해질 수 있다(시장 영향 요소). 법률 자문이 필요하다.

### Gaps
- 한국어 방언 영상(사투리 유튜브)으로 ASR/LALM 학습 데이터를 만든 논문이나 데이터셋은 찾지 못했다(한국어 검색 포함).
- YODAS 한국어 중 수동 자막 대비 자동 자막 비율과 한국어 자동 자막 오류율은 원 논문에 없다.
- 한국 법원의 AI 학습 공정이용 판결, YouTube 약관 위반에 대한 연구 목적 예외 판례는 찾지 못했다. EU AI Act 제2조 제6항(연구 전용 모델 제외)의 정확한 범위도 확인하지 못했다.
- 2025년 8월 PIPC 생성형 AI 가이드라인의 확정 여부, 음성의 생체정보 해당 여부도 1차 출처로 확인하지 못했다.

---

## 5. 이 LALM들을 한국어(특히 방언)로 미세조정한 보고

### Takeaway
한국어 미세조정 공개 사례는 Qwen3-ASR(TeamUNIVA 커뮤니티 모델: 한국어 CER 6.86% → 5.37%; MLC-SLM 2026 대회: 한국어 tcpCER 27.17 → 25.14)과 Qwen2-Audio(LoASR-Bench: CommonVoice 한국어 2시간 학습으로 WER 0.34 → 0.12) 정도다. Qwen2.5-Omni·Qwen3-Omni를 한국어 방언으로 미세조정한 공개 보고는 찾지 못했다. 한국어 음성 LLM으로는 KRAFTON Raon-Speech(9B, 영·한 138만 시간)가 있지만 방언은 다루지 않는다.

### Cited Findings
- TeamUNIVA/qwen3_asr_1.7b_ko_beta: 한국어 합계 CER 0.0686 → 0.0537, KsponSpeech 0.0865 → 0.0670, common_voice_ko 0.0719 → 0.0628, fleurs_ko 0.0142 → 0.0138, zeroth 0.0264 → 0.0230이다. 평가 세트는 사람이 검토했다(8,489개 중 8,383개 유지). 후속 모델은 `TeamUNIVA/qwen3_asr_1.7b_ko`이고 라이선스는 apache-2.0이다. 학습 데이터·방법은 비공개다. — [HF 모델 카드](https://huggingface.co/TeamUNIVA/qwen3_asr_1.7b_ko_beta)
- MLC-SLM 2026(2인 대화, 다국어) Qwen3-ASR-1.7B 적응: 한국어 tcpCER은 Whisper-large-v3 31.58, Omniasr-LLM-7B-v2 33.25, Qwen3-ASR-1.7B 기본 27.17, 적응 후 25.14다. 저자들은 한국어·일본어·태국어에서 반복과 환각이 특히 많았다고 보고했다. — [arXiv 2607.08208](https://arxiv.org/abs/2607.08208)
- LoASR-Bench 한국어(CommonVoice 16.1, 학습 2시간, WER): xlsr-53 0.20, Whisper-M 0.18, Whisper-L 0.17, Qwen2-Audio 0.34, Qwen2-Audio 미세조정 0.12, Qwen3-Omni 0.09. — [arXiv 2603.20042](https://arxiv.org/abs/2603.20042)
- Qwen3-ASR 공식 한국어 수치(1.7B): CommonVoice 5.88, Fleurs 2.57, MLC-SLM 8.61이다. Qwen3-ASR-Flash-1208은 각각 3.82, 2.07, 8.09다. — [arXiv 2601.21337](https://arxiv.org/abs/2601.21337)
- Raon-Speech(KRAFTON, 2026): Qwen3-VL-8B-Instruct 백본 + AuT 기반 인코더 + 2층 MLP adapter로 총 9B이고, 영·한 음성·텍스트 약 138만 시간으로 학습했다. 3단계는 adapter 정렬 → 인코더를 뺀 전체 종단 사전학습 + 텍스트 LLM 교사 KD → SimPO다. 42개 벤치마크에서 Qwen2.5-Omni 등 8개 모델과 비교했지만 한국어 CER 수치와 방언 언급은 개요에 없다. — [arXiv 2605.23912](https://arxiv.org/abs/2605.23912); [alphaXiv 개요](https://www.alphaxiv.org/overview/2605.23912)
- KRAFTON 벤치마크 표에서 Qwen2.5-Omni의 한국어 종합 점수는 0.55(영어 0.73)로 6위였다(검색 요약, 지표 정의 미확인). — [KRAFTON/A.X-K2-Raon-Speech-21B-A3B](https://huggingface.co/KRAFTON/A.X-K2-Raon-Speech-21B-A3B)
- 다른 공개 한국어 영상 기반 자원으로는 KMSAV(CC YouTube 약 2,000시간 비라벨 + 150시간 전사)와 OWSM v4 정제 YODAS 한국어 10.9k h(ESPnet 공개 예정이라고 밝힘)가 있다. — [KMSAV](https://doi.org/10.4218/etrij.2023-0352); [arXiv 2506.00338](https://arxiv.org/abs/2506.00338)

### Inferences
- 한국어 표준어 ASR 기본기는 Qwen3-ASR ≥ Qwen3-Omni > Qwen2.5-Omni 순으로 보인다(Fleurs 한국어 Qwen3-ASR 2.57, Qwen3-Omni 19개 언어 평균 5.33, Qwen2.5-Omni 14.04). 방언 verbatim 전사는 어느 모델이든 미세조정이 필요하다.
- OWSM v4가 정제한 YODAS 한국어 10.9k h는 일반 한국어(대부분 표준어) 보강 또는 표준어 유지 데이터로 쓸 수 있는 큰 공개 자원이다. 다만 방언 비율은 알 수 없다.

### Gaps
- 한국 방언(강원·충청·경상·제주) 데이터로 Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, Qwen3-ASR을 미세조정한 논문이나 모델은 찾지 못했다. AI Hub 방언 데이터를 쓴 LALM 미세조정 보고도 찾지 못했다.
- 한국어 LALM의 억양(지역) 분류 정확도를 보고한 자료도 찾지 못했다.
- 정제된 YODAS 한국어의 실제 공개 여부와 라이선스는 확인하지 못했다.
