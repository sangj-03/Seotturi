# Conversation-partner scenes

The app's interviewers, audiences and everyday-conversation partners (`app/web/js/personas.js`) each have a scene, shown behind the preparation and the session, and a card avatar. They were drawn on the DGX Spark with [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (Apache-2.0) and its [8-step Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning), each edited from one of the app's own character images (`refs/`: friend, interviewer, audience, mascot) so that the soft 3D style of the team's design stays the same. 깐깐한 부장님, 우호적인 동료들 and 도윤 keep the original images.

## Pipeline

```bash
cd tools/img-gen
python download.py                     # Qwen-Image-Edit-2511 and the Lightning LoRA into the Hugging Face cache
./run_gen.sh                           # encode the prompts of jobs.json, then draw 2 seeds per job → out/<id>/s<seed>.png
python sheet.py sheet.png barista grandma ...   # contact sheet to choose a seed
./run_touchup.sh                       # second pass on the chosen drawings (touchup.json: background blur, fixes)
python finalize.py barista=s1 grandma=s0 ...    # scene + avatar into app/web/assets/personas/, eye and mouth boxes
python specs_block.py barista grandma ...       # writes those boxes into app/scripts/alive_frames.py
```

Then run `app/scripts/alive_frames.py` to draw the blink and talk frames.

- **Memory.** The GPU shares one memory with the CPU and with the site's engine, so both stages keep their footprint small. Encoding runs in its own process (the 17 GB text encoder is gone when it exits). Drawing builds the 20B transformer tensor by tensor on the GPU, adds the Lightning LoRA to each bf16 weight and stores it as FP8 (about 21 GB instead of 41), and waits while the system reports memory pressure. `watchdog.sh` stops the drawing when available memory falls below 6 GB or pressure rises above 40%.
- **Faces.** `faces.py` finds the eyes and mouth corners with OpenCV's YuNet (`models/yunet.onnx`, from the [OpenCV model zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)); `finalize.py` turns them into the eye and mouth boxes, and `overrides.json` corrects the faces it gets wrong.
- **Checks.** `boxview.py` shows a face with its boxes on a grid, `aliveview.py` the original next to its blink and talk frames.
- `diffusers` 0.39.0 is installed into `pylib/` (`pip install --target pylib diffusers==0.39.0`); the scripts run with `PYTHONPATH=pylib`.

| File | Purpose |
| --- | --- |
| `jobs.json` | one job per persona: reference image, prompt, shared style text |
| `touchup.json` | second-pass jobs on the chosen drawings |
| `gen.py` | `encode` and `draw` stages |
| `finalize.py`, `avatar.py`, `faces.py` | scene and avatar files, eye and mouth boxes |
| `specs_block.py`, `specs.json` | boxes for `alive_frames.py` |
| `judges_spec.json` | hand-set boxes for the panel of interviewers |
