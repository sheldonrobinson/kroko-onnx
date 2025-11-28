## **Open-source speech recognition built for developers.**
>
> Our engine is fully open-source, and you choose how to deploy models: use our **CC-BY-SA licensed community models** or upgrade to **commercial models** with premium performance. We focus on building **fast, high-quality production models** and providing **examples that take the guesswork out** of integration.
 ### Supported functions

|Speech recognition| [Speech synthesis][tts-url] | [Source separation][ss-url] |
|------------------|------------------|-------------------|
|   ✔️              |         ✔️        |       ✔️           |

|Speaker identification| [Speaker diarization][sd-url] | Speaker verification |
|----------------------|-------------------- |------------------------|
|   ✔️                  |         ✔️           |            ✔️           |

| [Spoken Language identification][slid-url] | [Audio tagging][at-url] | [Voice activity detection][vad-url] |
|--------------------------------|---------------|--------------------------|
|                 ✔️              |          ✔️    |                ✔️         |

| [Keyword spotting][kws-url] | [Add punctuation][punct-url] | [Speech enhancement][se-url] |
|------------------|-----------------|--------------------|
|     ✔️            |       ✔️         |      ✔️             |


### Supported platforms

|Architecture| Android | iOS     | Windows    | macOS | linux | HarmonyOS |
|------------|---------|---------|------------|-------|-------|-----------|
|   x64      |  ✔️      |         |   ✔️      | ✔️    |  ✔️    |   ✔️   |
|   x86      |  ✔️      |         |   ✔️      |       |        |        |
|   arm64    |  ✔️      | ✔️      |   ✔️      | ✔️    |  ✔️    |   ✔️   |
|   arm32    |  ✔️      |         |           |       |  ✔️    |   ✔️   |
|   riscv64  |          |         |           |       |  ✔️    |        |

### Supported programming languages

| 1. C++ | 2. C  | 3. Python | 4. JavaScript |
|--------|-------|-----------|---------------|
|   ✔️    | ✔️     | ✔️         |    ✔️          |

|5. Java | 6. C# | 7. Kotlin | 8. Swift |
|--------|-------|-----------|----------|
| ✔️      |  ✔️    | ✔️         |  ✔️       |

| 9. Go | 10. Dart | 11. Rust | 12. Pascal |
|-------|----------|----------|------------|
| ✔️     |  ✔️       |   ✔️      |    ✔️       |

For Rust support, please see [sherpa-rs][sherpa-rs]

It also supports WebAssembly.

### Supported NPUs

| [1. Rockchip NPU (RKNN)][rknpu-doc] | [2. Qualcomm NPU (QNN)][qnn-doc]  | [3. Ascend NPU][ascend-doc] |
|-------------------------------------|-----------------------------------|-----------------------------|
|     ✔️                              |                  ✔️               |     ✔️                      |

[Join our discord](https://discord.gg/fJdxzg2VbG)


## Introduction

This repository supports running the following functions **locally**

  - Speech-to-text (i.e., ASR); both streaming and non-streaming are supported
  - Text-to-speech (i.e., TTS)
  - Speaker diarization
  - Speaker identification
  - Speaker verification
  - Spoken language identification
  - Audio tagging
  - VAD (e.g., [silero-vad][silero-vad])
  - Speech enhancement (e.g., [gtcrn][gtcrn])
  - Keyword spotting
  - Source separation (e.g., [spleeter][spleeter], [UVR][UVR])

on the following platforms and operating systems:

  - x86, ``x86_64``, 32-bit ARM, 64-bit ARM (arm64, aarch64), RISC-V (riscv64), **RK NPU**, **Ascend NPU**
  - Linux, macOS, Windows, openKylin
  - Android, WearOS
  - iOS
  - HarmonyOS
  - NodeJS
  - WebAssembly
  - [NVIDIA Jetson Orin NX][NVIDIA Jetson Orin NX] (Support running on both CPU and GPU)
  - [NVIDIA Jetson Nano B01][NVIDIA Jetson Nano B01] (Support running on both CPU and GPU)
  - [Raspberry Pi][Raspberry Pi]
  - [RV1126][RV1126]
  - [LicheePi4A][LicheePi4A]
  - [VisionFive 2][VisionFive 2]
  - [旭日X3派][旭日X3派]
  - [爱芯派][爱芯派]
  - [RK3588][RK3588]
  - etc

with the following APIs

  - C++, C, Python, Go, ``C#``
  - Java, Kotlin, JavaScript
  - Swift, Rust
  - Dart, Object Pascal

### Links for Huggingface Spaces

<details>
<summary>You can visit the following Huggingface spaces to try sherpa-onnx without
installing anything. All you need is a browser.</summary>

| Description                                           | URL                                     | 中国镜像                               |
|-------------------------------------------------------|-----------------------------------------|----------------------------------------|
| Speaker diarization                                   | [Click me][hf-space-speaker-diarization]| [镜像][hf-space-speaker-diarization-cn]|
| Speech recognition                                    | [Click me][hf-space-asr]                | [镜像][hf-space-asr-cn]                |
| Speech recognition with [Whisper][Whisper]            | [Click me][hf-space-asr-whisper]        | [镜像][hf-space-asr-whisper-cn]        |
| Speech synthesis                                      | [Click me][hf-space-tts]                | [镜像][hf-space-tts-cn]                |
| Generate subtitles                                    | [Click me][hf-space-subtitle]           | [镜像][hf-space-subtitle-cn]           |
| Audio tagging                                         | [Click me][hf-space-audio-tagging]      | [镜像][hf-space-audio-tagging-cn]      |
| Source separation                                     | [Click me][hf-space-source-separation]  | [镜像][hf-space-source-separation-cn]  |
| Spoken language identification with [Whisper][Whisper]| [Click me][hf-space-slid-whisper]       | [镜像][hf-space-slid-whisper-cn]       |

We also have spaces built using WebAssembly. They are listed below:

| Description                                                                              | Huggingface space| ModelScope space|
|------------------------------------------------------------------------------------------|------------------|-----------------|
|Voice activity detection with [silero-vad][silero-vad]                                    | [Click me][wasm-hf-vad]|[地址][wasm-ms-vad]|
|Real-time speech recognition (Chinese + English) with Zipformer                           | [Click me][wasm-hf-streaming-asr-zh-en-zipformer]|[地址][wasm-hf-streaming-asr-zh-en-zipformer]|
|Real-time speech recognition (Chinese + English) with Paraformer                          |[Click me][wasm-hf-streaming-asr-zh-en-paraformer]| [地址][wasm-ms-streaming-asr-zh-en-paraformer]|
|Real-time speech recognition (Chinese + English + Cantonese) with [Paraformer-large][Paraformer-large]|[Click me][wasm-hf-streaming-asr-zh-en-yue-paraformer]| [地址][wasm-ms-streaming-asr-zh-en-yue-paraformer]|
|Real-time speech recognition (English) |[Click me][wasm-hf-streaming-asr-en-zipformer]    |[地址][wasm-ms-streaming-asr-en-zipformer]|
|VAD + speech recognition (Chinese) with [Zipformer CTC](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/icefall/zipformer.html#sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03-chinese)|[Click me][wasm-hf-vad-asr-zh-zipformer-ctc-07-03]| [地址][wasm-ms-vad-asr-zh-zipformer-ctc-07-03]|
|VAD + speech recognition (Chinese + English + Korean + Japanese + Cantonese) with [SenseVoice][SenseVoice]|[Click me][wasm-hf-vad-asr-zh-en-ko-ja-yue-sense-voice]| [地址][wasm-ms-vad-asr-zh-en-ko-ja-yue-sense-voice]|
|VAD + speech recognition (English) with [Whisper][Whisper] tiny.en|[Click me][wasm-hf-vad-asr-en-whisper-tiny-en]| [地址][wasm-ms-vad-asr-en-whisper-tiny-en]|
|VAD + speech recognition (English) with [Moonshine tiny][Moonshine tiny]|[Click me][wasm-hf-vad-asr-en-moonshine-tiny-en]| [地址][wasm-ms-vad-asr-en-moonshine-tiny-en]|
|VAD + speech recognition (English) with Zipformer trained with [GigaSpeech][GigaSpeech]    |[Click me][wasm-hf-vad-asr-en-zipformer-gigaspeech]| [地址][wasm-ms-vad-asr-en-zipformer-gigaspeech]|
|VAD + speech recognition (Chinese) with Zipformer trained with [WenetSpeech][WenetSpeech]  |[Click me][wasm-hf-vad-asr-zh-zipformer-wenetspeech]| [地址][wasm-ms-vad-asr-zh-zipformer-wenetspeech]|
|VAD + speech recognition (Japanese) with Zipformer trained with [ReazonSpeech][ReazonSpeech]|[Click me][wasm-hf-vad-asr-ja-zipformer-reazonspeech]| [地址][wasm-ms-vad-asr-ja-zipformer-reazonspeech]|
|VAD + speech recognition (Thai) with Zipformer trained with [GigaSpeech2][GigaSpeech2]      |[Click me][wasm-hf-vad-asr-th-zipformer-gigaspeech2]| [地址][wasm-ms-vad-asr-th-zipformer-gigaspeech2]|
|VAD + speech recognition (Chinese 多种方言) with a [TeleSpeech-ASR][TeleSpeech-ASR] CTC model|[Click me][wasm-hf-vad-asr-zh-telespeech]| [地址][wasm-ms-vad-asr-zh-telespeech]|
|VAD + speech recognition (English + Chinese, 及多种中文方言) with Paraformer-large          |[Click me][wasm-hf-vad-asr-zh-en-paraformer-large]| [地址][wasm-ms-vad-asr-zh-en-paraformer-large]|
|VAD + speech recognition (English + Chinese, 及多种中文方言) with Paraformer-small          |[Click me][wasm-hf-vad-asr-zh-en-paraformer-small]| [地址][wasm-ms-vad-asr-zh-en-paraformer-small]|
|VAD + speech recognition (多语种及多种中文方言) with [Dolphin][Dolphin]-base          |[Click me][wasm-hf-vad-asr-multi-lang-dolphin-base]| [地址][wasm-ms-vad-asr-multi-lang-dolphin-base]|
|Speech synthesis (English)                                                                  |[Click me][wasm-hf-tts-piper-en]| [地址][wasm-ms-tts-piper-en]|
|Speech synthesis (German)                                                                   |[Click me][wasm-hf-tts-piper-de]| [地址][wasm-ms-tts-piper-de]|
|Speaker diarization                                                                         |[Click me][wasm-hf-speaker-diarization]|[地址][wasm-ms-speaker-diarization]|

</details>

### Links for pre-built Android APKs

<details>

<summary>You can find pre-built Android APKs for this repository in the following table</summary>

| Description                            | URL                                | 中国用户                          |
|----------------------------------------|------------------------------------|-----------------------------------|
| Speaker diarization                    | [Address][apk-speaker-diarization] | [点此][apk-speaker-diarization-cn]|
| Streaming speech recognition           | [Address][apk-streaming-asr]       | [点此][apk-streaming-asr-cn]      |
| Simulated-streaming speech recognition | [Address][apk-simula-streaming-asr]| [点此][apk-simula-streaming-asr-cn]|
| Text-to-speech                         | [Address][apk-tts]                 | [点此][apk-tts-cn]                |
| Voice activity detection (VAD)         | [Address][apk-vad]                 | [点此][apk-vad-cn]                |
| VAD + non-streaming speech recognition | [Address][apk-vad-asr]             | [点此][apk-vad-asr-cn]            |
| Two-pass speech recognition            | [Address][apk-2pass]               | [点此][apk-2pass-cn]              |
| Audio tagging                          | [Address][apk-at]                  | [点此][apk-at-cn]                 |
| Audio tagging (WearOS)                 | [Address][apk-at-wearos]           | [点此][apk-at-wearos-cn]          |
| Speaker identification                 | [Address][apk-sid]                 | [点此][apk-sid-cn]                |
| Spoken language identification         | [Address][apk-slid]                | [点此][apk-slid-cn]               |
| Keyword spotting                       | [Address][apk-kws]                 | [点此][apk-kws-cn]                |

</details>

### Links for pre-built Flutter APPs

<details>

#### Real-time speech recognition

| Description                    | URL                                 | 中国用户                            |
|--------------------------------|-------------------------------------|-------------------------------------|
| Streaming speech recognition   | [Address][apk-flutter-streaming-asr]| [点此][apk-flutter-streaming-asr-cn]|

#### Text-to-speech

| Description                              | URL                                | 中国用户                           |
|------------------------------------------|------------------------------------|------------------------------------|
| Android (arm64-v8a, armeabi-v7a, x86_64) | [Address][flutter-tts-android]     | [点此][flutter-tts-android-cn]     |
| Linux (x64)                              | [Address][flutter-tts-linux]       | [点此][flutter-tts-linux-cn]       |
| macOS (x64)                              | [Address][flutter-tts-macos-x64]   | [点此][flutter-tts-macos-x64-cn] |
| macOS (arm64)                            | [Address][flutter-tts-macos-arm64] | [点此][flutter-tts-macos-arm64-cn]   |
| Windows (x64)                            | [Address][flutter-tts-win-x64]     | [点此][flutter-tts-win-x64-cn]     |

## Demos

### ▶️ Android App
Run speech recognition **natively on your phone** using ONNX Runtime.

### 🌐 Browser (WASM)
Experience transcription **directly in your browser**, no server required.
- [Hugging Face Spaces Demo](https://huggingface.co/spaces/Banafo/Kroko-Streaming-ASR-Wasm)
## Documentation

Full documentation could be found [here](https://docs.kroko.ai/on-premise/#)

## Our Community

Join the Kroko community to learn, share, and contribute:

- 💬 **[Discord](https://discord.gg/JT7wdtnK79)** – chat with developers, ask questions, and share projects.  
- 📢 **[Reddit](https://www.reddit.com/r/kroko_ai/)** – join discussions, showcase your integrations, and follow updates.
- 🤗 **[Hugging Face](https://huggingface.co/Banafo/Kroko-ASR)** – explore our models, try live demos, and contribute feedback.

---

## Table of Contents

1. [Building `kroko-onnx`](#1-building-kroko-onnx)  
   1.1 [Linux (x64 or arm64)](#linux-x64-or-arm64)  
   1.2 [Docker](#docker)  
   1.3 [Python](#python)  

2. [Usage Examples (WebSocket Server)](#2-usage-examples-websocket-server)  
   2.1 [WebSocket Server Format](#websocket-server-format)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.1.1 [Input](#input)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.1.2 [Output](#output)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.1.3 [Output Fields](#output-fields)  

3. [Using `kroko-onnx` from Python](#3-using-kroko-onnx-from-python)  
   3.1 [Import and Create a Recognizer](#import-and-create-a-recognizer)  
   3.2 [Parameter Reference](#parameter-reference)  
   3.3 [Running the Recognizer on Audio Files](#running-the-recognizer-on-audio-files)

---

## 1. Building `kroko-onnx`

### Linux (x64 or arm64)

```bash
git clone https://github.com/orgs/kroko-ai/kroko-onnx
cd kroko-onnx
mkdir build
cd build

# By default, it builds static libraries and uses static link and works only with Kroko free models
cmake -DCMAKE_BUILD_TYPE=Release ..

# To build it with an option to use Kroko Pro models
cmake -DCMAKE_BUILD_TYPE=Release -DKROKO_LICENSE=ON ..

make -j6
```

> ⚠️ **IMPORTANT:** If you build with the license option enabled (`-DKROKO_LICENSE=ON`), and later want to switch back to a license-free build,  
> you **must delete the `build/` directory** first, or explicitly rerun `cmake` with `-DKROKO_LICENSE=OFF` to clear the CMake cache.  
> Otherwise, the license configuration may persist in the build.

After building, you will find the executable `kroko-onnx-online-websocket-server` inside the `bin` directory.

> For GPU builds, refer to:  
> [Sherpa-ONNX GPU Install Guide](https://k2-fsa.github.io/sherpa/onnx/install/linux.html)

---

### Docker

```bash
git clone https://github.com/kroko-ai/kroko-onnx.git
cd kroko-onnx

# For Kroko free models
docker build -t kroko-onnx .

# For Kroko Pro models
docker build -t kroko-onnx --build-arg KROKO_LICENSE=ON .
```

After building, you will find the executable `kroko-onnx-online-websocket-server` and the `kroko-onnx` Python package installed.

---

### Python

```bash
git clone https://github.com/kroko-ai/kroko-onnx
cd kroko-onnx

# For Kroko free models
pip install .

# For Kroko Pro models
KROKO_LICENSE=ON pip install .
```

After installation, you can use the `kroko-onnx` Python package.

> 🛠️ Windows and macOS build instructions coming soon!

---

## 2. Usage Examples (WebSocket Server)

```bash
./kroko-onnx-online-websocket-server --key=LICENSE_KEY --model=/path/to/model.data
```

Starts the server listening on the **default port (6006)**.

```bash
./kroko-onnx-online-websocket-server --key=LICENSE_KEY --port=6007 --model=/path/to/model.data
```

Starts the server listening on a **specified port**.

```bash
./kroko-onnx-online-websocket-server --help
```

Shows the full list of parameters.

---

### WebSocket Server Format

#### Input

- The samples should be **16kHz**, **single channel**, and **16-bit**.
- The WebSocket connection accepts a buffer in the following format:
  - `data`: float32 buffer

##### Python Example: Convert Audio to Float32 Buffer

```python
samples = f.readframes(num_samples)
samples_int16 = np.frombuffer(samples, dtype=np.int16)
samples_float32 = samples_int16.astype(np.float32)
buf = samples_float32.tobytes()
```

---

#### Output

The result is in **JSON** format:

```json
{
  "type": "partial",
  "text": "Text from the current segment",
  "segment": 0,
  "startedAt": 0.0,
  "elements": {
    "segments": [
      {
        "type": "segment",
        "text": "",
        "startedAt": 0.0,
        "segment": 0
      }
    ],
    "words": [
      {
        "type": "word",
        "text": "",
        "startedAt": 0.0,
        "segment": 0
      }
    ]
  }
}
```

---

#### Output Fields

Each section contains the following elements:

##### `type` – The type of the element:

- `final` – the full text of the decoded segment  
- `partial` – the text of a not-yet-finished segment  
- `segment` – part of the transcript, same as the text in the main segment (for Banafo Online).  
- `word` – individual word

##### `text`

The transcript of the segment or individual word.

##### `startedAt`

The timestamp (in seconds, float value) indicating the beginning of the element.  
> Example: `1.42` = 1 second and 420 milliseconds

##### `elements`

Contains:

- `segments`: array of segment objects  
- `words`: array of word objects

---

## 3. Using `kroko-onnx` from Python

### Import and Create a Recognizer

```python
import kroko_onnx

recognizer = kroko_onnx.OnlineRecognizer.from_transducer(
    model_path="path/to/model",
    key="",
    referralcode="",
    num_threads=1,
    provider="cpu",
    sample_rate=16000,
    decoding_method="modified_beam_search",
    blank_penalty=0.0,
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=2.4,
    rule2_min_trailing_silence=1.2,
    rule3_min_utterance_length=20.0,
)
```

> ⚠️ Only `model_path` is required. All other parameters are optional.

---

### Parameter Reference

| Argument                     | Type     | Default     | Description |
|-----------------------------|----------|-------------|-------------|
| `model_path`                | `str`    | **Required** | Path to the Kroko model file. |
| `key`                       | `str`    | `""`        | License key. Required only for **Pro models**. |
| `referralcode`              | `str`    | `""`        | Optional project referral code. Contact Kroko for revenue sharing options. |
| `num_threads`               | `int`    | `1`         | Number of threads used for neural network computation. |
| `provider`                  | `str`    | `"cpu"`     | Execution provider. Valid values: `cpu`, `cuda`, `coreml`. |
| `sample_rate`               | `int`    | `16000`     | Sample rate of the input audio. Resampling is performed if it differs. |
| `decoding_method`           | `str`    | `"modified_beam_search"` | Valid values: `greedy_search`, `modified_beam_search`. |
| `blank_penalty`             | `float`  | `0.0`       | Penalty applied to the blank symbol during decoding (applied as: `logits[:, 0] -= blank_penalty`). |
| `enable_endpoint_detection`| `bool`   | `True`      | Enables endpoint detection using rule-based logic. |
| `rule1_min_trailing_silence`| `float` | `2.4`       | Rule 1: Minimum trailing silence (in seconds) to trigger endpoint. |
| `rule2_min_trailing_silence`| `float` | `1.2`       | Rule 2: Minimum trailing silence (in seconds) to trigger endpoint. |
| `rule3_min_utterance_length`| `float` | `20.0`      | Rule 3: Minimum utterance length (in seconds) to trigger endpoint. |

---

### Running the Recognizer on Audio Files

Below is a complete example of how to use the recognizer to transcribe one or more `.wav` files:

```python
import numpy as np
from kroko_onnx.utils import read_wave, assert_file_exists

streams = []
total_duration = 0

for wave_filename in args.sound_files:
    assert_file_exists(wave_filename)

    samples, sample_rate = read_wave(wave_filename)
    duration = len(samples) / sample_rate
    total_duration += duration

    # Create a new stream for this audio
    s = recognizer.create_stream()

    # Send waveform data
    s.accept_waveform(sample_rate, samples)

    # Add 0.66 seconds of padding silence
    tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
    s.accept_waveform(sample_rate, tail_paddings)

    s.input_finished()
    streams.append(s)

# Decode all ready streams in parallel
while True:
    ready_list = [s for s in streams if recognizer.is_ready(s)]
    if not ready_list:
        break
    recognizer.decode_streams(ready_list)

# Collect results
results = [recognizer.get_result(s) for s in streams]

# Print transcriptions
for i, result in enumerate(results):
    print(f"{args.sound_files[i]}: {result.text}")
```

---

> 🔁 You can process multiple files at once using this pattern.  
> 📎 Each stream corresponds to one audio file.
### [Go WebSocket Server](https://github.com/bbeyondllove/asr_server)

It provides a WebSocket server based on the Go programming language for sherpa-onnx.

### [Making robot Paimon, Ep10 "The AI Part 1"](https://www.youtube.com/watch?v=KxPKkwxGWZs)

It is a [YouTube video](https://www.youtube.com/watch?v=KxPKkwxGWZs),
showing how the author tried to use AI so he can have a conversation with Paimon.

It uses sherpa-onnx for speech-to-text and text-to-speech.
|1|
|---|
|![](https://github.com/user-attachments/assets/f6eea2d5-1807-42cb-9160-be8da2971e1f)|

### [TtsReader - Desktop application](https://github.com/ys-pro-duction/TtsReader)

A desktop text-to-speech application built using Kotlin Multiplatform.

### [MentraOS](https://github.com/Mentra-Community/MentraOS)

> Smart glasses OS, with dozens of built-in apps. Users get AI assistant, notifications,
> translation, screen mirror, captions, and more. Devs get to write 1 app that runs on
> any pair of smart glasses.

It uses sherpa-onnx for real-time speech recognition on iOS and Android devices.
See also <https://github.com/Mentra-Community/MentraOS/pull/861>

It uses Swift for iOS and Java for Android.

### [flet_sherpa_onnx](https://github.com/SamYuan1990/flet_sherpa_onnx)

Flet ASR/STT component based on sherpa-onnx.
Example [a chat box agent](https://github.com/SamYuan1990/i18n-agent-action)

### [elderly-companion](https://github.com/SearocIsMe/elderly-companion)

It uses sherpa-onnx's Python API for real-time speech recognition in ROS2 with RK NPU.

### [achatbot-go](https://github.com/ai-bot-pro/achatbot-go)

a multimodal chatbot based on go with sherpa-onnx's speech lib api.

[sherpa-rs]: https://github.com/thewh1teagle/sherpa-rs
[silero-vad]: https://github.com/snakers4/silero-vad
[Raspberry Pi]: https://www.raspberrypi.com/
[RV1126]: https://www.rock-chips.com/uploads/pdf/2022.8.26/191/RV1126%20Brief%20Datasheet.pdf
[LicheePi4A]: https://sipeed.com/licheepi4a
[VisionFive 2]: https://www.starfivetech.com/en/site/boards
[旭日X3派]: https://developer.horizon.ai/api/v1/fileData/documents_pi/index.html
[爱芯派]: https://wiki.sipeed.com/hardware/zh/maixIII/ax-pi/axpi.html
[hf-space-speaker-diarization]: https://huggingface.co/spaces/k2-fsa/speaker-diarization
[hf-space-speaker-diarization-cn]: https://hf.qhduan.com/spaces/k2-fsa/speaker-diarization
[hf-space-asr]: https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition
[hf-space-asr-cn]: https://hf.qhduan.com/spaces/k2-fsa/automatic-speech-recognition
[Whisper]: https://github.com/openai/whisper
[hf-space-asr-whisper]: https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition-with-whisper
[hf-space-asr-whisper-cn]: https://hf.qhduan.com/spaces/k2-fsa/automatic-speech-recognition-with-whisper
[hf-space-tts]: https://huggingface.co/spaces/k2-fsa/text-to-speech
[hf-space-tts-cn]: https://hf.qhduan.com/spaces/k2-fsa/text-to-speech
[hf-space-subtitle]: https://huggingface.co/spaces/k2-fsa/generate-subtitles-for-videos
[hf-space-subtitle-cn]: https://hf.qhduan.com/spaces/k2-fsa/generate-subtitles-for-videos
[hf-space-audio-tagging]: https://huggingface.co/spaces/k2-fsa/audio-tagging
[hf-space-audio-tagging-cn]: https://hf.qhduan.com/spaces/k2-fsa/audio-tagging
[hf-space-source-separation]: https://huggingface.co/spaces/k2-fsa/source-separation
[hf-space-source-separation-cn]: https://hf.qhduan.com/spaces/k2-fsa/source-separation
[hf-space-slid-whisper]: https://huggingface.co/spaces/k2-fsa/spoken-language-identification
[hf-space-slid-whisper-cn]: https://hf.qhduan.com/spaces/k2-fsa/spoken-language-identification
[wasm-hf-vad]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-sherpa-onnx
[wasm-ms-vad]: https://modelscope.cn/studios/csukuangfj/web-assembly-vad-sherpa-onnx
[wasm-hf-streaming-asr-zh-en-zipformer]: https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-zh-en
[wasm-ms-streaming-asr-zh-en-zipformer]: https://modelscope.cn/studios/k2-fsa/web-assembly-asr-sherpa-onnx-zh-en
[wasm-hf-streaming-asr-zh-en-paraformer]: https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-zh-en-paraformer
[wasm-ms-streaming-asr-zh-en-paraformer]: https://modelscope.cn/studios/k2-fsa/web-assembly-asr-sherpa-onnx-zh-en-paraformer
[Paraformer-large]: https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
[wasm-hf-streaming-asr-zh-en-yue-paraformer]: https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-zh-cantonese-en-paraformer
[wasm-ms-streaming-asr-zh-en-yue-paraformer]: https://modelscope.cn/studios/k2-fsa/web-assembly-asr-sherpa-onnx-zh-cantonese-en-paraformer
[wasm-hf-streaming-asr-en-zipformer]: https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-en
[wasm-ms-streaming-asr-en-zipformer]: https://modelscope.cn/studios/k2-fsa/web-assembly-asr-sherpa-onnx-en
[SenseVoice]: https://github.com/FunAudioLLM/SenseVoice
[wasm-hf-vad-asr-zh-zipformer-ctc-07-03]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-zipformer-ctc
[wasm-ms-vad-asr-zh-zipformer-ctc-07-03]: https://modelscope.cn/studios/csukuangfj/web-assembly-vad-asr-sherpa-onnx-zh-zipformer-ctc/summary
[wasm-hf-vad-asr-zh-en-ko-ja-yue-sense-voice]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-ja-ko-cantonese-sense-voice
[wasm-ms-vad-asr-zh-en-ko-ja-yue-sense-voice]: https://www.modelscope.cn/studios/csukuangfj/web-assembly-vad-asr-sherpa-onnx-zh-en-jp-ko-cantonese-sense-voice
[wasm-hf-vad-asr-en-whisper-tiny-en]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-en-whisper-tiny
[wasm-ms-vad-asr-en-whisper-tiny-en]: https://www.modelscope.cn/studios/csukuangfj/web-assembly-vad-asr-sherpa-onnx-en-whisper-tiny
[wasm-hf-vad-asr-en-moonshine-tiny-en]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-en-moonshine-tiny
[wasm-ms-vad-asr-en-moonshine-tiny-en]: https://www.modelscope.cn/studios/csukuangfj/web-assembly-vad-asr-sherpa-onnx-en-moonshine-tiny
[wasm-hf-vad-asr-en-zipformer-gigaspeech]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-en-zipformer-gigaspeech
[wasm-ms-vad-asr-en-zipformer-gigaspeech]: https://www.modelscope.cn/studios/k2-fsa/web-assembly-vad-asr-sherpa-onnx-en-zipformer-gigaspeech
[wasm-hf-vad-asr-zh-zipformer-wenetspeech]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-zipformer-wenetspeech
[wasm-ms-vad-asr-zh-zipformer-wenetspeech]: https://www.modelscope.cn/studios/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-zipformer-wenetspeech
[reazonspeech]: https://research.reazon.jp/_static/reazonspeech_nlp2023.pdf
[wasm-hf-vad-asr-ja-zipformer-reazonspeech]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-ja-zipformer
[wasm-ms-vad-asr-ja-zipformer-reazonspeech]: https://www.modelscope.cn/studios/csukuangfj/web-assembly-vad-asr-sherpa-onnx-ja-zipformer
[gigaspeech2]: https://github.com/speechcolab/gigaspeech2
[wasm-hf-vad-asr-th-zipformer-gigaspeech2]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-th-zipformer
[wasm-ms-vad-asr-th-zipformer-gigaspeech2]: https://www.modelscope.cn/studios/csukuangfj/web-assembly-vad-asr-sherpa-onnx-th-zipformer
[telespeech-asr]: https://github.com/tele-ai/telespeech-asr
[wasm-hf-vad-asr-zh-telespeech]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-telespeech
[wasm-ms-vad-asr-zh-telespeech]: https://www.modelscope.cn/studios/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-telespeech
[wasm-hf-vad-asr-zh-en-paraformer-large]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-paraformer
[wasm-ms-vad-asr-zh-en-paraformer-large]: https://www.modelscope.cn/studios/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-paraformer
[wasm-hf-vad-asr-zh-en-paraformer-small]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-paraformer-small
[wasm-ms-vad-asr-zh-en-paraformer-small]: https://www.modelscope.cn/studios/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-paraformer-small
[dolphin]: https://github.com/dataoceanai/dolphin
[wasm-ms-vad-asr-multi-lang-dolphin-base]: https://modelscope.cn/studios/csukuangfj/web-assembly-vad-asr-sherpa-onnx-multi-lang-dophin-ctc
[wasm-hf-vad-asr-multi-lang-dolphin-base]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-multi-lang-dophin-ctc

[wasm-hf-tts-piper-en]: https://huggingface.co/spaces/k2-fsa/web-assembly-tts-sherpa-onnx-en
[wasm-ms-tts-piper-en]: https://modelscope.cn/studios/k2-fsa/web-assembly-tts-sherpa-onnx-en
[wasm-hf-tts-piper-de]: https://huggingface.co/spaces/k2-fsa/web-assembly-tts-sherpa-onnx-de
[wasm-ms-tts-piper-de]: https://modelscope.cn/studios/k2-fsa/web-assembly-tts-sherpa-onnx-de
[wasm-hf-speaker-diarization]: https://huggingface.co/spaces/k2-fsa/web-assembly-speaker-diarization-sherpa-onnx
[wasm-ms-speaker-diarization]: https://www.modelscope.cn/studios/csukuangfj/web-assembly-speaker-diarization-sherpa-onnx
[apk-speaker-diarization]: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/apk.html
[apk-speaker-diarization-cn]: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/apk-cn.html
[apk-streaming-asr]: https://k2-fsa.github.io/sherpa/onnx/android/apk.html
[apk-streaming-asr-cn]: https://k2-fsa.github.io/sherpa/onnx/android/apk-cn.html
[apk-simula-streaming-asr]: https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr.html
[apk-simula-streaming-asr-cn]: https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr-cn.html
[apk-tts]: https://k2-fsa.github.io/sherpa/onnx/tts/apk-engine.html
[apk-tts-cn]: https://k2-fsa.github.io/sherpa/onnx/tts/apk-engine-cn.html
[apk-vad]: https://k2-fsa.github.io/sherpa/onnx/vad/apk.html
[apk-vad-cn]: https://k2-fsa.github.io/sherpa/onnx/vad/apk-cn.html
[apk-vad-asr]: https://k2-fsa.github.io/sherpa/onnx/vad/apk-asr.html
[apk-vad-asr-cn]: https://k2-fsa.github.io/sherpa/onnx/vad/apk-asr-cn.html
[apk-2pass]: https://k2-fsa.github.io/sherpa/onnx/android/apk-2pass.html
[apk-2pass-cn]: https://k2-fsa.github.io/sherpa/onnx/android/apk-2pass-cn.html
[apk-at]: https://k2-fsa.github.io/sherpa/onnx/audio-tagging/apk.html
[apk-at-cn]: https://k2-fsa.github.io/sherpa/onnx/audio-tagging/apk-cn.html
[apk-at-wearos]: https://k2-fsa.github.io/sherpa/onnx/audio-tagging/apk-wearos.html
[apk-at-wearos-cn]: https://k2-fsa.github.io/sherpa/onnx/audio-tagging/apk-wearos-cn.html
[apk-sid]: https://k2-fsa.github.io/sherpa/onnx/speaker-identification/apk.html
[apk-sid-cn]: https://k2-fsa.github.io/sherpa/onnx/speaker-identification/apk-cn.html
[apk-slid]: https://k2-fsa.github.io/sherpa/onnx/spoken-language-identification/apk.html
[apk-slid-cn]: https://k2-fsa.github.io/sherpa/onnx/spoken-language-identification/apk-cn.html
[apk-kws]: https://k2-fsa.github.io/sherpa/onnx/kws/apk.html
[apk-kws-cn]: https://k2-fsa.github.io/sherpa/onnx/kws/apk-cn.html
[apk-flutter-streaming-asr]: https://k2-fsa.github.io/sherpa/onnx/flutter/asr/app.html
[apk-flutter-streaming-asr-cn]: https://k2-fsa.github.io/sherpa/onnx/flutter/asr/app-cn.html
[flutter-tts-android]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-android.html
[flutter-tts-android-cn]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-android-cn.html
[flutter-tts-linux]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-linux.html
[flutter-tts-linux-cn]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-linux-cn.html
[flutter-tts-macos-x64]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-macos-x64.html
[flutter-tts-macos-arm64-cn]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-macos-arm64-cn.html
[flutter-tts-macos-arm64]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-macos-arm64.html
[flutter-tts-macos-x64-cn]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-macos-x64-cn.html
[flutter-tts-win-x64]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-win.html
[flutter-tts-win-x64-cn]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-win-cn.html
[lazarus-subtitle]: https://k2-fsa.github.io/sherpa/onnx/lazarus/download-generated-subtitles.html
[lazarus-subtitle-cn]: https://k2-fsa.github.io/sherpa/onnx/lazarus/download-generated-subtitles-cn.html
[asr-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
[tts-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
[vad-models]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
[kws-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
[at-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models
[sid-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
[slid-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
[punct-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models
[speaker-segmentation-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
[GigaSpeech]: https://github.com/SpeechColab/GigaSpeech
[WenetSpeech]: https://github.com/wenet-e2e/WenetSpeech
[sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
[sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16.tar.bz2
[sherpa-onnx-streaming-zipformer-korean-2024-06-16]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2
[sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2
[sherpa-onnx-streaming-zipformer-en-20M-2023-02-17]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
[sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2
[sherpa-onnx-zipformer-ru-2024-09-18]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ru-2024-09-18.tar.bz2
[sherpa-onnx-zipformer-korean-2024-06-24]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2
[sherpa-onnx-zipformer-thai-2024-06-20]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-thai-2024-06-20.tar.bz2
[sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24.tar.bz2
[sherpa-onnx-paraformer-zh-2024-03-09]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2
[sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24.tar.bz2
[sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
[sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
[sherpa-onnx-streaming-zipformer-fr-2023-04-14]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-fr-2023-04-14.tar.bz2
[Moonshine tiny]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
[NVIDIA Jetson Orin NX]: https://developer.download.nvidia.com/assets/embedded/secure/jetson/orin_nx/docs/Jetson_Orin_NX_DS-10712-001_v0.5.pdf?RCPGu9Q6OVAOv7a7vgtwc9-BLScXRIWq6cSLuditMALECJ_dOj27DgnqAPGVnT2VpiNpQan9SyFy-9zRykR58CokzbXwjSA7Gj819e91AXPrWkGZR3oS1VLxiDEpJa_Y0lr7UT-N4GnXtb8NlUkP4GkCkkF_FQivGPrAucCUywL481GH_WpP_p7ziHU1Wg==&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLmhrLyJ9
[NVIDIA Jetson Nano B01]: https://www.seeedstudio.com/blog/2020/01/16/new-revision-of-jetson-nano-dev-kit-now-supports-new-jetson-nano-module/
[speech-enhancement-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
[source-separation-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models
[RK3588]: https://www.rock-chips.com/uploads/pdf/2022.8.26/192/RK3588%20Brief%20Datasheet.pdf
[spleeter]: https://github.com/deezer/spleeter
[UVR]: https://github.com/Anjok07/ultimatevocalremovergui
[gtcrn]: https://github.com/Xiaobin-Rong/gtcrn
[tts-url]: https://k2-fsa.github.io/sherpa/onnx/tts/all-in-one.html
[ss-url]: https://k2-fsa.github.io/sherpa/onnx/source-separation/index.html
[sd-url]: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html
[slid-url]: https://k2-fsa.github.io/sherpa/onnx/spoken-language-identification/index.html
[at-url]: https://k2-fsa.github.io/sherpa/onnx/audio-tagging/index.html
[vad-url]: https://k2-fsa.github.io/sherpa/onnx/vad/index.html
[kws-url]: https://k2-fsa.github.io/sherpa/onnx/kws/index.html
[punct-url]: https://k2-fsa.github.io/sherpa/onnx/punctuation/index.html
[se-url]: https://k2-fsa.github.io/sherpa/onnx/speech-enhancement/index.html
[rknpu-doc]: https://k2-fsa.github.io/sherpa/onnx/rknn/index.html
[qnn-doc]: https://k2-fsa.github.io/sherpa/onnx/qnn/index.html
[ascend-doc]: https://k2-fsa.github.io/sherpa/onnx/ascend/index.html
