import runpod
import base64
import io
import soundfile as sf
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel


# Model loading and text processing
# -------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

model = Qwen3TTSModel.from_pretrained(MODEL_ID)
model = model.to("cuda")

def synthesize(text: str):
    output = model.generate_custom_voice(
        text=text,
        speaker="Ryan",
        language="English"
    )

    audio_list, sr = output
    waveform = audio_list[0]
    waveform = np.asarray(waveform, dtype=np.float32)

    # Flatten if needed
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    return waveform, sr

# Handler itself
# --------------------------------------------------------------
def handler(job):
    text = job["input"].get("text", "")

    waveform, sr = synthesize(text)
    waveform = np.asarray(waveform, dtype="float32")

    buffer = io.BytesIO()
    sf.write(buffer, waveform, sr, format="WAV")
    buffer.seek(0)

    audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "audio": audio_base64
    }

# Star the job
# ---------------------------------------------------
runpod.serverless.start({"handler": handler})