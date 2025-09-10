from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import queue

rate = 16000
qsamples = queue.Queue()
model = WhisperModel("small", device="cpu", compute_type="int8")

def callback(indata, frames, time, status):
    qsamples.put(indata.copy())

stream = sd.InputStream(samplerate=rate, channels=1, dtype='float32', callback=callback)
stream.start()

def read_seconds(sec=2.0):
    listen = int(rate*sec)
    buffer = []
    heard = 0
    while heard < listen:
        raw = qsamples.get()
        buffer.append(raw)
        heard += len(raw)
    return np.concatenate(buffer, axis=0).squeeze()


def transcribe():
    raw_audio = read_seconds(2.0)
    segments, info = model.transcribe(raw_audio, language="en", vad_filter=True, beam_size=2)
    text = "".join(segment.text for segment in segments).strip()
    return text

print("Listening...")
try:
    while True:
        text = transcribe()
        if text:
            print("You:", text)
except KeyboardInterrupt:
    pass
finally:
    stream.stop()
    stream.close()