import threading

from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import queue
from pynput.keyboard import Key
from pynput import keyboard
import time

rate = 16000
# bounded queue to prevent unbounded memory growth
mic_samples = queue.Queue()
model = WhisperModel("small", device="cuda", compute_type="float16")
space_pressed = False
running = True

def callback(indata, frames, time, status):
    mic_samples.put(indata.copy())

stream = sd.InputStream(samplerate=rate, channels=1, dtype='float32', callback=callback)
stream.start()

def on_press(key):
    global space_pressed
    if key == Key.space:
        if not space_pressed:
            space_pressed = True
            print("Listening...")
    if key == Key.esc:
        global running
        running = False
        return False

def on_release(key):
    global space_pressed
    if key == Key.space:
        if space_pressed:
            space_pressed = False
            print("Stopped listening.")

def push_to_talk_loop():
    buffer = []
    while running:
        if space_pressed:
            if not buffer:
                while True:
                    try:
                        mic_samples.get_nowait()
                    except queue.Empty:
                        break
            try:
                raw = mic_samples.get(timeout=0.1)
                buffer.append(raw)
            except queue.Empty:
                continue
        if not space_pressed and buffer:
            while True:
                try:
                    buffer.append(mic_samples.get_nowait())
                except queue.Empty:
                    break
            raw_audio = np.concatenate(buffer, axis=0).squeeze()
            segments, info = model.transcribe(raw_audio, language="en", vad_filter=True, beam_size=2)
            text = "".join(segment.text for segment in segments).strip()
            print("You: ", text)
            buffer = []
        else:
            time.sleep(0.1)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
action_thread = threading.Thread(target=push_to_talk_loop, daemon=True)
action_thread.start()

# audio callback cleanup
try:
    listener.join()
    action_thread.join()
finally:
    stream.stop()
    stream.close()
