from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import queue
from pynput.keyboard import Key
from pynput import keyboard
import time
import torch

class SpeechToText():
    def __init__(self):
        self.rate = 16000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel("small", device=self.device, compute_type=self.compute_type)
        self.mic_samples = queue.Queue()
        self.space_pressed = False
        self.running = True

        self.stream = sd.InputStream(samplerate=self.rate, channels=1, dtype='float32', callback=self.callback)
        self.stream.start()

        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def callback(self, indata, frames, time, status):
        self.mic_samples.put(indata.copy())

    def on_press(self, key):
        if key == Key.space:
            if not self.space_pressed:
                self.space_pressed = True
                print("Listening...")
        if key == Key.esc:
            self.running = False
            return False

    def on_release(self, key):
        if key == Key.space:
            if self.space_pressed:
                self.space_pressed = False
                print("Stopped listening.")

    def get_input(self):
        buffer = []
        while self.running:
            if self.space_pressed:
                if not buffer:
                    while True:
                        try:
                            self.mic_samples.get_nowait()
                        except queue.Empty:
                            break
                try:
                    raw = self.mic_samples.get(timeout=0.1)
                    buffer.append(raw)
                except queue.Empty:
                    continue
            if not self.space_pressed and buffer:
                while True:
                    try:
                        buffer.append(self.mic_samples.get_nowait())
                    except queue.Empty:
                        break
                raw_audio = np.concatenate(buffer, axis=0).squeeze()
                segments, info = self.model.transcribe(raw_audio, language="en", vad_filter=True, beam_size=2)
                text = "".join(segment.text for segment in segments).strip()
                return text
            else:
                time.sleep(0.1)

    def shutdown(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
        self.listener.stop()