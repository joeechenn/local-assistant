from speech_to_text import SpeechToText
from llm import LLM

llm = LLM()
stt = SpeechToText()

try:
    while stt.running:
        text = stt.get_input()
        if text:
            print("You: ", text)
            response = llm.send_message(text)
            print("Assistant: ", response)
except KeyboardInterrupt:
    pass
finally:
    stt.shutdown()