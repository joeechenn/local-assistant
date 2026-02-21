# AI Local Assistant

A locally-run AI assistant that can see, listen, and talk. Built from the ground up as a step toward creating intelligent agents that interact with the physical world.

## What Is This?

This is a multimodal AI system that runs entirely on local hardware. Everything happens on a single GPU. The idea is simple: build an assistant that can perceive its environment (vision, voice, face recognition) and respond intelligently.

Right now, it works like this: you hold spacebar, say something, and the system transcribes your speech and sends it to a locally-running LLM that responds. Behind the scenes, there's also a face recognition system that can identify who it's talking to. The groundwork for the assistant to behave differently depending on the person (personal assistant for me, general secretary for others).

## What's Implemented

**Speech-to-Text**: Faster Whisper running on GPU with a push-to-talk interface. Hold spacebar to speak, release to transcribe. The system uses an event-driven architecture with threaded audio capture and non-blocking queue management.

**Language Model**: Llama 3.2 11B Vision Instruct, quantized to 4-bit using BitsAndBytes (NF4 quantization with double quantization). This keeps a large model running on consumer VRAM while maintaining solid response quality. Full conversation history is maintained across turns.

**Face Recognition**: A custom Siamese CNN trained to distinguish between known and unknown faces. Uses shared-weight architecture with contrastive loss, operating in a learned embedding space. Trained on personal photos with varied conditions and supplemented with samples from the LFW dataset. Achieves commercial-grade accuracy on unseen test data.

## What's Next

- **Face-authenticated interactions**: using face detection as a gate to determine assistant behavior
- **RAG (Retrieval-Augmented Generation)**: personal knowledge base for storing preferences, conversation history, and context
- **QLoRA finetuning**: teaching the model behavioral patterns and communication styles
- **Vision integration**: leveraging the multimodal capabilities of Llama 3.2 for real-time visual understanding

## Tech Stack

Python, PyTorch, Faster Whisper, Hugging Face Transformers, BitsAndBytes, OpenCV, MediaPipe