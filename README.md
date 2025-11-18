# WhisperFusion

<h2 align="center">
  <a href="https://www.youtube.com/watch?v=_PnaP0AQJnk"><img
src="https://img.youtube.com/vi/_PnaP0AQJnk/0.jpg" style="background-color:rgba(0,0,0,0);" height=300 alt="WhisperFusion"></a>
  <br><br>Seamless conversations with AI (with ultra-low latency)<br><br>
</h2>

Welcome to WhisperFusion. WhisperFusion builds upon the capabilities of
the [WhisperLive](https://github.com/collabora/WhisperLive) and
[WhisperSpeech](https://github.com/collabora/WhisperSpeech) by
integrating Mistral, a Large Language Model (LLM), on top of the
real-time speech-to-text pipeline. Both LLM and
Whisper are optimized to run efficiently as TensorRT engines, maximizing
performance and real-time processing capabilities. While WhiperSpeech is 
optimized with torch.compile.

## Features

- **Real-Time Speech-to-Text**: Utilizes OpenAI WhisperLive to convert
  spoken language into text in real-time.

- **Large Language Model Integration**: Adds Mistral, a Large Language
  Model, to enhance the understanding and context of the transcribed
  text.

- **TensorRT Optimization**: Both LLM and Whisper are optimized to
  run as TensorRT engines, ensuring high-performance and low-latency
  processing.
- **torch.compile**: WhisperSpeech uses torch.compile to speed up 
  inference which makes PyTorch code run faster by JIT-compiling PyTorch
  code into optimized kernels.

## Hardware Requirements

- A GPU with at least 24GB of RAM
- For optimal latency, the GPU should have a similar FP16 (half) TFLOPS as the RTX 4090. Here are the [hardware specifications](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889) for the RTX 4090.

The demo was run on a single RTX 4090 GPU. WhisperFusion uses the Nvidia TensorRT-LLM library for CUDA optimized versions of popular LLM models. TensorRT-LLM supports multiple GPUs, so it should be possible to run WhisperFusion for even better performance on multiple GPUs.

## Getting Started
We provide a Docker Compose setup to streamline the deployment of the pre-built TensorRT-LLM docker container. This setup includes both Whisper and Phi converted to TensorRT engines, and the WhisperSpeech model is pre-downloaded to quickly start interacting with WhisperFusion. Additionally, we include a simple web server for the Web GUI.

- Build and Run with docker compose
```bash
mkdir docker/scratch-space
cp docker/scripts/build-* docker/scripts/run-whisperfusion.sh docker/scratch-space/

docker compose build
export MODEL=Phi-3-mini-4k-instruct    #Phi-3-mini-128k-instruct or phi-2, By default WhisperFusion uses phi-2
docker compose up
```

- Start Web GUI on `http://localhost:8000`

**NOTE**

## CPU-Only Demo Mode

For users without a GPU or TensorRT, WhisperFusion now includes a CPU-only fallback mode. This demo mode uses the same three-process architecture (transcription, LLM, TTS) but with CPU-compatible components.

### Features of CPU Mode:
- **File-based processing**: Drop WAV files into `input_audio/` directory
- **Flexible LLM options**: Use OpenAI API or local HuggingFace models
- **Text outputs**: Results written to `outputs/` directory
- **Optional audio**: Uses `espeak` for audio playback if available

### Quick Start (Ubuntu/Linux):

1. **Run the setup script**:
   ```bash
   bash run_cpu.sh
   ```

2. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Set up OpenAI API key (optional, for OpenAI LLM provider)**:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

4. **Run with OpenAI provider**:
   ```bash
   python main.py --cpu --llm_provider openai
   ```
   
   **OR run with HuggingFace local model**:
   ```bash
   python main.py --cpu --llm_provider hf
   ```

5. **Process audio files**:
   - Drop WAV files into the `input_audio/` directory
   - The pipeline will automatically process them
   - Check `outputs/` directory for results

### CPU Mode Options:
- `--cpu`: Enable CPU-only mode
- `--llm_provider`: Choose `openai` (requires API key) or `hf` (local HuggingFace model)
- `--hf_model`: Specify HuggingFace model (default: `google/flan-t5-small`)

### Notes:
- CPU mode is intended as a demo/test pipeline, not for real-time low-latency applications
- First run with HuggingFace models will download the model (may take time)
- For audio playback, install espeak: `sudo apt-get install espeak`
- OpenAI API usage will incur costs based on your usage

## Contact Us

For questions or issues, please open an issue. Contact us at:
marcus.edel@collabora.com, jpc@collabora.com,
vineet.suryan@collabora.com
