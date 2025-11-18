import multiprocessing
import argparse
import threading
import ssl
import time
import sys
import functools
import ctypes

from multiprocessing import Process, Manager, Value, Queue


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu',
                        action="store_true",
                        help='Run in CPU-only mode without TensorRT')
    parser.add_argument('--llm_provider',
                        type=str,
                        default='openai',
                        choices=['openai', 'hf'],
                        help='LLM provider for CPU mode: openai or hf (HuggingFace)')
    parser.add_argument('--hf_model',
                        type=str,
                        default='google/flan-t5-small',
                        help='HuggingFace model name for CPU mode')
    parser.add_argument('--whisper_tensorrt_path',
                        type=str,
                        default="/root/TensorRT-LLM/examples/whisper/whisper_small_en",
                        help='Whisper TensorRT model path')
    parser.add_argument('--mistral',
                        action="store_true",
                        help='Mistral')
    parser.add_argument('--mistral_tensorrt_path',
                        type=str,
                        default=None,
                        help='Mistral TensorRT model path')
    parser.add_argument('--mistral_tokenizer_path',
                        type=str,
                        default="teknium/OpenHermes-2.5-Mistral-7B",
                        help='Mistral TensorRT model path')
    parser.add_argument('--phi',
                        action="store_true",
                        help='Phi')
    parser.add_argument('--phi_tensorrt_path',
                        type=str,
                        default="/root/TensorRT-LLM/examples/phi/phi_engine",
                        help='Phi TensorRT model path')
    parser.add_argument('--phi_tokenizer_path',
                        type=str,
                        default="/root/TensorRT-LLM/examples/phi/phi-2",
                        help='Phi Tokenizer path')
    parser.add_argument('--phi_model_type',
                        type=str,
                        default=None,
                        help='Phi model type')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # CPU mode - use CPU adapters
    if args.cpu:
        from cpu_transcriber import CPUTranscriptionServer
        from cpu_llm import CPULLMEngine
        from cpu_tts import CPUTTS
        
        print(f"\n{'='*60}")
        print("Running WhisperFusion in CPU-only mode")
        print(f"LLM Provider: {args.llm_provider}")
        if args.llm_provider == 'hf':
            print(f"HuggingFace Model: {args.hf_model}")
        print(f"{'='*60}\n")
        
        multiprocessing.set_start_method('spawn')
        
        should_send_server_ready = Value(ctypes.c_bool, False)
        transcription_queue = Queue()
        llm_queue = Queue()
        audio_queue = Queue()
        
        # CPU Transcription process
        cpu_transcriber = CPUTranscriptionServer()
        whisper_process = multiprocessing.Process(
            target=cpu_transcriber.run,
            args=(
                "0.0.0.0",
                6006,
                transcription_queue,
                llm_queue,
                should_send_server_ready
            )
        )
        whisper_process.start()
        
        # CPU LLM process
        cpu_llm = CPULLMEngine(provider=args.llm_provider, hf_model=args.hf_model)
        llm_process = multiprocessing.Process(
            target=cpu_llm.run,
            args=(
                transcription_queue,
                llm_queue,
                audio_queue,
            )
        )
        llm_process.start()
        
        # CPU TTS process
        cpu_tts = CPUTTS()
        tts_process = multiprocessing.Process(
            target=cpu_tts.run,
            args=("0.0.0.0", 8888, audio_queue, should_send_server_ready)
        )
        tts_process.start()
        
        # Wait for all processes
        llm_process.join()
        whisper_process.join()
        tts_process.join()
    
    # GPU mode - use TensorRT (original behavior)
    else:
        from whisper_live.trt_server import TranscriptionServer
        from llm_service import TensorRTLLMEngine
        from tts_service import WhisperSpeechTTS
        
        if not args.whisper_tensorrt_path:
            raise ValueError("Please provide whisper_tensorrt_path to run the pipeline.")
            import sys
            sys.exit(0)
        
        if args.mistral:
            if not args.mistral_tensorrt_path or not args.mistral_tokenizer_path:
                raise ValueError("Please provide mistral_tensorrt_path and mistral_tokenizer_path to run the pipeline.")
                import sys
                sys.exit(0)

        if args.phi:
            if not args.phi_tensorrt_path or not args.phi_tokenizer_path:
                raise ValueError("Please provide phi_tensorrt_path and phi_tokenizer_path to run the pipeline.")
                import sys
                sys.exit(0)

        multiprocessing.set_start_method('spawn')
        
        lock = multiprocessing.Lock()
        
        manager = Manager()
        shared_output = manager.list()
        should_send_server_ready = Value(ctypes.c_bool, False)
        transcription_queue = Queue()
        llm_queue = Queue()
        audio_queue = Queue()


        whisper_server = TranscriptionServer()
        whisper_process = multiprocessing.Process(
            target=whisper_server.run,
            args=(
                "0.0.0.0",
                6006,
                transcription_queue,
                llm_queue,
                args.whisper_tensorrt_path,
                should_send_server_ready
            )
        )
        whisper_process.start()

        llm_provider = TensorRTLLMEngine()
        # llm_provider = MistralTensorRTLLMProvider()
        llm_process = multiprocessing.Process(
            target=llm_provider.run,
            args=(
                # args.mistral_tensorrt_path,
                # args.mistral_tokenizer_path,
                args.phi_tensorrt_path,
                args.phi_tokenizer_path,
                args.phi_model_type,
                transcription_queue,
                llm_queue,
                audio_queue,
            )
        )
        llm_process.start()

        # audio process
        tts_runner = WhisperSpeechTTS()
        tts_process = multiprocessing.Process(target=tts_runner.run, args=("0.0.0.0", 8888, audio_queue, should_send_server_ready))
        tts_process.start()

        llm_process.join()
        whisper_process.join()
        tts_process.join()
