import os
import time
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logging.warning("[CPU Transcriber] faster-whisper not available, falling back to whisper")

if not FASTER_WHISPER_AVAILABLE:
    try:
        import whisper
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False
        logging.error("[CPU Transcriber] Neither faster-whisper nor whisper available!")


class CPUTranscriptionServer:
    """
    CPU-based transcription server that watches a directory for audio files,
    transcribes them using faster-whisper or whisper, and enqueues the results.
    """
    
    def __init__(self):
        self.input_dir = Path("input_audio")
        self.processed_dir = self.input_dir / "processed"
        self.model = None
        
    def initialize_model(self, model_size="base"):
        """Initialize the whisper model (faster-whisper if available, else whisper)"""
        logging.info(f"[CPU Transcriber] Initializing model: {model_size}")
        
        if FASTER_WHISPER_AVAILABLE:
            # Use faster-whisper (more efficient on CPU)
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            self.use_faster_whisper = True
            logging.info("[CPU Transcriber] Using faster-whisper")
        elif WHISPER_AVAILABLE:
            # Fallback to standard whisper
            self.model = whisper.load_model(model_size)
            self.use_faster_whisper = False
            logging.info("[CPU Transcriber] Using standard whisper")
        else:
            raise RuntimeError("No whisper implementation available!")
    
    def transcribe_file(self, audio_path):
        """Transcribe a single audio file"""
        try:
            if self.use_faster_whisper:
                segments, info = self.model.transcribe(str(audio_path), beam_size=5)
                # Combine all segments into a single transcript
                transcript = " ".join([segment.text for segment in segments])
            else:
                result = self.model.transcribe(str(audio_path))
                transcript = result["text"]
            
            return transcript.strip()
        except Exception as e:
            logging.error(f"[CPU Transcriber] Error transcribing {audio_path}: {e}")
            return None
    
    def run(self, host, port, transcription_queue, llm_queue, should_send_server_ready=None):
        """
        Main loop that watches for audio files and transcribes them.
        
        Args:
            host: Not used in CPU mode (kept for interface compatibility)
            port: Not used in CPU mode (kept for interface compatibility)
            transcription_queue: Queue to send transcription outputs
            llm_queue: Queue for LLM communication (not used in file-based mode)
            should_send_server_ready: Multiprocessing Value to signal readiness
        """
        # Create directories
        self.input_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.initialize_model()
        
        if should_send_server_ready is not None:
            should_send_server_ready.value = True
        
        logging.info(f"[CPU Transcriber] Watching {self.input_dir} for WAV files...")
        logging.info("[CPU Transcriber] Drop WAV files into input_audio/ to process them")
        
        processed_files = set()
        uid_counter = 0
        
        while True:
            try:
                # Look for WAV files
                wav_files = list(self.input_dir.glob("*.wav"))
                
                for wav_file in wav_files:
                    # Skip if already processed
                    if wav_file.name in processed_files:
                        continue
                    
                    logging.info(f"[CPU Transcriber] Processing: {wav_file.name}")
                    
                    # Transcribe the file
                    transcript = self.transcribe_file(wav_file)
                    
                    if transcript:
                        # Create transcription output in same format as GPU version
                        transcription_output = {
                            "uid": f"cpu_{uid_counter}",
                            "prompt": transcript,
                            "eos": True  # File-based processing always has EOS
                        }
                        
                        # Send to transcription queue
                        transcription_queue.put(transcription_output)
                        logging.info(f"[CPU Transcriber] Transcribed: '{transcript}'")
                        
                        uid_counter += 1
                    
                    # Move processed file
                    dest_path = self.processed_dir / wav_file.name
                    shutil.move(str(wav_file), str(dest_path))
                    processed_files.add(wav_file.name)
                    logging.info(f"[CPU Transcriber] Moved to processed/")
                
                # Sleep before next check
                time.sleep(1)
                
            except KeyboardInterrupt:
                logging.info("[CPU Transcriber] Shutting down...")
                break
            except Exception as e:
                logging.error(f"[CPU Transcriber] Error in main loop: {e}")
                time.sleep(1)
