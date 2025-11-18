import os
import time
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)


class CPUTTS:
    """
    CPU-based TTS that writes outputs to files and optionally uses espeak for playback.
    """
    
    def __init__(self):
        self.output_dir = Path("outputs")
        self.use_espeak = False
        
    def check_espeak(self):
        """Check if espeak is available"""
        import subprocess
        try:
            subprocess.run(['espeak', '--version'], 
                         capture_output=True, 
                         check=True,
                         timeout=5)
            self.use_espeak = True
            logging.info("[CPU TTS] espeak is available and will be used for audio playback")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.use_espeak = False
            logging.info("[CPU TTS] espeak not available, will only write text outputs")
    
    def speak_text(self, text):
        """Use espeak to speak the text"""
        if not self.use_espeak:
            return
        
        try:
            import subprocess
            # Use espeak to speak the text
            subprocess.Popen(
                ['espeak', text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logging.error(f"[CPU TTS] Error using espeak: {e}")
    
    def write_output(self, text):
        """Write output text to a timestamped file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_file = self.output_dir / f"output_{timestamp}.txt"
            
            with open(output_file, 'w') as f:
                f.write(text)
            
            logging.info(f"[CPU TTS] Wrote output to: {output_file.name}")
            return output_file
        except Exception as e:
            logging.error(f"[CPU TTS] Error writing output: {e}")
            return None
    
    def run(self, host, port, audio_queue, should_send_server_ready=None):
        """
        Main loop that processes LLM outputs and writes them to files.
        
        Args:
            host: Not used in CPU mode (kept for interface compatibility)
            port: Not used in CPU mode (kept for interface compatibility)
            audio_queue: Queue to receive LLM outputs
            should_send_server_ready: Multiprocessing Value to signal readiness
        """
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Check for espeak
        self.check_espeak()
        
        if should_send_server_ready is not None:
            should_send_server_ready.value = True
        
        logging.info(f"[CPU TTS] Ready. Outputs will be written to: {self.output_dir}/")
        
        last_output = None
        
        while True:
            try:
                # Get LLM output from queue
                llm_response = audio_queue.get()
                
                # Skip if queue has more items (process latest only)
                if audio_queue.qsize() != 0:
                    continue
                
                llm_output = llm_response["llm_output"][0]
                eos = llm_response.get("eos", True)
                
                # Only process if output is different from last
                if last_output != llm_output.strip():
                    logging.info(f"[CPU TTS] Processing: '{llm_output}'")
                    
                    # Write to file
                    output_file = self.write_output(llm_output)
                    
                    # Speak if espeak is available and this is final output
                    if eos and output_file:
                        self.speak_text(llm_output)
                    
                    last_output = llm_output.strip()
                
            except KeyboardInterrupt:
                logging.info("[CPU TTS] Shutting down...")
                break
            except Exception as e:
                logging.error(f"[CPU TTS] Error in main loop: {e}")
                time.sleep(1)
