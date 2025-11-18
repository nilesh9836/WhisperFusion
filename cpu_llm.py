import os
import logging
import time

logging.basicConfig(level=logging.INFO)


class CPULLMEngine:
    """
    CPU-based LLM engine that supports OpenAI API and local HuggingFace models.
    """
    
    def __init__(self, provider='openai', hf_model='google/flan-t5-small'):
        """
        Initialize the LLM engine.
        
        Args:
            provider: 'openai' or 'hf' (HuggingFace)
            hf_model: HuggingFace model name to use if provider is 'hf' or as fallback
        """
        self.provider = provider
        self.hf_model = hf_model
        self.model = None
        self.client = None
        
    def initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            import openai
            
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                logging.warning("[CPU LLM] OPENAI_API_KEY not found, falling back to HuggingFace")
                return False
            
            self.client = openai.OpenAI(api_key=api_key)
            logging.info("[CPU LLM] Initialized OpenAI client")
            return True
        except ImportError:
            logging.warning("[CPU LLM] openai package not installed, falling back to HuggingFace")
            return False
        except Exception as e:
            logging.warning(f"[CPU LLM] Error initializing OpenAI: {e}, falling back to HuggingFace")
            return False
    
    def initialize_huggingface(self):
        """Initialize HuggingFace model"""
        try:
            from transformers import pipeline
            
            logging.info(f"[CPU LLM] Loading HuggingFace model: {self.hf_model}")
            logging.info("[CPU LLM] This may take a while on first run (downloading model)...")
            
            # Use text2text-generation for models like flan-t5
            self.model = pipeline(
                "text2text-generation",
                model=self.hf_model,
                device=-1  # CPU
            )
            
            logging.info("[CPU LLM] HuggingFace model loaded")
            return True
        except Exception as e:
            logging.error(f"[CPU LLM] Error loading HuggingFace model: {e}")
            return False
    
    def generate_openai(self, prompt):
        """Generate response using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Keep your responses concise."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"[CPU LLM] OpenAI API error: {e}")
            return None
    
    def generate_huggingface(self, prompt):
        """Generate response using HuggingFace model"""
        try:
            # Format prompt for better responses
            formatted_prompt = f"Respond to this message: {prompt}"
            
            result = self.model(
                formatted_prompt,
                max_length=150,
                num_return_sequences=1,
                do_sample=False
            )
            
            return result[0]['generated_text'].strip()
        except Exception as e:
            logging.error(f"[CPU LLM] HuggingFace generation error: {e}")
            return None
    
    def run(self, transcription_queue, llm_queue, audio_queue):
        """
        Main loop that processes transcriptions and generates LLM responses.
        
        Args:
            transcription_queue: Queue to receive transcription outputs
            llm_queue: Queue to send LLM outputs
            audio_queue: Queue to send outputs for TTS
        """
        # Initialize the appropriate provider
        if self.provider == 'openai':
            openai_initialized = self.initialize_openai()
            if not openai_initialized:
                # Fall back to HuggingFace
                self.provider = 'hf'
                self.initialize_huggingface()
        else:
            self.initialize_huggingface()
        
        logging.info(f"[CPU LLM] Using provider: {self.provider}")
        logging.info("[CPU LLM] Ready to process transcriptions...")
        
        conversation_history = {}
        
        while True:
            try:
                # Get transcription from queue
                transcription_output = transcription_queue.get()
                
                # Skip if queue has more items (process latest only)
                if transcription_queue.qsize() != 0:
                    continue
                
                uid = transcription_output["uid"]
                prompt = transcription_output['prompt'].strip()
                eos = transcription_output.get("eos", True)
                
                logging.info(f"[CPU LLM] Processing: '{prompt}'")
                
                # Initialize conversation history for this UID
                if uid not in conversation_history:
                    conversation_history[uid] = []
                
                # Generate response
                start_time = time.time()
                
                if self.provider == 'openai':
                    response = self.generate_openai(prompt)
                else:
                    response = self.generate_huggingface(prompt)
                
                inference_time = time.time() - start_time
                
                if response:
                    logging.info(f"[CPU LLM] Generated response in {inference_time:.2f}s: '{response}'")
                    
                    # Add to conversation history
                    conversation_history[uid].append((prompt, response))
                    
                    # Create output in same format as GPU version
                    llm_output = {
                        "uid": uid,
                        "llm_output": [response],
                        "eos": eos,
                        "latency": inference_time
                    }
                    
                    # Send to queues
                    llm_queue.put(llm_output)
                    audio_queue.put({"llm_output": [response], "eos": eos})
                else:
                    logging.warning("[CPU LLM] Failed to generate response")
                
            except KeyboardInterrupt:
                logging.info("[CPU LLM] Shutting down...")
                break
            except Exception as e:
                logging.error(f"[CPU LLM] Error in main loop: {e}")
                time.sleep(1)
