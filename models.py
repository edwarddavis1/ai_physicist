import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient
import os

class GetHuggingFaceModelLocal:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initialize the HuggingFace model.
        
        Args:
            model_id (str): Hugging Face model identifier
        """
        self.model_id = model_id
        self.device = "cpu"  # Using CPU for now
        self.system_prompt = None
        
        print(f"Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        print(f"Model loaded on {self.device}")

    def ask_question(self, question, max_length=512,
                    temperature=0.7, top_p=0.9):
        """
        Ask a question to the loaded model and get a response.
        
        Args:
            question (str): The question to ask
            system_prompt (str, optional): System prompt to set context
            max_length (int): Maximum length of the response
            temperature (float): Controls randomness (0.0 = deterministic, 1.0 = very random)
            top_p (float): Controls diversity via nucleus sampling
        
        Returns:
            str: The model's response
        """
        # Prepare messages
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})
        
        # Apply chat template and tokenize
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response (works for Qwen models)
        if "assistant" in full_response:
            assistant_response = full_response.split("assistant")[-1].strip()
            return assistant_response
        else:
            return full_response.strip()

    def set_system_prompt(self, prompt):
        """
        Set a system prompt that will be used for all subsequent questions.
        
        Args:
            prompt (str): The system prompt to set
        """
        self.system_prompt = prompt
        print(f"System prompt set: {prompt[:50]}...")

    def clear_system_prompt(self):
        """
        Clear the current system prompt.
        """
        self.system_prompt = None
        print("System prompt cleared")

    def get_model_info(self):
        """
        Get information about the current model.
        
        Returns:
            dict: Model information including ID, device, and memory usage
        """
        return {
            "model_id": self.model_id,
            "device": self.device,
            "api_type": "Local HuggingFace Transformers",
            "system_prompt": self.system_prompt is not None
        }


class GetHuggingFaceModel:
    def __init__(self, model_id="microsoft/DialoGPT-medium", api_token=None):
        """
        Initialize the HuggingFace Inference API client.
        
        Args:
            model_id (str): Hugging Face model identifier for inference API
            api_token (str, optional): HuggingFace API token. If None, will try to get from environment
        """
        self.model_id = model_id
        self.system_prompt = None
        
        # Get API token from parameter or environment variable
        if api_token is None:
            api_token = os.getenv("HUGGINGFACE_API_TOKEN")
            if api_token is None:
                print("Warning: No HuggingFace API token provided. Some models may not work.")
        
        print(f"Initializing HuggingFace Inference client for model: {model_id}")
        
        # Initialize the inference client
        self.client = InferenceClient(
            model=model_id,
            token=api_token
        )
        
        # Initialize tokenizer for message formatting (lightweight operation)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Set pad token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.has_tokenizer = True
            print(f"Tokenizer loaded for message formatting")
        except Exception as e:
            print(f"Could not load tokenizer for {model_id}: {e}")
            print("Will use simple string formatting for messages")
            self.has_tokenizer = False
        
        print(f"HuggingFace Inference client ready")

    def ask_question(self, question, max_new_tokens=512, 
                    temperature=0.7, top_p=0.9, stream=False):
        """
        Ask a question to the model via HuggingFace Inference API and get a response.
        
        Args:
            question (str): The question to ask
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Controls randomness (0.0 = deterministic, 1.0 = very random)
            top_p (float): Controls diversity via nucleus sampling
            stream (bool): Whether to stream the response (returns generator if True)
        
        Returns:
            str or generator: The model's response, or generator if streaming
        """
        try:
            # Prepare messages for chat models
            messages = []
            if self.system_prompt is not None:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": question})
            
            # Try chat completion first (for modern instruction-tuned models)
            try:
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream
                )
                
                if stream:
                    # Return generator for streaming responses
                    return self._process_stream(response)
                else:
                    # Extract the response content
                    return response.choices[0].message.content.strip()
                    
            except Exception as chat_error:
                print(f"Chat completion failed: {chat_error}")
                print("Falling back to text generation...")
                
                # Fallback to text generation for models that don't support chat
                if self.has_tokenizer:
                    # Use tokenizer to format messages properly
                    input_text = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # Simple string formatting fallback
                    if self.system_prompt:
                        input_text = f"System: {self.system_prompt}\nUser: {question}\nAssistant:"
                    else:
                        input_text = f"User: {question}\nAssistant:"
                
                # Use text generation endpoint
                response = self.client.text_generation(
                    prompt=input_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream,
                    return_full_text=False  # Only return generated text, not the prompt
                )
                
                if stream:
                    # Return generator for streaming responses
                    return self._process_text_stream(response)
                else:
                    return response.strip()
                    
        except Exception as e:
            print(f"Error during inference: {e}")
            return f"Error: Could not generate response. {str(e)}"

    def _process_stream(self, stream):
        """
        Process streaming chat completion responses.
        
        Args:
            stream: Generator from chat_completion with stream=True
            
        Yields:
            str: Incremental response tokens
        """
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def _process_text_stream(self, stream):
        """
        Process streaming text generation responses.
        
        Args:
            stream: Generator from text_generation with stream=True
            
        Yields:
            str: Incremental response tokens
        """
        for token in stream:
            yield token

    def set_system_prompt(self, prompt):
        """
        Set a system prompt that will be used for all subsequent questions.
        
        Args:
            prompt (str): The system prompt to set
        """
        self.system_prompt = prompt
        print(f"System prompt set: {prompt[:50]}...")

    def clear_system_prompt(self):
        """
        Clear the current system prompt.
        """
        self.system_prompt = None
        print("System prompt cleared")

    def get_model_info(self):
        """
        Get information about the current model.
        
        Returns:
            dict: Model information including ID and API status
        """
        return {
            "model_id": self.model_id,
            "api_type": "HuggingFace Inference API",
            "has_tokenizer": self.has_tokenizer,
            "system_prompt": self.system_prompt is not None
        }

