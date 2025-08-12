import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class GetHuggingFaceModel:
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

