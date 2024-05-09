from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
import os

def load_model(base_model_path, lora_path, quantization, trust_remote_code = False, 
               devices="0", use_fast_tokenizer = False, use_flash_attention_2=False, torch_dtype="f32"):
	tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code, use_fast=use_fast_tokenizer) 
	# This is for llama2 models, but doesn't seem to have
	# adverse effects on benchmarks for other models.

	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "right"
	if torch_dtype == "f32":
		torch_dtype = torch.float32
	elif torch_dtype == "f16":
		torch_dtype = torch.float16
	elif torch_dtype == "b16":
		torch_dtype = torch.bfloat16
	else:
		raise ValueError("Invalid torch_dtype: " + torch_dtype)
	# Quantization Config
	if quantization == '4bit':
		# load as 4 bit
		quant_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.float16,
			bnb_4bit_use_double_quant=False
		)
	elif quantization == '8bit':
		# load as 8 bit
		quant_config = BitsAndBytesConfig(
			load_in_8bit=True,
		)
	else:
		quant_config = None

	# Model
	if quant_config:
		base_model = AutoModelForCausalLM.from_pretrained(
			base_model_path,
			quantization_config=quant_config,
			device_map="auto",
			trust_remote_code=trust_remote_code,
			use_flash_attention_2=use_flash_attention_2,
			torch_dtype=torch_dtype,
		)
	else:
		base_model = AutoModelForCausalLM.from_pretrained(
			base_model_path,
			device_map="auto",
			trust_remote_code=trust_remote_code,
   			use_flash_attention_2=use_flash_attention_2,
      		torch_dtype=torch_dtype,
		)

	if lora_path:
		peft_model = PeftModel.from_pretrained(base_model, lora_path)
		return peft_model, tokenizer
	else:
		return base_model, tokenizer
