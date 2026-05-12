import openai
import os
import base64
import httpx
from enum import Enum

class QwenModel(Enum):
	QWEN_2_5_VL_7B = (
		"qwen-2-5-vl-7b",
		"Qwen/Qwen2.5-VL-7B-Instruct",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 1.0},
		"2026-03-01"
	)
	QWEN_2_5_VL_72B = (
		"qwen-2-5-vl-72b",
		"Qwen/Qwen2.5-VL-72B-Instruct",
		{"maxTokens": 2000, "temperature": 1.0, "topP": 1.0},
		"2026-03-01"
	)

	def __init__(self, name: str, model_id: str, inference_config: dict, eol_date: str):
		self._model_id = model_id
		self._name = name
		self._inference_config = inference_config
		self._eol_date = eol_date

	def list_models():
		return [
			{
				"name": model.name,
				"model_id": model.model_id,
				"eol_date": model.eol_date,
				"provider": model.provider
			}
			for model in QwenModel
		]
	@property
	def model_id(self):
		return self._model_id    
	
	@property
	def name(self):
		return self._name    

	@property
	def inference_config(self):
		return self._inference_config

	@property
	def eol_date(self):
		return self._eol_date

	@property
	def provider(self):
		return "Qwen"

_client = None

class Qwen(object):

	def __init__(self):
		self.api_key = os.getenv("HYPERBOLIC_API_KEY")

	def get_client(self, connect_timeout=10, read_timeout=60):
		global _client
		if _client is None:
			_client = openai.OpenAI(
				api_key=self.api_key,
				base_url="https://api.hyperbolic.xyz/v1",
				timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
			)
		return _client

	def fetch(self, photo_file, model: QwenModel = QwenModel.QWEN_2_5_VL_7B, prompt=None, connect_timeout=10, read_timeout=60):
		client = self.get_client(connect_timeout, read_timeout)

		with open(photo_file, 'rb') as image:
			image_content = base64.b64encode(image.read()).decode('utf-8')

		prompt =  [ 
				{ "role": "system", "content": "You are a helpful assistant." }, 
				{ "role": "user", "content": [  
					{ 
						"type": "text", 
						"text": prompt or "Describe this image:"
					},
					{ 
						"type": "image_url",
						"image_url": {
							"url": f"data:image/png;base64,{image_content}"
						}
					}
				] } 
			]
		try: 
			response = client.chat.completions.create(
				model = model.model_id,
				messages = prompt,
				max_tokens = model.inference_config["maxTokens"]
			)
			
			return {
				"body": response.choices[0].message.content,
				"model": response.model,
				"provider": model.provider,
				"status": 200,
				"full": response.model_dump()
			}

		except Exception as e:
			return {
				"body": None,
				"model": model.model_id,
				"provider": model.provider,
				"status": 500,
				"description": str(e),
				"full": None
			}
