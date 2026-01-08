import openai 
import os
import base64
from enum import Enum

class QwenModel(Enum):
	QWEN_2_5_VL_7B = (
		"qwen-2-5-vl-7b",
		"Qwen/Qwen2.5-VL-7B-Instruct"
	)
	QWEN_2_5_VL_72B = (
		"qwen-2-5-vl-72b",
		"Qwen/Qwen2.5-VL-72B-Instruct"
	)

	def __init__(self, name: str, model_id: str):
		self._model_id = model_id
		self._name = name

	def list_models():
		return [
			{
				"name": model.name,
				"model_id": model.model_id
			}
			for model in QwenModel
		]
	@property
	def model_id(self):
		return self._model_id    
	
	@property
	def name(self):
		return self._name    

class Qwen(object):

	def __init__(self):
		self.api_key = os.getenv("HYPERBOLIC_API_KEY")

	def fetch(self, photo_file, model: QwenModel = QwenModel.QWEN_2_5_VL_7B):
		client = openai.OpenAI(
			api_key = self.api_key,
			base_url="https://api.hyperbolic.xyz/v1"
		)

		with open(photo_file, 'rb') as image:
			image_content = base64.b64encode(image.read()).decode('utf-8')

		prompt =  [ 
				{ "role": "system", "content": "You are a helpful assistant." }, 
				{ "role": "user", "content": [  
					{ 
						"type": "text", 
						"text": "Describe this image:" 
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
				max_tokens = 2000
			)
			
			result = {
				"description": response.model_dump(),
				"prompt": "",
				"status": 200
			}

			return result

		except Exception as e:
			return {
				"status": 500,
				"prompt": prompt,
				"description": str(e)
			}
