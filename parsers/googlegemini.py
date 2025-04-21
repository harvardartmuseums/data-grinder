import os
import base64
import requests
import json
from google import genai
from google.genai import types

class GoogleGemini(object):

	def __init__(self):
		self.api_key = os.getenv("GOOGLE_API_KEY")
			  			  
	def fetch(self, photo_file, model="gemini-2-0-flash-lite"):
		# Fetch the image data from the local copy
		with open(photo_file, 'rb') as image:
			image_content = base64.b64encode(image.read()).decode('utf-8')

		# Determine which model to call
		model_id = ""
		if model == "gemini-2-0-flash-lite":
			model_id = "gemini-2.0-flash-lite"

		if model == "gemini-2-0-flash":
			model_id = "gemini-2.0-flash"

		try:
			url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={self.api_key}"

			params = {
				"contents": [
					{
						"parts": [
							{
								"text": "Describe this image"
							},
							{
								"inline_data": {
									"mime_type": "image/jpeg",
									"data": image_content
								}
							}
						]
					}
				]
			}
			headers = {
				'Content-Type': 'application/json'
			}
			response = requests.post(url, headers=headers, json=params)

			result = response.json()
			response = result
			response['model'] = model_id
			response['status'] = 200			
			return response

		except Exception as e:
			response = {"status": 500}
			return response						
			



		# # image_content.decode('UTF-8')
		# # Construct the message
		# sys_message = "You are a helpful assistant."
		# message = [
		# 	"Describe this image",
		# 	image_content,
		# ]

		# try: 
		# 	# Run the prompt
		# 	client = genai.Client(api_key=self.api_key)
		# 	geminiresponse = client.models.generate_content(
		# 		config=types.GenerateContentConfig(
		# 			system_instruction=sys_message
		# 		),
		# 		model=model_id,
		# 		contents=message
		# 	)

		# 	response = geminiresponse.model_dump()

		# 	response['model'] = model_id
		# 	response['status'] = 200
		# 	return response
		
		# except Exception as e:
		# 	print(e)
		# 	response = {"status": 500}
		# 	return response			
