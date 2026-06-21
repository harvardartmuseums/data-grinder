import requests
import os
import io
from enum import Enum
from PIL import Image

class MCSVisionModel(Enum):
	BASE = (
		"mcs",
		"",
		["analyze", "describe"],
		4194304,
		None
	)

	def __init__(self, name: str, model_id: str, functions: list, image_size_limit: int, eol_date: str):
		self._model_id = model_id
		self._name = name
		self._functions = functions
		self._image_size_limit = image_size_limit
		self._eol_date = eol_date

	def list_models():
		return [
			{
				"name": model.name,
				"model_id": model.model_id,
				"functions": model.functions,
				"image_size_limit": model.image_size_limit,
				"provider": model.provider,
				"eol_date": model.eol_date
			}
			for model in MCSVisionModel
		]

	@property
	def model_id(self):
		return self._model_id    
	
	@property
	def name(self):
		return self._name   
	
	@property
	def functions(self):
		return self._functions

	@property
	def image_size_limit(self):
		return self._image_size_limit
	
	@property
	def provider(self):
		return "Microsoft Cognitive Services"

	@property
	def eol_date(self):
		return self._eol_date

class MCSVision(object):

	def __init__(self):
		self.api_key = os.getenv("MICROSOFT_CS_KEY")
		self.endpoint = os.getenv("MICROSOFT_CS_ENDPOINT")
		self.api_version = os.getenv("MICROSOFT_CS_API_VERSION")

		self.base_url = self.endpoint + 'vision/' + self.api_version + '/'

	def _read_image_bytes(self, photo_file):
		"""Read image file, resampling if necessary to fit within the model's image size limit.

		If the model has no image size limit or the raw bytes are already within the limit,
		returns raw bytes unchanged. Otherwise, iteratively reduces JPEG quality and then
		pixel dimensions until the encoded size is strictly under the limit.

		Raises:
			ValueError: If unable to resample image below the model's limit.
		"""
		limit = MCSVisionModel.BASE.image_size_limit

		with open(photo_file, 'rb') as image:
			raw_bytes = image.read()

		if limit is None or len(raw_bytes) < limit:
			return raw_bytes

		# Raw bytes exceed the model's image size limit — resample
		img = Image.open(io.BytesIO(raw_bytes))
		quality = 95

		while True:
			buf = io.BytesIO()
			img.save(buf, format="JPEG", quality=quality, optimize=True)
			data = buf.getvalue()

			if len(data) < limit:
				return data

			# Reduce quality first
			if quality > 20:
				quality -= 5
				continue

			# Quality floor reached — shrink pixel dimensions by 10%
			width, height = img.size
			new_width = int(width * 0.9)
			new_height = int(height * 0.9)

			if new_width < 10 or new_height < 10:
				raise ValueError(
					f"Unable to resample image '{photo_file}' below {limit} bytes. "
					f"Original size: {len(raw_bytes)} bytes."
				)

			img = img.resize((new_width, new_height), Image.LANCZOS)

	def fetch_analyze(self, photo_file):
		url = self.base_url + 'analyze'

		params = {'visualFeatures': 'Categories,Tags,Description,Faces,Color,Objects'}
		headers = {
			'Ocp-Apim-Subscription-Key': self.api_key,
			'Content-Type': 'application/octet-stream'
		}

		image_content = self._read_image_bytes(photo_file)
		response = requests.post(url, data=image_content, params=params, headers=headers)
		result = response.json()
		result['provider'] = MCSVisionModel.BASE.provider
		return result
	
	def fetch_description(self, photo_file):
		url = self.base_url + 'describe'

		params = {'maxCandidates': 3}
		headers = {
			'Ocp-Apim-Subscription-Key': self.api_key,
			'Content-Type': 'application/octet-stream'
		}

		image_content = self._read_image_bytes(photo_file)
		response = requests.post(url, data=image_content, params=params, headers=headers)
		result = response.json()
		result['provider'] = MCSVisionModel.BASE.provider
		return result
