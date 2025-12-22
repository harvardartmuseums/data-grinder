import requests

class IIIFImage(object):

	def __init__(self, id):
		self.id = id
		self.base_uri = f"https://ids.lib.harvard.edu/mps/{self.id}"
		self.info_url = f"{self.base_uri}/info.json"

		self.info = requests.get(self.info_url).json()

	def get_base_uri(self):
		return self.base_uri

	def get_full_image_url(self):
		return f"{self.base_uri}/full/full/0/default.jpg"
	
	def get_scaled_image_url(self, scale):
		return f"{self.base_uri}/full/{scale}/0/default.jpg"
	
	def get_fragment_image_url(self, x, y, w, h):
		return f"{self.base_uri}/{x},{y},{w},{h}/full/0/default.jpg"

	def fetch(self):
		return self.info
	
	def info(self):
		return self.info