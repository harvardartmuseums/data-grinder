import requests
import config

class Imagga(object):

	def __init__(self):
		self.api_key = config.IMAGGA_KEY
		self.api_secret = config.IMAGGA_SECRET

	def fetch(self, photo_file):
		response = requests.get('https://api.imagga.com/v1/tagging?url=%s' % photo_file, auth=(self.api_key, self.api_secret))

		return response.json()

	def fetch_categories(self, photo_file):
		response = requests.get('https://api.imagga.com/v1/categorizations/personal_photos?url=%s' % photo_file, auth=(self.api_key, self.api_secret))

		return response.json()

	def fetch_colors(self, photo_file):
		response = requests.get('https://api.imagga.com/v1/colors?url=%s' % photo_file, auth=(self.api_key, self.api_secret))

		return response.json()		