import requests
import os

class Imagga(object):

	def __init__(self):
		self.api_key = os.getenv("IMAGGA_KEY")
		self.api_secret = os.getenv("IMAGGA_SECRET")

	def fetch(self, photo_file):
		response = requests.get('https://api.imagga.com/v2/tags?image_url=%s' % photo_file, auth=(self.api_key, self.api_secret))

		return response.json()

	def fetch_categories(self, photo_file):
		response = requests.get('https://api.imagga.com/v2/categories/personal_photos?image_url=%s' % photo_file, auth=(self.api_key, self.api_secret))

		return response.json()

	def fetch_colors(self, photo_file):
		response = requests.get('https://api.imagga.com/v2/colors?image_url=%s' % photo_file, auth=(self.api_key, self.api_secret))

		return response.json()		