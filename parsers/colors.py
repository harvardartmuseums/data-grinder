import os
import requests

class Colors(object):

	def __init__(self):
		self.color_service = os.getenv("COLOR_SERVICE")

	def fetch_colors(self, photo_file):
		response = requests.get('%s/extract?image_url=%s' % (self.color_service, photo_file))

		return response.json()		