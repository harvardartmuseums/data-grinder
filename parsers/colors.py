import requests

class Colors(object):

	# def __init__(self):
		# self.api_key = config.IMAGGA_KEY
		# self.api_secret = config.IMAGGA_SECRET

	def fetch_colors(self, photo_file):
		response = requests.get('https://ham-color-service.herokuapp.com/extract?image_url=%s' % photo_file)

		return response.json()		