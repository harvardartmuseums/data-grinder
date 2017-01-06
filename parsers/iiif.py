import requests

class IIIFImage(object):

	def fetch(self, iiifImageURI):
		response = requests.get('%s/info.json' % iiifImageURI)

		return response.json()