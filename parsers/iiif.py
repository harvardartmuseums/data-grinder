import requests

class IIIFImage(object):

	def get_base_uri(id):
		return 'https://ids.lib.harvard.edu/ids/iiif/%s' % id

	def fetch(self, iiifImageURI):
		response = requests.get('%s/info.json' % iiifImageURI)

		return response.json()