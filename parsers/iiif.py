import logging
import requests
from typing import Optional, Dict, Any
from urllib.parse import urlparse

class IIIFImage(object):
	"""A class for handling IIIF (International Image Interoperability Framework) images."""

	logger = logging.getLogger(__name__)
	
	# Status constants
	STATUS_OK = "ok"
	STATUS_BAD = "bad"
	STATUS_UNKNOWN = "unknown"
	
	# Request timeout
	REQUEST_TIMEOUT = 21
	
	# Known URL patterns and their corresponding IDs
	KNOWN_URL_MAPPINGS = {
		'https://ids.lib.harvard.edu/ids/': 'harvard_ids',
		'https://ids.lib.harvard.edu/mps/': 'harvard_mps',
		'https://mps.lib.harvard.edu/assets/image/drs:': 'harvard_mps_drs_asset',
	}

	def __init__(self, uri: str):
		"""
		Initialize IIIFImage with a base URI.
		
		Args:
			uri: The base URI for the IIIF image service
		"""
		if not uri or not isinstance(uri, str):
			raise ValueError("URI must be a non-empty string")
		
		# Validate URI format
		parsed = urlparse(uri)
		if not parsed.scheme or not parsed.netloc:
			raise ValueError("URI must be a valid URL with scheme and netloc")
		
		self.original_uri = uri.rstrip('/')
		self.base_uri = self.original_uri
		self.info_url = f"{self.base_uri}/info.json"
		self.id: Optional[int] = -1
		self.status = self.STATUS_UNKNOWN
		self.info: Dict[str, Any] = {}
		self.request_metrics = {
			"info_json_fetches": 0,
		}
		
		self.__fetch_info()

	def get_base_uri(self) -> str:
		"""Get the base URI for this IIIF image service."""
		return self.base_uri

	def get_full_image_url(self) -> str:
		"""Get the URL for the full-size image."""
		return f"{self.base_uri}/full/full/0/default.jpg"
	
	def get_scaled_image_url(self, scale: str) -> str:
		"""
		Get the URL for a scaled version of the image.
		
		Args:
			scale: Scale parameter (e.g., '!150,150', '50%', 'max')
			
		Returns:
			URL for the scaled image
		"""
		if not scale:
			raise ValueError("Scale parameter cannot be empty")
		return f"{self.base_uri}/full/{scale}/0/default.jpg"
	
	def get_fragment_image_url(self, x: int, y: int, w: int, h: int) -> str:
		"""
		Get the URL for a rectangular fragment of the image.
		
		Args:
			x: X coordinate of the top-left corner
			y: Y coordinate of the top-left corner  
			w: Width of the fragment
			h: Height of the fragment
			
		Returns:
			URL for the image fragment
			
		Raises:
			ValueError: If any coordinate is negative
		"""
		if any(coord < 0 for coord in [x, y, w, h]):
			raise ValueError("All coordinates must be non-negative")
		if w == 0 or h == 0:
			raise ValueError("Width and height must be greater than 0")
		return f"{self.base_uri}/{x},{y},{w},{h}/full/0/default.jpg"

	def fetch(self) -> Dict[str, Any]:
		"""
		Get the IIIF info.json data.
		
		Returns:
			Dictionary containing the IIIF info.json data
		"""
		return self.info

	def is_valid(self) -> bool:
		"""Check if the IIIF image service is valid and accessible."""
		return self.status == self.STATUS_OK

	def get_id(self) -> Optional[int]:
		"""Get the extracted ID for this IIIF image."""
		return self.id

	def get_status(self) -> str:
		"""Get the current status of the IIIF image service."""
		return self.status

	def get_request_metrics(self) -> Dict[str, int]:
		"""Get counters for outbound IIIF requests."""
		return dict(self.request_metrics)
	
	def __fetch_info(self):
		"""Fetch IIIF info.json with error handling."""
		try:
			self.request_metrics["info_json_fetches"] += 1
			self.logger.info("Fetching IIIF info.json: %s", self.info_url)
			response = requests.get(self.info_url, timeout=self.REQUEST_TIMEOUT)
			response.raise_for_status()
			resolved_base_uri = response.url.removesuffix("/info.json").rstrip("/")
			self.logger.info(
				"Fetched IIIF info.json: requested=%s resolved=%s status=%s",
				self.info_url,
				response.url,
				response.status_code,
			)
			self.status = self.STATUS_OK
			self.base_uri = resolved_base_uri
			self.id = self.__extract_id_from_url(resolved_base_uri)
			self.info = response.json()
		except requests.exceptions.RequestException as e:
			self.logger.error("Failed to fetch info.json from %s: %s", self.info_url, e)
			self.status = self.STATUS_BAD
			self.info = {}
		except ValueError as e:  # JSON decode error
			self.logger.error("Invalid JSON in info.json from %s: %s", self.info_url, e)
			self.status = self.STATUS_BAD
			self.info = {}
	
	def __extract_id_from_url(self, url: str) -> int:
		"""
		Extract ID from URL using known URL mappings.
		
		Args:
			url: The URL to extract ID from
			
		Returns:
			Extracted ID string
		"""
		if not url:
			return -1
			
		for base_url, service_id in self.KNOWN_URL_MAPPINGS.items():
			if url.startswith(base_url):
				# Extract the path after the base URL
				path_after_base = url[len(base_url):].strip('/')
				return path_after_base
		
		# Fallback: if no known pattern matches, use the original logic
		return -1
