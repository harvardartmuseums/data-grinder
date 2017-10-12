# Data Grinder

Data enrichment via a variety of third party services

Setup
=====

Clone config-template.py to config.py.  
Enter your API keys and credentials.  
Create a file named vision-credentials.json and paste in a sevice account key generated through your Google account API dashboard. Read more at https://cloud.google.com/vision/docs/common/auth.  


Implemented
===========

Clarifai: tag features, extract colors  
Imagga: tag features, extract colors, categorize  
Google Vision: tag features, find faces, find text  
Microsoft Vision: categories, tags, description, faces, color  


Not Implemented
===============

Betaface: find faces  
Open Calais: extract and tag entities