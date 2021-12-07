# Public ping pong map

Data science retreat (https://datascienceretreat.com/) portfolio project.

We use satellite images and computer vision to spot and tag public ping pong 
tables on a map.
  
## Webserver

Short instructions:

- create new environment: `conda create --name anaspingpong`
- activate: `conda activate anaspingpong`
- install requirements: `pip install -r requirements`
- `cd webserver`
- `export FLASK_APP=anaspingpong`
- `export FLASK_ENV=development`
- `flask init-db`
- `flask run`
