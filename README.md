# Ana's Ping Pong

Data science retreat (https://datascienceretreat.com/) portfolio project.

We use satellite images and computer vision to spot and tag public ping pong 
tables on a map.
  
## Webserver

Short instructions:

- create new environment: `python3 -m venv venv`
- activate: `. venv/bin/activate`
- install requirements: `pip install -r requirements`
- `pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu  -f https://download.pytorch.org/whl/cpu/torch_stable.html`
- `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git')`
- `cd webserver`
- `export FLASK_APP=anaspingpong`
- `export FLASK_ENV=development`
- `flask init-db`
- `flask run`
