import os
from flask import Flask
from pathlib import Path

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    Path("anaspingpong/data").mkdir(exist_ok=True)
    app.config.from_mapping(
     #   SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'anaspingpong.sqlite'),
        DATA_XML='data/tables.xml',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
        print('loading config from config.py')
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import db
    db.init_app(app)

  #  from . import prediction

    from . import load
    app.register_blueprint(load.bp)
    app.add_url_rule('/', endpoint='index')

    return app
