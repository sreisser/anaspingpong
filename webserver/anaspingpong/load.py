from flask import (
    Blueprint, flash, g, redirect, render_template, send_file, request, url_for
)

from anaspingpong.db import get_db
from flask import current_app
import re
import os

bp = Blueprint('load', __name__)

def writeTablestoXML(tables, output_file):
    with open(output_file, 'w') as output:
        output.write('<?xml version="1.0" ?>\n')
        output.write('<markers>\n')
        for table in tables:
            table_marker = f"  <marker " \
                          f"lat=\"{table['latitude']}\" " \
                          f"lng=\"{table['longitude']}\"/>\n"
            output.write(table_marker)
        output.write('</markers>\n')

@bp.route('/')
def index():
    return render_template('load/index.html')

@bp.route('/data')
def data():
    return send_file(current_app.config['DATA_XML'])


@bp.route('/predict', methods=('GET', 'POST'))
def predict():
    if request.method == 'POST':
        center = request.form['location']
        zoom = int(request.form['zoom'])
        center = re.sub('[(),]', "", center).split()
        error = None
        center_lat = float(center[0])
        center_lon = float(center[1])
        hash_id = int(center_lat*center_lon*10_000)

    #    predictions = get_tables(center_lat, center_lon)

        if error is not None:
            flash(error)
        else:
            db = get_db()

            db.execute(
                'INSERT OR IGNORE INTO tables (hash, latitude, longitude)'
                ' VALUES (?, ?, ?)',
                (hash_id, center_lat, center_lon)
            )
            db.commit()
            tables = db.execute(
                'SELECT latitude, longitude '
                ' FROM tables'
            ).fetchall()
            writeTablestoXML(tables, os.path.join('anaspingpong',
                                                  current_app.config['DATA_XML']))
           # return redirect(url_for('load.index'))
    return render_template('load/index.html',
                           center_lat=f'{center_lat:.6f}',
                           center_lon=f'{center_lon:.6f}',
                           zoom=f'{zoom}')

    # return render_template('load/index.html',
    #                        center_lat="",
    #                        center_lon="",
    #                        zoom="")

