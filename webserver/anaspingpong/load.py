
from flask import (
    Blueprint, flash, g, redirect, render_template, send_file, request, url_for
)


from anaspingpong.db import get_db
from anaspingpong.prediction import get_tables
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
    return render_template('load/index.html',
                           key=current_app.config['GOOGLE_MAPS_KEY'])

@bp.route('/data')
def data():
    return send_file(current_app.config['DATA_XML'])


@bp.route('/predict', methods=('GET', 'POST'))
def predict():
    if request.method == 'POST':
        zoom = int(request.form['zoom'])
        error = None
        center_lat = float(request.form['center_lat'])
        center_lon = float(request.form['center_lon'])
        hash_id = int(center_lat*center_lon*10_000)

        pred_lon, pred_lat = get_tables(center_lat, center_lon)
        hashes = [int(lat * lon * 10_000) for lat, lon in zip(pred_lat, pred_lon)]

        if error is not None:
            flash(error)
        else:
            db = get_db()
            for i in range(len(hashes)):
                db.execute(
                    'INSERT OR IGNORE INTO tables (hash, latitude, longitude)'
                    ' VALUES (?, ?, ?)',
                    (hashes[i], pred_lat[i], pred_lon[i])
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
                            key=current_app.config['GOOGLE_MAPS_KEY'],
                            center_lat=f'{center_lat:.6f}',
                            center_lon=f'{center_lon:.6f}',
                            zoom=f'{zoom}')
    return render_template('load/index.html',
                           key=current_app.config['GOOGLE_MAPS_KEY'])

