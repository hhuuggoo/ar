from os.path import join, abspath, split

import jinja2
import numpy as np
import pandas as pd
from flask import jsonify, send_from_directory, request
from werkzeug.exceptions import BadRequest

from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page

from objects import AjaxDataSource, TaxiApp, ARDataSource
_templates_path = join(abspath(split(__file__)[0]), "templates")
_web_path = join(abspath(split(__file__)[0]), "web")

def render(template, **context):
    with open(join(_templates_path, template)) as f:
        template = f.read()
        template = jinja2.Template(template)
    return template.render(**context)

@bokeh_app.route("/bokeh/taxis/")
@object_page("taxis")
def main():
    return TaxiApp.create()


@bokeh_app.route('/bokehjs/static/app/<path:filename>')
def arstatics(filename):
    print (_web_path, filename)
    return send_from_directory(_web_path, filename)

from partition import ARDataset, KSXChunkedGrid
ds = ARDataset()
def get_data(pickup, local_bounds, filters):
    if pickup:
        xfield = 'pickup_longitude'
        yfield = 'pickup_latitude'
    else:
        xfield = 'dropoff_longitude'
        yfield = 'dropoff_latitude'
    local_indexes, (grid_shape, results) = ds.project(
        local_bounds, xfield, yfield, filters
    )
    lxdim1, lxdim2, lydim1, lydim2 = local_indexes
    grid = KSXChunkedGrid(results, grid_shape[-1])
    data = grid.get(lxdim1, lxdim2, lydim1, lydim2)
    data = data.T[:]
    data = data ** 0.2
    return data

gbounds = (-74.19, -73.7, 40.5, 40.99)
@bokeh_app.route("/bokeh/taxidata/<pickup>/", methods=['GET', 'POST'])
def taxidata(pickup):
    if pickup == 'pickup':
        pickup = True
    else:
        pickup = False
    try:
        bounds = request.get_json()
    except BadRequest:
        import pdb; pdb.set_trace()
        bounds = {}
    filters = None
    print '***BOUNDS', bounds
    if bounds:
        bounds = (bounds['xmin'], bounds['xmax'], bounds['ymin'], bounds['ymax'])
    else:
        bounds = gbounds
    # global_offset_x = xmin
    # global_offset_y = ymin
    # global_x_range = [xmin, xmax]
    # global_y_range = [ymin, ymax]
    data = get_data(pickup, bounds, filters)
    xmin, xmax, ymin, ymax = bounds
    output = dict(x=[xmin],
                  y=[ymin],
                  dw=[xmax-xmin],
                  dh=[ymax-ymin],
                  palette=["Greys-256"])
    output['image'] = [data.tolist()]
    return jsonify(output)
