from os.path import join, abspath, split
import time
import jinja2
import numpy as np
import pandas as pd
from flask import jsonify, send_from_directory, request
from werkzeug.exceptions import BadRequest
import logging
logging.getLogger("requests.packages.urllib3.connectionpool").setLevel(logging.WARNING)
from kitchensink import setup_client, client, do, du, dp
setup_client('http://power:6323/')


from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.server.views import make_json
import ujson

from objects import AjaxDataSource, TaxiApp, ARDataSource, ds, get_data
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
    return send_from_directory(_web_path, filename)

gbounds = ds.gbounds
@bokeh_app.route("/bokeh/taxidata/<pickup>/", methods=['GET', 'POST'])
def taxidata(pickup):
    st = time.time()
    if pickup == 'pickup':
        pickup = True
    else:
        pickup = False
    try:
        data = request.get_json()
    except BadRequest:
        import pdb; pdb.set_trace()
        data = {}
    if data.get('filter_url'):
        filters = du(data.get('filter_url'))
    else:
        filters = None
    if data:
        bounds = (data['xmin'], data['xmax'], data['ymin'], data['ymax'])
        bounds = (max(bounds[0], gbounds[0]),
                  min(bounds[1], gbounds[1]),
                  max(bounds[2], gbounds[2]),
                  min(bounds[3], gbounds[3]))
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
    print output
    output['image'] = [data.tolist()]
    result = make_json(ujson.dumps(output))
    return result
