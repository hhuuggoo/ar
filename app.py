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

from objects import (AjaxDataSource, TaxiApp, ARDataSource, ds, get_data,
                     get_time_histogram, get_distance_histogram)
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
    data = get_data(pickup, bounds, filters)
    data = data ** 0.2
    data = data - data.min()
    data = data / data.max()
    data = data *255
    data =  data.astype('int64')

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

@bokeh_app.route("/bokeh/taxidata/timehist/", methods=['GET', 'POST'])
def taxidatatimehist():
    try:
        data = request.get_json()
    except BadRequest:
        data = {}
    if data.get('filter_url'):
        filters = du(data.get('filter_url'))
    else:
        filters = None
    print 'HIST', filters
    data = get_time_histogram(filters)
    data['y_bounds'] = [min(data['counts']), max(data['counts'])]
    result = make_json(ujson.dumps(data))
    return result

@bokeh_app.route("/bokeh/taxidata/distancehist/", methods=['GET', 'POST'])
def taxidatadistancehist():
    try:
        data = request.get_json()
    except BadRequest:
        data = {}
    if data.get('filter_url'):
        filters = du(data.get('filter_url'))
    else:
        filters = None
    print 'HIST', filters
    data = get_distance_histogram(filters)
    data['y_bounds'] = [min(data['counts']), max(data['counts'])]
    result = make_json(ujson.dumps(data))
    return result


gbounds = ds.gbounds
@bokeh_app.route("/bokeh/taxidatavsregular/<pickup>/", methods=['GET', 'POST'])
def taxidatavsregular(pickup):
    st = time.time()
    if pickup == 'pickup':
        pickup = True
    else:
        pickup = False
    try:
        data = request.get_json()
    except BadRequest:
        data = {}
    if data.get('filter_url'):
        filters = du(data.get('filter_url'))
    else:
        filters = None
    if data and data.get('xmin'):
        bounds = (data['xmin'], data['xmax'], data['ymin'], data['ymax'])
        bounds = (max(bounds[0], gbounds[0]),
                  min(bounds[1], gbounds[1]),
                  max(bounds[2], gbounds[2]),
                  min(bounds[3], gbounds[3]))
    else:
        bounds = gbounds
    if filters:
        unfiltered = get_data(pickup, bounds, None)
        filtered = get_data(pickup, bounds, filters)
        percents = np.percentile(unfiltered[unfiltered!=0], np.arange(100))
        unfiltered = np.interp(unfiltered, percents, np.arange(100))
        percents = np.percentile(filtered[filtered!=0], np.arange(100))
        filtered = np.interp(filtered, percents, np.arange(100))
        data = filtered - unfiltered
        data[data > 0] = data[data > 0] / data.max()
        data[data < 0] = - (data[data < 0] / data.min())
        data = data - data.min()
        data = data / data.max()
        data = data *255
        data =  data.astype('int64')
        palette = 'seismic-256'
    else:
        data = get_data(pickup, bounds, None)
        data = data ** 0.2
        data = data - data.min()
        data = data / data.max()
        data = data *255
        data =  data.astype('int64')
        palette = 'Greys-256'
    xmin, xmax, ymin, ymax = bounds
    output = dict(x=[xmin],
                  y=[ymin],
                  dw=[xmax-xmin],
                  dh=[ymax-ymin],
                  palette=[palette])
    print output
    output['image'] = [data.tolist()]
    result = make_json(ujson.dumps(output))
    return result
