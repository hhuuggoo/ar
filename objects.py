import time
import datetime as dt

from bokeh.objects import  ServerDataSource, Plot, ColumnDataSource
from bokeh.widgets import HBox, VBox
from bokeh.plot_object import PlotObject
from bokeh.properties import (
    Datetime, HasProps, Dict, Enum, Either, Float, Instance, Int,
    List, String, Color, Include, Bool, Tuple, Any
)
from bokeh.plotting import image

class AjaxDataSource(ServerDataSource):
    url = String()

class ARDataSource(ServerDataSource):
    selector = Dict(String, Any)
    url = String()
    filter_url = String()

class TaxiApp(HBox):
    extra_generated_classes = [["TaxiApp", "TaxiApp", "HBox"]]
    extra_scripts = ['/bokehjs/static/app/src/js/ar_data_source.js']
    extra_js=['window.ar_data_source.main();']
    pickup_plot = Instance(Plot)
    dropoff_plot = Instance(Plot)
    gbounds = (-74.05, -73.75, 40.5, 40.99)
    pickup_raw_plot_source = Instance(ColumnDataSource)
    dropoff_raw_plot_source = Instance(ColumnDataSource)
    pickup_ar_plot_source = Instance(ARDataSource)
    dropoff_ar_plot_source = Instance(ARDataSource)


    @classmethod
    def create(cls):
        gbounds = cls.gbounds
        xmin, xmax, ymin, ymax = gbounds
        app = cls()
        data = ARDataSource(
            data_url="/bokeh/taxidata/pickup/",
            data=dict(
                x=[0], y=[0], dw=[xmax-xmin], dh=[ymax-ymin], palette=["Greys-256"]
            )
        )
        app.pickup_ar_plot_source = data
        plot = image(source=data,
                     image="image",
                     x="x",
                     y="y",
                     dw="dw",
                     dh="dh",
                     plot_width=400,
                     plot_height=600,
                     palette='palette',
                     x_range=[xmin, xmax], y_range=[ymin, ymax],
                     tools="pan,wheel_zoom,box_zoom,select,reset",
                     title='pickup'
        )
        app.pickup_plot = plot
        app.pickup_raw_plot_source = plot.select({'type' : ColumnDataSource})[0]
        data = ARDataSource(
            data_url="/bokeh/taxidata/dropoff/",
            data=dict(
                x=[0], y=[0], dw=[xmax-xmin], dh=[ymax-ymin], palette=["Greys-256"]
            )
        )
        app.dropoff_ar_plot_source = data
        plot = image(source=data,
                     image="image",
                     plot_width=400,
                     plot_height=600,
                     x="x",
                     y="y",
                     dw="dw",
                     dh="dh",
                     palette='palette',
                     x_range=[xmin, xmax], y_range=[ymin, ymax],
                     tools="pan,wheel_zoom,box_zoom,reset,select,reset",
                     title='dropoff'
        )
        app.dropoff_plot = plot
        app.dropoff_raw_plot_source = plot.select({'type' : ColumnDataSource})[0]
        app.children = [app.pickup_plot, app.dropoff_plot]
        return app

    def pickup_selector(self, obj, attrname, old, new):
        geom = new['data_geometry']
        if geom is None:
            self.pickup_ar_plot_source.filter_url = None;
            self.dropoff_ar_plot_source.filter_url = None;
            return
        print 'PICKUP**', geom
        lxmin = min(geom['x0'], geom['x1'])
        lxmax = max(geom['x0'], geom['x1'])
        lymin = min(geom['y0'], geom['y1'])
        lymax = max(geom['y0'], geom['y1'])
        query_dict = {}
        query_dict['pickup_latitude'] = [lambda x : (x >= lymin) & (x <= lymax)]
        query_dict['pickup_longitude'] = [lambda x : (x >= lxmin) & (x <= lxmax)]
        obj = ds.query(query_dict)
        self.pickup_ar_plot_source.filter_url = obj.data_url
        self.dropoff_ar_plot_source.filter_url = obj.data_url

    def dropoff_selector(self, obj, attrname, old, new):
        geom = new['data_geometry']
        if geom is None:
            self.pickup_ar_plot_source.filter_url = None;
            self.dropoff_ar_plot_source.filter_url = None;
            return
        print 'dropoff**', geom
        lxmin = min(geom['x0'], geom['x1'])
        lxmax = max(geom['x0'], geom['x1'])
        lymin = min(geom['y0'], geom['y1'])
        lymax = max(geom['y0'], geom['y1'])
        query_dict = {}
        query_dict['dropoff_latitude'] = [lambda x : (x >= lymin) & (x <= lymax)]
        query_dict['dropoff_longitude'] = [lambda x : (x >= lxmin) & (x <= lxmax)]
        obj = ds.query(query_dict)
        self.pickup_ar_plot_source.filter_url = obj.data_url
        self.dropoff_ar_plot_source.filter_url = obj.data_url

    def setup_events(self):
        if self.pickup_ar_plot_source:
            self.pickup_ar_plot_source.on_change('selector', self, 'pickup_selector')
        if self.dropoff_ar_plot_source:
            self.dropoff_ar_plot_source.on_change('selector', self, 'dropoff_selector')

from partition import ARDataset
ds = ARDataset()
def get_data(pickup, local_bounds, filters):
    if pickup:
        xfield = 'pickup_longitude'
        yfield = 'pickup_latitude'
    else:
        xfield = 'dropoff_longitude'
        yfield = 'dropoff_latitude'
    st = time.time()
    (grid_shape, results) = ds.project(
        local_bounds, xfield, yfield, filters
    )
    md = time.time()
    print 'PROJECT', md-st
    data = ds.aggregate(results, grid_shape).obj()
    ed = time.time()
    print 'GRID', ed-md
    data = data.T[:]
    data = data ** 0.2
    return data
