import time
import datetime as dt

from bokeh.objects import  ServerDataSource, Plot
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
    url = String()

class TaxiApp(VBox):
    extra_generated_classes = [["TaxiApp", "TaxiApp", "VBox"]]
    extra_scripts = ['/bokehjs/static/app/src/js/ar_data_source.js']
    extra_js=['window.ar_data_source.main();']
    pickup_plot = Instance(Plot)
    dropoff_plot = Instance(Plot)
    gbounds = (-74.05, -73.75, 40.5, 40.99)
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
        plot = image(source=data,
                     image="image",
                     x="x",
                     y="y",
                     dw="dw",
                     dh="dh",
                     plot_width=600,
                     plot_height=600,
                     palette='palette',
                     x_range=[xmin, xmax], y_range=[ymin, ymax],
                     tools="pan,wheel_zoom,box_zoom,reset,previewsave",
                     title='pickup'
        )
        app.pickup_plot = plot
        data = ARDataSource(
            data_url="/bokeh/taxidata/dropoff/",
            data=dict(
                x=[0], y=[0], dw=[xmax-xmin], dh=[ymax-ymin], palette=["Greys-256"]
            )
        )
        plot = image(source=data,
                     image="image",
                     plot_width=600,
                     plot_height=600,
                     x="x",
                     y="y",
                     dw="dw",
                     dh="dh",
                     palette='palette',
                     x_range=[xmin, xmax], y_range=[ymin, ymax],
                     tools="pan,wheel_zoom,box_zoom,reset,previewsave",
                     title='dropoff'
        )
        app.dropoff_plot = plot
        app.children = [app.pickup_plot, app.dropoff_plot]
        return app
        
from partition import ARDataset, KSXChunkedGrid
ds = ARDataset()
def get_data(pickup, local_bounds, filters):
    if pickup:
        xfield = 'pickup_longitude'
        yfield = 'pickup_latitude'
    else:
        xfield = 'dropoff_longitude'
        yfield = 'dropoff_latitude'
    st = time.time()
    local_indexes, (grid_shape, results) = ds.project(
        local_bounds, xfield, yfield, filters
    )
    ed = time.time()
    print 'PROJECT', ed-st
    lxdim1, lxdim2, lydim1, lydim2 = local_indexes
    print 'LINDX', lxdim1, lxdim2, lydim1, lydim2
    st = time.time()
    grid = KSXChunkedGrid(results, grid_shape[-1])
    data = grid.get(lxdim1, lxdim2, lydim1, lydim2)
    ed = time.time()
    print 'GRID', ed-st
    data = data.T[:]
    data = data ** 0.2
    return data

