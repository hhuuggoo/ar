import time
import datetime as dt
import numpy as np
import pandas as pd
from kitchensink import setup_client, client, do, du, dp

from bokeh.objects import  ServerDataSource, Plot, ColumnDataSource, Range1d
from bokeh.widgets import HBox, VBox, VBoxForm, DateRangeSlider, Paragraph
from bokeh.plot_object import PlotObject
from bokeh.crossfilter.plotting import make_histogram
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
    trip_distance_source = Instance(ColumnDataSource)
    trip_time_source = Instance(ColumnDataSource)
    widgets = Instance(VBox)
    date_slider = Instance(DateRangeSlider)
    filters = Dict(String, Any)
    trip_time_bins = np.linspace(0, 3600, 25)
    trip_distance_bins = np.linspace(0.01, 10, 25)
    distance_histogram = Instance(Plot)
    time_histogram = Instance(Plot)

    def compute_histograms(self):
        st = time.time()
        filter_url = self.pickup_ar_plot_source.filter_url
        filters = None
        if filter_url:
            filters = du(filter_url)
        c1 = ds.histogram('trip_distance', self.trip_distance_bins, filters=filters)
        c2 = ds.histogram('trip_time_in_secs', self.trip_time_bins, filters=filters)
        hist1 = ds.finish_histogram(c1.br(profile='distance_histogram'))
        hist2 = ds.finish_histogram(c2.br(profile='time_histogram'))
        ed = time.time()
        print 'COMPUTE HIST', ed-st
        return hist1, hist2

    def make_trip_distance_histogram(self, counts):
        bins = self.trip_distance_bins
        centers = pd.rolling_mean(bins, 2)[1:]
        data={'counts': counts.tolist(), 'centers': centers.tolist(), 'y' : (counts/2.0).tolist()}
        if self.trip_distance_source is not None:
            self.trip_distance_source.data = data
            self.distance_histogram.y_range.end = counts.max()
        else:
            source = ColumnDataSource(data=data)
            self.trip_distance_source = source
            self.distance_histogram = make_histogram(
                self.trip_distance_source,
                bar_width=np.mean(np.diff(centers)) * 0.7,
                plot_width=300, plot_height=200, min_border=20,
                tools="pan,wheel_zoom,box_zoom,select,reset")
            self.distance_histogram.title = "trip distance in miles"

    def make_trip_time_histogram(self, counts):
        bins = self.trip_time_bins
        centers = pd.rolling_mean(bins, 2)[1:]
        data={'counts': counts.tolist(), 'centers': centers.tolist(), 'y' : (counts/2.0).tolist()}
        if self.trip_time_source is not None:
            self.trip_time_source.data = data
            self.time_histogram.y_range.end = counts.max()
        else:
            source = ColumnDataSource(data=data)
            self.trip_time_source = source
            self.time_histogram = make_histogram(
                self.trip_time_source,
                bar_width=np.mean(np.diff(centers)) * 0.7,
                plot_width=300, plot_height=200, min_border=20,
                tools="pan,wheel_zoom,box_zoom,select,reset")
            self.time_histogram.title = "trip time in seconds"

    def update_filters(self, obj, attrname, old, new):
        ##hack - only call this once per req/rep cycle
        from flask import request
        if hasattr(request, 'filters_updated'):
            return
        if not self.trip_time_source.data_geometry:
            self.filters.pop('trip_time', None)
        else:
            geom = self.trip_time_source.data_geometry
            lxmin = min(geom['x0'], geom['x1'])
            lxmax = max(geom['x0'], geom['x1'])
            self.filters['trip_time'] = [lxmin, lxmax]
        if not self.trip_distance_source.data_geometry:
            self.filters.pop('trip_distance', None)
        else:
            geom = self.trip_distance_source.data_geometry
            lxmin = min(geom['x0'], geom['x1'])
            lxmax = max(geom['x0'], geom['x1'])
            self.filters['trip_distance'] = [lxmin, lxmax]
        if not self.pickup_raw_plot_source.data_geometry:
            self.filters.pop('pickup_latitude', None)
            self.filters.pop('pickup_longitude', None)
        else:
            geom = self.pickup_raw_plot_source.data_geometry
            lxmin = min(geom['x0'], geom['x1'])
            lxmax = max(geom['x0'], geom['x1'])
            lymin = min(geom['y0'], geom['y1'])
            lymax = max(geom['y0'], geom['y1'])
            self.filters['pickup_latitude'] = [lymin, lymax]
            self.filters['pickup_longitude'] = [lxmin, lxmax]
        if not self.dropoff_raw_plot_source.data_geometry:
            self.filters.pop('dropoff_latitude', None)
            self.filters.pop('dropoff_longitude', None)
        else:
            geom = self.dropoff_raw_plot_source.data_geometry
            lxmin = min(geom['x0'], geom['x1'])
            lxmax = max(geom['x0'], geom['x1'])
            lymin = min(geom['y0'], geom['y1'])
            lymax = max(geom['y0'], geom['y1'])
            self.filters['dropoff_latitude'] = [lymin, lymax]
            self.filters['dropoff_longitude'] = [lxmin, lxmax]
        self._dirty = True
        try:
            request.filters_updated = True
        except RuntimeError:
            pass
        self.filter()

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
        hist1, hist2 = app.compute_histograms()
        app.make_trip_distance_histogram(hist1)
        app.make_trip_time_histogram(hist2)
        app.widgets = VBoxForm()
        app.date_slider = DateRangeSlider(value=(dt.datetime(2012, 1, 1),
                                                 dt.datetime(2013, 1, 28)),
                                          bounds=(dt.datetime(2012, 12, 31),
                                                  dt.datetime(2013, 1, 31)),
                                          step={'days' : 1},
                                          range=({'days' : 1},{'days':30}),
                                          name='period',
                                          title='period'
        )
        title = Paragraph(text="NYC Taxi Cab Data", width=250, height=50)
        app.widgets.children=[title, app.date_slider,
                              app.distance_histogram,
                              Paragraph(text="",
                                        width=250, height=50),
                              app.time_histogram]
        app.children = [app.widgets, app.pickup_plot, app.dropoff_plot]
        return app

    def pickup_selector(self, obj, attrname, old, new):
        if attrname != 'data_geometry':
            return
        geom = new
        if geom is None:
            first = self.filters.pop('pickup_latitude', None)
            second = self.filters.pop('pickup_longitude', None)
            if first or second:
                print 'POPPED PICKUP'
                self.filter()
            return
        lxmin = min(geom['x0'], geom['x1'])
        lxmax = max(geom['x0'], geom['x1'])
        lymin = min(geom['y0'], geom['y1'])
        lymax = max(geom['y0'], geom['y1'])
        self.filters['pickup_latitude'] = [lymin, lymax]
        self.filters['pickup_longitude'] = [lxmin, lxmax]
        self._dirty = True
        self.filter()

    def dropoff_selector(self, obj, attrname, old, new):
        if attrname != 'data_geometry':
            return
        geom = new
        if geom is None:
            first = self.filters.pop('dropoff_latitude', None)
            second = self.filters.pop('dropoff_longitude', None)
            if first or second:
                print 'POPPED DROPOFF'
                self.filter()
            return
        lxmin = min(geom['x0'], geom['x1'])
        lxmax = max(geom['x0'], geom['x1'])
        lymin = min(geom['y0'], geom['y1'])
        lymax = max(geom['y0'], geom['y1'])
        self.filters['dropoff_latitude'] = [lymin, lymax]
        self.filters['dropoff_longitude'] = [lxmin, lxmax]
        self._dirty = True
        self.filter()

    def filter(self):
        st = time.time()
        query_dict = {}
        def selector(minval, maxval):
            return lambda x : (x >= minval) & (x <= maxval)
        for k,v in self.filters.items():
            if k in {'pickup_datetime', 'pickup_latitude',
                     'pickup_longitude',
                     'dropoff_latitude', 'dropoff_longitude',
                     'trip_distance', 'trip_time_in_secs',
            }:
                minval = min(v)
                maxval = max(v)
                query_dict[k] = [selector(minval, maxval)]
        print query_dict
        obj = ds.query(query_dict)
        self.pickup_ar_plot_source.filter_url = obj.data_url
        self.dropoff_ar_plot_source.filter_url = obj.data_url
        ed = time.time()
        print 'FILTERING', ed-st
        hist1, hist2 = self.compute_histograms()
        self.make_trip_distance_histogram(hist1)
        self.make_trip_time_histogram(hist2)


    def date_slider_change(self, obj, attrname, old, new):
        minval = min(new)
        maxval = max(new)
        if isinstance(minval, basestring):
            minval = np.datetime64(minval, 'ns').astype('int64')
        if isinstance(maxval, basestring):
            maxval = np.datetime64(maxval, 'ns').astype('int64')
        self.filters['pickup_datetime'] = [minval, maxval]
        self._dirty = True
        self.filter()

    def setup_events(self):
        if self.pickup_raw_plot_source:
            self.pickup_raw_plot_source.on_change('data_geometry',
                                                  self, 'update_filters')
        if self.dropoff_raw_plot_source:
            self.dropoff_raw_plot_source.on_change('data_geometry',
                                                   self, 'update_filters')
        if self.date_slider:
            self.date_slider.on_change('value', self, 'date_slider_change')
        if self.trip_distance_source:
            self.trip_distance_source.on_change('data_geometry', self,
                                                'update_filters')
        if self.trip_time_source:
            self.trip_time_source.on_change('data_geometry', self,
                                            'update_filters')
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
    data = ds.aggregate(results, grid_shape)
    ed = time.time()
    print 'GRID', ed-md
    data = data.T[:]
    data = data ** 0.2
    return data
