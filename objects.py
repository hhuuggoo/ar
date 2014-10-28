import time
import datetime as dt
import numpy as np
import pandas as pd
from kitchensink import setup_client, client, do, du, dp

from bokeh.objects import  ServerDataSource, Plot, ColumnDataSource, Range1d
from bokeh.widgets import (HBox, VBox, VBoxForm,
                           DateRangeSlider, Paragraph, Select,
                           MultiSelect)

from bokeh.plot_object import PlotObject
from bokeh.crossfilter.plotting import make_histogram
from bokeh.plotting import figure, hold, rect
from bokeh.plotting_helpers import _get_select_tool
from bokeh.properties import (
    Datetime, HasProps, Dict, Enum, Either, Float, Instance, Int,
    List, String, Color, Include, Bool, Tuple, Any
)
from partition import ARDataset
ds = ARDataset()

from bokeh.plotting import image

class AjaxDataSource(ServerDataSource):
    url = String()

class ARDataSource(ServerDataSource):
    url = String()
    filter_url = String()

class HistogramDataSource(ServerDataSource):
    url = String()
    filter_url = String()

class TaxiApp(HBox):
    extra_generated_classes = [["TaxiApp", "TaxiApp", "HBox"]]
    extra_scripts = ['/bokehjs/static/app/src/js/ar_data_source.js']
    extra_js=['window.ar_data_source.main();']
    gbounds = ds.gbounds

    pickup_plot = Instance(Plot)
    pickup_raw_plot_source = Instance(ColumnDataSource)
    pickup_ar_plot_source = Instance(ARDataSource)

    dropoff_plot = Instance(Plot)
    dropoff_raw_plot_source = Instance(ColumnDataSource)
    dropoff_ar_plot_source = Instance(ARDataSource)

    pickup_comparison_plot = Instance(Plot)
    pickup_comparison_raw_plot_source = Instance(ColumnDataSource)
    pickup_comparison_ar_plot_source = Instance(ARDataSource)

    dropoff_comparison_plot = Instance(Plot)
    dropoff_comparison_raw_plot_source = Instance(ColumnDataSource)
    dropoff_comparison_ar_plot_source = Instance(ARDataSource)

    trip_distance_source = Instance(ColumnDataSource)
    trip_time_source = Instance(ColumnDataSource)
    trip_distance_ar_source = Instance(HistogramDataSource)
    trip_time_ar_source = Instance(HistogramDataSource)

    widgets = Instance(VBox)
    date_slider = Instance(DateRangeSlider)
    filters = Dict(String, Any)
    trip_time_bins = np.linspace(0, 3600, 25)
    trip_distance_bins = np.linspace(0.01, 20, 25)
    distance_histogram = Instance(Plot)
    time_histogram = Instance(Plot)
    hour_selector = Instance(Select)
    day_of_week_selector = Instance(Select)

    regular = Instance(HBox)
    filtered = Instance(HBox)
    images = Instance(VBox)

    def make_trip_distance_histogram(self):
        bins = self.trip_distance_bins
        centers = pd.rolling_mean(bins, 2)[1:]
        figure(title="trip distance in miles",
               title_text_font='12pt',
               plot_width=300,
               plot_height=200,
               x_range=[bins[0], bins[-1]],
               y_range=[0, 1],
               tools="pan,wheel_zoom,box_zoom,select,reset"
        )
        source = HistogramDataSource(
            data_url="/bokeh/taxidata/distancehist/",
        )
        hold()
        plot = rect("centers", "y", np.mean(np.diff(centers)) * 0.7, "counts",
                    source=source)
        self.trip_distance_source = plot.select({'type' : ColumnDataSource})[0]
        self.trip_distance_ar_source = source
        plot.min_border=0
        plot.h_symmetry=False
        plot.v_symmetry=False
        select_tool = _get_select_tool(plot)
        if select_tool:
            select_tool.dimensions = ['width']
        self.distance_histogram = plot

    def make_trip_time_histogram(self):
        bins = self.trip_time_bins
        centers = pd.rolling_mean(bins, 2)[1:]
        figure(title="trip time in secs",
               title_text_font='12pt',
               plot_width=300,
               plot_height=200,
               x_range=[bins[0], bins[-1]],
               y_range=[0, 1],
               tools="pan,wheel_zoom,box_zoom,select,reset"
        )
        source = HistogramDataSource(
            data_url="/bokeh/taxidata/timehist/",
        )
        hold()
        plot = rect("centers", "y", np.mean(np.diff(centers)) * 0.7, "counts",
                    source=source)
        self.trip_time_source = plot.select({'type' : ColumnDataSource})[0]
        self.trip_time_ar_source = source
        plot.min_border=0
        plot.h_symmetry=False
        plot.v_symmetry=False
        select_tool = _get_select_tool(plot)
        if select_tool:
            select_tool.dimensions = ['width']
        self.time_histogram = plot

    def update_filters(self, obj, attrname, old, new):
        ##hack - only call this once per req/rep cycle
        from flask import request
        if hasattr(request, 'filters_updated'):
            return
        if not self.trip_time_source.data_geometry:
            self.filters.pop('trip_time_in_secs', None)
        else:
            geom = self.trip_time_source.data_geometry
            lxmin = min(geom['x0'], geom['x1'])
            lxmax = max(geom['x0'], geom['x1'])
            self.filters['trip_time_in_secs'] = [lxmin, lxmax]
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

        # if not self.pickup_comparison_raw_plot_source.data_geometry:
        #     self.filters.pop('pickup_latitude', None)
        #     self.filters.pop('pickup_longitude', None)
        # else:
        #     geom = self.pickup_comparison_raw_plot_source.data_geometry
        #     lxmin = min(geom['x0'], geom['x1'])
        #     lxmax = max(geom['x0'], geom['x1'])
        #     lymin = min(geom['y0'], geom['y1'])
        #     lymax = max(geom['y0'], geom['y1'])
        #     self.filters['pickup_latitude'] = [lymin, lymax]
        #     self.filters['pickup_longitude'] = [lxmin, lxmax]
        # if not self.dropoff_comparison_raw_plot_source.data_geometry:
        #     self.filters.pop('dropoff_latitude', None)
        #     self.filters.pop('dropoff_longitude', None)
        # else:
        #     geom = self.dropoff_comparison_raw_plot_source.data_geometry
        #     lxmin = min(geom['x0'], geom['x1'])
        #     lxmax = max(geom['x0'], geom['x1'])
        #     lymin = min(geom['y0'], geom['y1'])
        #     lymax = max(geom['y0'], geom['y1'])
        #     self.filters['dropoff_latitude'] = [lymin, lymax]
        #     self.filters['dropoff_longitude'] = [lxmin, lxmax]

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
                     plot_height=400,
                     palette='palette',
                     x_range=[xmin, xmax], y_range=[ymin, ymax],
                     tools="pan,wheel_zoom,box_zoom,select,reset",
                     title='pickup'
        )
        plot.title_text_font='12pt'
        app.pickup_plot = plot
        app.pickup_raw_plot_source = plot.select({'type' : ColumnDataSource})[0]

        data = ARDataSource(
            data_url="/bokeh/taxidatavsregular/pickup/",
            data=dict(
                x=[0], y=[0], dw=[xmax-xmin], dh=[ymax-ymin], palette=["Greys-256"]
            )
        )
        app.pickup_comparison_ar_plot_source = data
        plot = image(source=data,
                     image="image",
                     x="x",
                     y="y",
                     dw="dw",
                     dh="dh",
                     plot_width=400,
                     plot_height=400,
                     palette='palette',
                     x_range=[xmin, xmax], y_range=[ymin, ymax],
                     tools="pan,wheel_zoom,box_zoom,select,reset",
                     title='pickup comparison plot'
        )
        plot.title_text_font='12pt'
        app.pickup_comparison_plot = plot
        app.pickup_comparison_raw_plot_source = plot.select({'type' : ColumnDataSource})[0]
        data = ARDataSource(
            data_url="/bokeh/taxidatavsregular/dropoff/",
            data=dict(
                x=[0], y=[0], dw=[xmax-xmin], dh=[ymax-ymin], palette=["Greys-256"]
            )
        )
        app.dropoff_comparison_ar_plot_source = data
        plot = image(source=data,
                     image="image",
                     x="x",
                     y="y",
                     dw="dw",
                     dh="dh",
                     plot_width=400,
                     plot_height=400,
                     palette='palette',
                     x_range=[xmin, xmax], y_range=[ymin, ymax],
                     tools="pan,wheel_zoom,box_zoom,select,reset",
                     title='dropoff comparison plot'
        )
        plot.title_text_font='12pt'
        app.dropoff_comparison_plot = plot
        app.dropoff_comparison_raw_plot_source = plot.select({'type' : ColumnDataSource})[0]

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
                     plot_height=400,
                     x="x",
                     y="y",
                     dw="dw",
                     dh="dh",
                     palette='palette',
                     x_range=[xmin, xmax], y_range=[ymin, ymax],
                     tools="pan,wheel_zoom,box_zoom,reset,select,reset",
                     title='dropoff'
        )
        plot.title_text_font='12pt'
        app.dropoff_plot = plot
        app.dropoff_raw_plot_source = plot.select({'type' : ColumnDataSource})[0]
        app.make_trip_distance_histogram()
        app.make_trip_time_histogram()
        app.widgets = VBoxForm()
        app.day_of_week_selector = Select.create(
            options=["-----", 'Weekday', 'Friday/Saturday/Sunday', 'Saturday/Sunday'],
            name='Day Of Week'
        )
        app.date_slider = DateRangeSlider(value=(dt.datetime(2012, 1, 1),
                                                 dt.datetime(2013, 1, 28)),
                                          bounds=(dt.datetime(2012, 12, 31),
                                                  dt.datetime(2013, 1, 31)),
                                          step={'days' : 1},
                                          range=({'days' : 1},{'days':30}),
                                          name='period',
                                          title='period'
        )
        app.hour_selector = Select.create(options=["-----",
                                                   '8am-12pm',
                                                   '12pm-4pm',
                                                   '4pm-8pm',
                                                   '8pm-12am',
                                                   '12am-4am'],
                                          name='Hour of the Day'
        )
        title = Paragraph(text="NYC Taxi Cab Data", width=250, height=50)
        app.widgets.children=[title, app.date_slider,
                              Paragraph(width=250, height=10),
                              app.hour_selector,
                              app.day_of_week_selector,
                              Paragraph(width=250, height=10),
                              app.distance_histogram,
                              Paragraph(text="",
                                        width=250, height=50),
                              app.time_histogram]
        app.images = VBox()
        app.regular = HBox()
        app.filtered = HBox()
        app.regular.children = [app.pickup_plot, app.dropoff_plot]
        app.filtered.children = [app.pickup_comparison_plot,
                                  app.dropoff_comparison_plot]
        app.images.children = [app.regular]
        app.children = [app.widgets, app.images]
        return app

    def set_images(self):
        if self.pickup_ar_plot_source.filter_url:
            self.images.children = [self.regular, self.filtered]
        else:
            self.images.children = [self.regular]

    def filter(self):
        st = time.time()
        query_dict = {}
        def selector(minval, maxval):
            return lambda x : (x >= minval) & (x <= maxval)

        def in1d(data):
            return lambda x : np.in1d(x, data)

        for k,v in self.filters.items():
            if k in {'pickup_datetime', 'pickup_latitude',
                     'pickup_longitude',
                     'dropoff_latitude', 'dropoff_longitude',
                     'trip_distance', 'trip_time_in_secs',
                     'hour_of_day',
            }:
                minval = min(v)
                maxval = max(v)
                query_dict[k] = [selector(minval, maxval)]
            if k in {'day_of_week'}:
                query_dict[k] = [in1d(v)]

        if len(query_dict) == 0:
            self.pickup_ar_plot_source.filter_url = None
            self.dropoff_ar_plot_source.filter_url = None
            self.trip_time_ar_source.filter_url = None
            self.trip_distance_ar_source.filter_url = None
            self.pickup_comparison_ar_plot_source.filter_url = None
            self.dropoff_comparison_ar_plot_source.filter_url = None
            self.set_images()
            return
        print query_dict
        obj = ds.query(query_dict)
        self.pickup_ar_plot_source.filter_url = obj.data_url
        self.dropoff_ar_plot_source.filter_url = obj.data_url
        self.trip_time_ar_source.filter_url = obj.data_url
        self.trip_distance_ar_source.filter_url = obj.data_url
        self.pickup_comparison_ar_plot_source.filter_url = obj.data_url
        self.dropoff_comparison_ar_plot_source.filter_url = obj.data_url
        self.set_images()
        ed = time.time()
        print 'FILTERING', ed-st



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

    def hour_change(self, obj, attrname, old, new):
        if new == "8am-12pm":
            self.filters['hour_of_day'] = [8,12]
        elif new == "12pm-4pm":
            self.filters['hour_of_day'] = [12,16]
        elif new == "4pm-8pm":
            self.filters['hour_of_day'] = [16,20]
        elif new == "8pm-12am":
            self.filters['hour_of_day'] = [20,24]
        elif new == "12am-4am":
            self.filters['hour_of_day'] = [0,4]
        elif new == "4am-8am":
            self.filters['hour_of_day'] = [4,8]
        else:
            self.filters.pop('hour_of_day')
        self._dirty = True
        self.filter()

    def day_of_week_change(self, obj, attrname, old, new):
        mapping = dict(
            Monday=0,
            Tuesday=1,
            Wednesday=2,
            Thursday=3,
            Friday=4,
            Saturday=5,
            Sunday=6
        )
        if new == 'Weekday':
            self.filters['day_of_week'] = [0,1,2,3,4]
        elif new == 'Friday/Saturday/Sunday':
            self.filters['day_of_week'] = [4,5,6]
        elif new == 'Saturday/Sunday':
            self.filters['day_of_week'] = [5,6]
        else:
            self.filters.pop('day_of_week')
        self._dirty = True
        self.filter()
    def setup_events(self):
        if self.hour_selector:
            self.hour_selector.on_change('value', self, 'hour_change')
        if self.day_of_week_selector:
            self.day_of_week_selector.on_change('value', self, 'day_of_week_change')
        if self.pickup_raw_plot_source:
            self.pickup_raw_plot_source.on_change('data_geometry',
                                                  self, 'update_filters')
        if self.dropoff_raw_plot_source:
            self.dropoff_raw_plot_source.on_change('data_geometry',
                                                   self, 'update_filters')
        if self.pickup_comparison_raw_plot_source:
            self.pickup_comparison_raw_plot_source.on_change('data_geometry',
                                                  self, 'update_filters')
        if self.dropoff_comparison_raw_plot_source:
            self.dropoff_comparison_raw_plot_source.on_change('data_geometry',
                                                   self, 'update_filters')

        if self.trip_distance_source:
            self.trip_distance_source.on_change('data_geometry', self,
                                                'update_filters')
        if self.trip_time_source:
            self.trip_time_source.on_change('data_geometry', self,
                                            'update_filters')
def get_data(pickup, local_bounds, filters):
    if pickup:
        xfield = 'pickup_longitude'
        yfield = 'pickup_latitude'
    else:
        xfield = 'dropoff_longitude'
        yfield = 'dropoff_latitude'
    st = time.time()
    data = ds.project(
        local_bounds, xfield, yfield, filters
    )
    data = data.T[:]
    return data

trip_time_bins = np.linspace(0, 3600, 25)
def get_time_histogram(filters):
    c = ds.histogram('trip_time_in_secs', trip_time_bins, filters=filters)
    counts = ds.finish_histogram(c.br(profile='time_histogram'))
    centers = pd.rolling_mean(trip_time_bins, 2)[1:]
    data={'counts': counts.tolist(),
          'centers': centers.tolist(),
          'y' : (counts/2.0).tolist()}
    return data

trip_distance_bins = np.linspace(0.01, 20, 25)
def get_distance_histogram(filters):
    c = ds.histogram('trip_distance', trip_distance_bins, filters=filters)
    counts = ds.finish_histogram(c.br(profile='distance_histogram'))
    centers = pd.rolling_mean(trip_distance_bins, 2)[1:]
    data={'counts': counts.tolist(),
          'centers': centers.tolist(),
          'y' : (counts/2.0).tolist()}
    return data
