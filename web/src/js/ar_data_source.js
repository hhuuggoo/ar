(function() {
  var ar_data_source,
    __hasProp = {}.hasOwnProperty,
    __extends = function(child, parent) { for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; };

  window.ar_data_source = {};

  ar_data_source = window.ar_data_source;

  ar_data_source.main = function() {
    var ARDataSource, ARDataSources, Backbone, HasProperties, HistogramDataSource, HistogramDataSources, Range1d, ajax_throttle, coll, logger, _;
    _ = Bokeh._;
    logger = Bokeh.logger;
    Backbone = Bokeh.Backbone;
    HasProperties = Bokeh.HasProperties;
    Range1d = Bokeh.Range1d;
    ajax_throttle = function(func) {
      var busy, callback, has_callback, resp;
      busy = false;
      resp = null;
      has_callback = false;
      callback = function() {
        if (busy) {
          if (has_callback) {
            return logger.debug('already bound, ignoring');
          } else {
            logger.debug('busy, so doing it later');
            has_callback = true;
            return resp.done(function() {
              has_callback = false;
              return callback();
            });
          }
        } else {
          logger.debug('executing');
          busy = true;
          resp = func();
          return resp.done(function() {
            logger.debug('done, setting to false');
            busy = false;
            return resp = null;
          });
        }
      };
      return callback;
    };
    HistogramDataSource = (function(_super) {
      __extends(HistogramDataSource, _super);

      function HistogramDataSource() {
        return HistogramDataSource.__super__.constructor.apply(this, arguments);
      }

      HistogramDataSource.prototype.type = 'HistogramDataSource';

      HistogramDataSource.prototype.initialize = function(attrs, options) {
        HistogramDataSource.__super__.initialize.call(this, attrs, options);
        return this.cache = {};
      };

      HistogramDataSource.prototype.update = function(column_data_source, renderer_view) {
        var resp;
        resp = this._update(column_data_source, renderer_view, true);
        return resp.done(((function(_this) {
          return function() {
            return _this.subscribe(column_data_source, renderer_view);
          };
        })(this)));
      };

      HistogramDataSource.prototype._update = function(column_data_source, renderer_view) {
        var data, resp;
        data = {};
        data['filter_url'] = this.get('filter_url');
        return resp = $.ajax({
          dataType: 'json',
          url: this.get('data_url'),
          data: JSON.stringify(data),
          xhrField: {
            withCredentials: true
          },
          method: 'POST',
          contentType: 'application/json'
        }).done((function(_this) {
          return function(data) {
            return _this.set_data(data, column_data_source, renderer_view);
          };
        })(this));
      };

      HistogramDataSource.prototype.subscribe = function(column_data_source, renderer_view) {
        var callback, pv;
        pv = renderer_view.plot_view;
        callback = ajax_throttle((function(_this) {
          return function() {
            console.log('UPDATE HIST');
            return _this._update(column_data_source, renderer_view);
          };
        })(this));
        callback = _.debounce(callback, 100);
        this.listenTo(this, 'change:filter_url', callback);
        return this.listenTo(column_data_source, 'change:data', (function(_this) {
          return function() {
            if (column_data_source.get('data').counts.length === 0) {
              return column_data_source.set('data', _this.cache[column_data_source.id]);
            }
          };
        })(this));
      };

      HistogramDataSource.prototype.set_data = function(data, column_data_source, renderer_view) {
        var orig_data, y_bounds;
        if (data.y_bounds != null) {
          y_bounds = data.y_bounds;
          delete data['y_bounds'];
          renderer_view.plot_view.y_range.set({
            'start': y_bounds[0],
            'end': y_bounds[1]
          });
        }
        orig_data = _.clone(column_data_source.get('data'));
        _.extend(orig_data, data);
        column_data_source.set('data', orig_data);
        return this.cache[column_data_source.id] = orig_data;
      };

      return HistogramDataSource;

    })(HasProperties);
    HistogramDataSources = (function(_super) {
      __extends(HistogramDataSources, _super);

      function HistogramDataSources() {
        return HistogramDataSources.__super__.constructor.apply(this, arguments);
      }

      HistogramDataSources.prototype.model = HistogramDataSource;

      HistogramDataSources.prototype.defaults = {
        url: "",
        expr: null
      };

      return HistogramDataSources;

    })(Backbone.Collection);
    coll = new HistogramDataSources;
    Bokeh.Collections.register("HistogramDataSource", coll);
    ARDataSource = (function(_super) {
      __extends(ARDataSource, _super);

      function ARDataSource() {
        return ARDataSource.__super__.constructor.apply(this, arguments);
      }

      ARDataSource.prototype.type = 'ARDataSource';

      ARDataSource.prototype.initialize = function(attrs, options) {
        ARDataSource.__super__.initialize.call(this, attrs, options);
        this.cache = {};
        return this.initial = true;
      };

      ARDataSource.prototype.update = function(column_data_source, renderer_view) {
        var resp;
        this._update = ajax_throttle((function(_this) {
          return function() {
            console.log('UPDATE');
            return _this._do_update(column_data_source, renderer_view);
          };
        })(this));
        this._update = _.debounce(this._update, 100);
        return resp = this._update(column_data_source, renderer_view);
      };

      ARDataSource.prototype._do_update = function(column_data_source, renderer_view) {
        var data, resp, st, url, xmax, xmin, ymax, ymin;
        if (!this.initial) {
          xmin = renderer_view.plot_view.x_range.get('start');
          xmax = renderer_view.plot_view.x_range.get('end');
          ymin = renderer_view.plot_view.y_range.get('start');
          ymax = renderer_view.plot_view.y_range.get('end');
          data = {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
          };
        } else {
          data = {};
        }
        url = this.get('data_url');
        data['filter_url'] = this.get('filter_url');
        st = Number(new Date());
        console.log('ajax', st);
        resp = $.ajax({
          dataType: 'json',
          url: url,
          data: JSON.stringify(data),
          xhrField: {
            withCredentials: true
          },
          method: 'POST',
          contentType: 'application/json'
        }).done((function(_this) {
          return function(data) {
            var ed;
            ed = Number(new Date());
            console.log('ajax done', ed - st);
            _this.set_data(data, column_data_source);
            if (_this.initial) {
              _this.subscribe(column_data_source, renderer_view);
            }
            return _this.initial = false;
          };
        })(this));
        return resp;
      };

      ARDataSource.prototype.subscribe = function(column_data_source, renderer_view) {
        var pv;
        pv = renderer_view.plot_view;
        this.listenTo(pv.x_range, 'change', this._update);
        this.listenTo(pv.y_range, 'change', this._update);
        this.listenTo(this, 'change:filter_url', this._update);
        return this.listenTo(column_data_source, 'change:data', (function(_this) {
          return function() {
            if (column_data_source.get('data').image.length === 0) {
              return column_data_source.set('data', _this.cache[column_data_source.id]);
            }
          };
        })(this));
      };

      ARDataSource.prototype.set_data = function(data, column_data_source) {
        var orig_data;
        console.log('setting palette', data.palette);
        orig_data = _.clone(column_data_source.get('data'));
        _.extend(orig_data, data);
        column_data_source.set('data', orig_data);
        return this.cache[column_data_source.id] = orig_data;
      };

      return ARDataSource;

    })(HasProperties);
    ARDataSources = (function(_super) {
      __extends(ARDataSources, _super);

      function ARDataSources() {
        return ARDataSources.__super__.constructor.apply(this, arguments);
      }

      ARDataSources.prototype.model = ARDataSource;

      ARDataSources.prototype.defaults = {
        url: "",
        expr: null
      };

      return ARDataSources;

    })(Backbone.Collection);
    coll = new ARDataSources;
    return Bokeh.Collections.register("ARDataSource", coll);
  };

}).call(this);
