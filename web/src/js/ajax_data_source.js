(function() {
  var ajax_data_source,
    __hasProp = {}.hasOwnProperty,
    __extends = function(child, parent) { for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; };

  window.ajax_data_source = {};

  ajax_data_source = window.ajax_data_source;

  ajax_data_source.main = function() {
    var AjaxDataSource, AjaxDataSources, Backbone, HasProperties, Range1d, coll, _;
    _ = Bokeh._;
    Backbone = Bokeh.Backbone;
    HasProperties = Bokeh.HasProperties;
    Range1d = Bokeh.Range1d;
    AjaxDataSource = (function(_super) {
      __extends(AjaxDataSource, _super);

      function AjaxDataSource() {
        return AjaxDataSource.__super__.constructor.apply(this, arguments);
      }

      AjaxDataSource.prototype.type = 'AjaxDataSource';

      AjaxDataSource.prototype.update = function(column_data_source, renderer_view) {
        return $.ajax({
          dataType: 'json',
          url: this.get('data_url'),
          xhrField: {
            withCredentials: true
          },
          method: 'POST',
          contentType: 'application/json'
        }).done((function(_this) {
          return function(data) {
            return _this.set_data(data, column_data_source);
          };
        })(this));
      };

      AjaxDataSource.prototype.set_data = function(data, column_data_source) {
        var orig_data;
        orig_data = _.clone(column_data_source.get('data'));
        _.extend(orig_data, data);
        return column_data_source.set('data', orig_data);
      };

      return AjaxDataSource;

    })(HasProperties);
    AjaxDataSources = (function(_super) {
      __extends(AjaxDataSources, _super);

      function AjaxDataSources() {
        return AjaxDataSources.__super__.constructor.apply(this, arguments);
      }

      AjaxDataSources.prototype.model = AjaxDataSource;

      AjaxDataSources.prototype.defaults = {
        url: "",
        expr: null
      };

      return AjaxDataSources;

    })(Backbone.Collection);
    coll = new AjaxDataSources;
    return Bokeh.Collections.register("AjaxDataSource", coll);
  };

}).call(this);
