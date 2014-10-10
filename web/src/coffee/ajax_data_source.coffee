window.ajax_data_source = {}
ajax_data_source = window.ajax_data_source
ajax_data_source.main = () ->
  _ = Bokeh._
  Backbone = Bokeh.Backbone
  HasProperties = Bokeh.HasProperties
  Range1d = Bokeh.Range1d
  #maybe generalize to ajax data source later?
  class AjaxDataSource extends HasProperties
    type: 'AjaxDataSource'

    update : (column_data_source, renderer_view) ->
      $.ajax(
        dataType: 'json'
        url : @get('data_url')
        xhrField :
          withCredentials : true
        method : 'POST'
        contentType : 'application/json'
      ).done((data) =>
        @set_data(data, column_data_source)
      )
    set_data : (data, column_data_source) ->
      orig_data = _.clone(column_data_source.get('data'))
      _.extend(orig_data, data)
      column_data_source.set('data', orig_data)

  class AjaxDataSources extends Backbone.Collection
    model: AjaxDataSource
    defaults:
      url : ""
      expr : null
  coll = new AjaxDataSources
  Bokeh.Collections.register("AjaxDataSource", coll)
