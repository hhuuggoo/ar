window.ar_data_source = {}
ar_data_source = window.ar_data_source
ar_data_source.main = () ->
  _ = Bokeh._
  logger = Bokeh.logger
  Backbone = Bokeh.Backbone
  HasProperties = Bokeh.HasProperties
  Range1d = Bokeh.Range1d
  #maybe generalize to ar data source later?
  ajax_throttle = (func) ->
    busy = false
    resp = null
    has_callback = false
    callback = () ->
      if busy
        if has_callback
          logger.debug('already bound, ignoring')
        else
          logger.debug('busy, so doing it later')
          has_callback = true
          resp.done(() ->
            has_callback = false
            callback()
          )
      else
        logger.debug('executing')
        busy = true
        resp = func()
        resp.done(() ->
          logger.debug('done, setting to false')
          busy = false
          resp = null
        )

    return callback

  class ARDataSource extends HasProperties
    type: 'ARDataSource'
    update : (column_data_source, renderer_view)  ->
      resp = @_update(column_data_source, renderer_view, true)
      resp.done (() =>
        @subscribe(column_data_source, renderer_view)
      )
    _update : (column_data_source, renderer_view, initial) ->
      if not initial
        xmin = renderer_view.plot_view.x_range.get('start')
        xmax = renderer_view.plot_view.x_range.get('end')
        ymin = renderer_view.plot_view.y_range.get('start')
        ymax = renderer_view.plot_view.y_range.get('end')
        bounds = {
          'xmin' : xmin,
          'ymin' : ymin,
          'xmax' : xmax,
          'ymax' : ymax
        }
      else
        bounds = {}
      resp = $.ajax(
        dataType: 'json'
        url : @get('data_url')
        data : JSON.stringify(bounds)
        xhrField :
          withCredentials : true
        method : 'POST'
        contentType : 'application/json'
      ).done((data) =>
        @set_data(data, column_data_source)
      )
    subscribe : (column_data_source, renderer_view) ->
      pv = renderer_view.plot_view
      callback = ajax_throttle(
        () =>
          console.log('UPDATE');
          return @_update(column_data_source, renderer_view)
      )
      @listenTo(pv.x_range, 'change', callback)
      @listenTo(pv.y_range, 'change', callback)

    set_data : (data, column_data_source) ->
      orig_data = _.clone(column_data_source.get('data'))
      _.extend(orig_data, data)
      column_data_source.set('data', orig_data)

  class ARDataSources extends Backbone.Collection
    model: ARDataSource
    defaults:
      url : ""
      expr : null
  coll = new ARDataSources
  Bokeh.Collections.register("ARDataSource", coll)
