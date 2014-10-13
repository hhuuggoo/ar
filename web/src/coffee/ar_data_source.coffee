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
    initialize : (attrs, options) ->
      super(attrs, options)
      @cache = {}
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
        data = {
          'xmin' : xmin,
          'ymin' : ymin,
          'xmax' : xmax,
          'ymax' : ymax
        }
      else
        data = {}
      data['filter_url'] = @get('filter_url')
      resp = $.ajax(
        dataType: 'json'
        url : @get('data_url')
        data : JSON.stringify(data)
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
      callback = _.debounce(callback, 500);
      @listenTo(pv.x_range, 'change', callback)
      @listenTo(pv.y_range, 'change', callback)
      @listenTo(this, 'change:filter_url', callback)

      ## HACK
      @listenTo(column_data_source, 'change:data', () =>
        if column_data_source.get('data').image.length == 0
          column_data_source.set('data', @cache[column_data_source.id])
      )
      # @listenTo(column_data_source, 'select', () =>
      #   geom = column_data_source.get('selector').get('geometry')
      #   xx = [geom['vx0'], geom['vx1']]
      #   yy = [geom['vy0'], geom['vy1']]
      #   bounds = pv.map_from_screen(xx, yy, 'data')
      #   x_bounds = bounds[0]
      #   y_bounds = bounds[1]
      #   @save('selector', {'data_geometry' : {
      #     'x0' : x_bounds[0], 'x1' : x_bounds[1],
      #     'y0' : y_bounds[0], 'y1' : y_bounds[1]
      #   }})
      # )
      @listenTo(column_data_source, 'deselect', () =>
        @save('selector', {'data_geometry' : null})
      )
    set_data : (data, column_data_source) ->
      orig_data = _.clone(column_data_source.get('data'))
      _.extend(orig_data, data)
      column_data_source.set('data', orig_data)
      @cache[column_data_source.id] = orig_data

  class ARDataSources extends Backbone.Collection
    model: ARDataSource
    defaults:
      url : ""
      expr : null
  coll = new ARDataSources
  Bokeh.Collections.register("ARDataSource", coll)
