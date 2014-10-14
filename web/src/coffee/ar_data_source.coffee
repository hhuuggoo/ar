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
  class HistogramDataSource extends HasProperties
    type: 'HistogramDataSource'
    initialize : (attrs, options) ->
      super(attrs, options)
      @cache = {}
    update : (column_data_source, renderer_view)  ->
      resp = @_update(column_data_source, renderer_view, true)
      resp.done (() =>
        @subscribe(column_data_source, renderer_view)
      )
    _update : (column_data_source, renderer_view) ->
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
        @set_data(data, column_data_source, renderer_view)
      )
    subscribe : (column_data_source, renderer_view) ->
      pv = renderer_view.plot_view
      callback = ajax_throttle(
        () =>
          console.log('UPDATE HIST');
          return @_update(column_data_source, renderer_view)
      )
      callback = _.debounce(callback, 100);
      @listenTo(this, 'change:filter_url', callback)
      ## HACK
      @listenTo(column_data_source, 'change:data', () =>
        if column_data_source.get('data').counts.length == 0
          column_data_source.set('data', @cache[column_data_source.id])
      )
    set_data : (data, column_data_source, renderer_view) ->
      if data.y_bounds?
        y_bounds = data.y_bounds
        delete data['y_bounds']
        renderer_view.plot_view.y_range.set({
          'start' : y_bounds[0],
          'end' : y_bounds[1],
        })
      orig_data = _.clone(column_data_source.get('data'))
      _.extend(orig_data, data)
      column_data_source.set('data', orig_data)
      @cache[column_data_source.id] = orig_data
  class HistogramDataSources extends Backbone.Collection
    model: HistogramDataSource
    defaults:
      url : ""
      expr : null
  coll = new HistogramDataSources
  Bokeh.Collections.register("HistogramDataSource", coll)

  class ARDataSource extends HasProperties
    type: 'ARDataSource'
    initialize : (attrs, options) ->
      super(attrs, options)
      @cache = {}
      @initial = true
    update : (column_data_source, renderer_view)  ->
      @_update = ajax_throttle(
        () =>
          console.log('UPDATE');
          return @_do_update(column_data_source, renderer_view)
      )
      @_update = _.debounce(@_update, 100);
      resp = @_update(column_data_source, renderer_view)
    _do_update : (column_data_source, renderer_view) ->
      if not @initial
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
      url = @get('data_url')
      data['filter_url'] = @get('filter_url')
      st = Number(new Date())
      console.log('ajax', st);
      resp = $.ajax(
        dataType: 'json'
        url : url
        data : JSON.stringify(data)
        xhrField :
          withCredentials : true
        method : 'POST'
        contentType : 'application/json'
      ).done((data) =>
        ed = Number(new Date())
        console.log('ajax done', ed-st)
        @set_data(data, column_data_source)
        if @initial
          @subscribe(column_data_source, renderer_view)
        @initial = false
      )
      return resp
    subscribe : (column_data_source, renderer_view) ->
      pv = renderer_view.plot_view
      @listenTo(pv.x_range, 'change', @_update)
      @listenTo(pv.y_range, 'change', @_update)
      @listenTo(this, 'change:filter_url', @_update)

      ## HACK
      @listenTo(column_data_source, 'change:data', () =>
        if column_data_source.get('data').image.length == 0
          column_data_source.set('data', @cache[column_data_source.id])
      )
    set_data : (data, column_data_source) ->
      console.log('setting palette', data.palette)
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
