from uuid import uuid4
from functools import lru_cache
from traceback import format_exc

from dash import callback_context
from dash.dependencies import Input, Output, State

from .data import DEFAULT_DATA_DQN, DEFAULT_DATA_DQN_UUID, \
    DEFAULT_DATA_FF, DEFAULT_DATA_FF_UUID, \
    format_data, score_data_helper, total_usage_data_helper, expand
from .layouts.tab1 import build_tab as build_tab1
from .layouts.tab2 import build_tab as build_tab2
from .utils.cache import LRUCache

from .server import app

log = app.logger
cache = LRUCache(100)


@lru_cache(maxsize=100)
def parse_content(contents):
    if contents == '#1':
        raw_data = DEFAULT_DATA_DQN
        name = 'ar2'
    elif contents == '#2':
        raw_data = DEFAULT_DATA_FF
        name = 'ff'
    else:
        content_type, content_string = contents.split(',')
        import base64
        import io
        import pickle
        decoded = base64.b64decode(content_string)
        raw_data = pickle.load(io.BytesIO(decoded))
        try:
            name = raw_data['name']
            raw_data = raw_data['data']
        except (TypeError, KeyError):
            name = None
    df = format_data(raw_data)
    max_intervals = len(raw_data)
    total_usage = total_usage_data_helper(df)
    scores = score_data_helper(df)
    uuid = str(uuid4())
    return uuid, name, df, max_intervals, total_usage, scores


cache.put(DEFAULT_DATA_DQN_UUID, tuple(parse_content('#1')[1:]))
cache.put(DEFAULT_DATA_FF_UUID, tuple(parse_content('#2')[1:]))


@app.callback(
    [
        Output('tab1-card-title-data', 'data'),
        Output('tab1-max-intervals', 'data'),
        Output('tab1-data-frame', 'data'),
    ],
    [Input('tab1-upload-data', 'contents')],
    [
        State('tab1-card-title', 'children'),
        State('tab1-upload-data', 'filename'),
        State('tab1-interval-component', 'disabled'),
        State('tab1-max-intervals', 'data'),
        State('tab1-data-frame', 'data'),
    ]
)
def parse_upload_tab1(content, _title_name, file_name, current, _max_intervals, _uuid):
    log.debug('>>> parse_upload')
    if not current:
        log.warning("Ignore upload file when playing.")
        return _title_name, _max_intervals, _uuid
    try:
        log.debug('>>> Parsing upload files')
        if content is not None:
            if file_name.endswith('.p'):
                uuid, name, df, max_intervals, total_usage, scores = parse_content(content)
                log.debug(f'Load file: {uuid}')
                title_name = name if name is not None else file_name[:-2]
                cache.put(uuid, (title_name, df, max_intervals, total_usage, scores))
                return title_name, max_intervals, uuid
            raise Exception('File format is wrong.')
        else:
            log.warning("The uploaded file is None, ignore.")
            return _title_name, _max_intervals, _uuid
    except Exception as e:
        log.error(format_exc())
        return _title_name, _max_intervals, _uuid


@app.callback(
    [
        Output('tab2-card-title-data', 'data'),
        Output('tab2-max-intervals', 'data'),
        Output('tab2-data-frame', 'data'),
    ],
    [Input('tab2-upload-data', 'contents')],
    [
        State('tab2-card-title', 'children'),
        State('tab2-upload-data', 'filename'),
        State('tab2-interval-component', 'disabled'),
        State('tab2-max-intervals', 'data'),
        State('tab2-data-frame', 'data'),
    ]
)
def parse_upload_tab2(list_of_contents, _title_name, list_of_names, current, _max_intervals, _uuids):
    log.debug('>>> parse_upload')
    if not current:
        log.warning("Ignore upload file when playing.")
        return _title_name, _max_intervals, _uuids
    try:
        if list_of_contents is not None and len(list_of_contents) == 2:
            uuids = []
            title_names = []
            result_max_intervals = []
            if list_of_names[0] > list_of_names[1]:
                list_of_names = reversed(list_of_names)
                list_of_contents = reversed(list_of_contents)
            log.debug(f'>>> Parsing upload files: {list_of_names}')
            for content, file_name in zip(list_of_contents, list_of_names):
                if file_name.endswith('.p'):
                    uuid, name, df, max_intervals, total_usage, scores = parse_content(content)
                    cache.put(uuid, (name, df, max_intervals, total_usage, scores))
                    log.debug(f'Load file: {uuid}')
                    uuids.append(uuid)
                    title_name = name if name is not None else file_name[:-2]
                    cache.put(uuid, (title_name, df, max_intervals, total_usage, scores))
                    title_names.append(title_name)
                    result_max_intervals.append(max_intervals)
                else:
                    raise Exception('File format is wrong.')

            title_name = f'{title_names[0]} vs {title_names[1]}'
            return title_name, max(result_max_intervals), uuids
        else:
            log.warning("The uploaded file number is not 2, ignore.")
            return _title_name, _max_intervals, _uuids
    except Exception as e:
        log.error(format_exc())
        return _title_name, _max_intervals, _uuids


@app.callback(
    Output('app-content', 'children'),
    [Input('app-tabs', 'active_tab')],
    [
        State('tab1-data-frame', 'data'),
        State('tab2-data-frame', 'data'),
        State('tab1-card-title-data', 'data'),
        State('tab2-card-title-data', 'data'),
    ]
)
def render_tab_content(tab_switch,
                       tab1_content_uuid, tab2_content_uuid, card1_title, card2_title):
    log.debug(f'>>> render_tab_content')
    if tab_switch == 'tab1':
        log.debug(tab1_content_uuid)
        name, df, _, total_usage, scores = cache.get(tab1_content_uuid)
        return build_tab1(0, df, total_usage, scores, card1_title)
    if tab_switch == 'tab2':
        log.debug(tab2_content_uuid[0])
        log.debug(tab2_content_uuid[1])
        name0, df0, len0, total_usage0, scores0 = cache.get(tab2_content_uuid[0])
        name1, df1, len1, total_usage1, scores1 = cache.get(tab2_content_uuid[1])
        # render_tab_content is always the first frame
        # we may not expand the frames
        #
        # df0 = expand(df0, max(len0, len1))
        # df1 = expand(df1, max(len0, len1))
        # total_usage0 = expand(total_usage0, max(len0, len1))
        # total_usage1 = expand(total_usage1, max(len0, len1))
        # scores0 = expand(scores0, max(len0, len1))
        # scores1 = expand(scores1, max(len0, len1))
        return build_tab2(0, df0, df1, total_usage0, total_usage1, scores0, scores1, card2_title)
    raise NotImplementedError()


@app.callback(
    [
        Output('tab1-led', 'value'),
        Output('tab1-gauge-cpu', 'value'),
        Output('tab1-gauge-mem', 'value'),
        Output('tab1-gauge-cpu', 'color'),
        Output('tab1-gauge-mem', 'color'),
        Output('tab1-graph0', 'figure'),
        Output('tab1-graph1', 'figure'),
        Output('tab1-graph2', 'figure')
    ],
    [
        Input('tab1-slider', 'value'),
        Input('tab1-interval-done', 'data')
    ],
    [
        State('app-tabs', 'active_tab'),
        State('tab1-data-frame', 'data'),
    ]
)
def update_graph_tab1(n_intervals, show_delta,
                      tab_switch, content_uuid):
    log.debug('>>> update_graph_tab1')
    if tab_switch != 'tab1':
        return (None,) * 8

    from .figures.tab1 import get_figure0, get_figure1, get_figure2, get_gauge_color, get_gauge_value

    name, df, max_interval, total_usage, scores = cache.get(content_uuid)
    cpu, mem = get_gauge_value(n_intervals, df)
    cpu_color, mem_color = get_gauge_color(cpu), get_gauge_color(mem)
    return scores['score'][n_intervals], \
        cpu, mem, cpu_color, mem_color, \
        get_figure0(n_intervals, df, show_delta), \
        get_figure1(n_intervals, df), \
        get_figure2(n_intervals, total_usage)


@app.callback(
    [
        Output('tab2-graph0', 'figure'),
        Output('tab2-graph1', 'figure'),
        Output('tab2-graph2', 'figure'),
        Output('tab2-graph3', 'figure')
    ],
    [
        Input('tab2-slider', 'value'),
        Input('tab2-interval-done', 'data')
    ],
    [
        State('app-tabs', 'active_tab'),
        State('tab2-data-frame', 'data'),
    ]
)
def update_graph_tab2(n_intervals, show_delta,
                      tab_switch, content_uuid):
    log.debug('>>> update_graph_tab2')
    if tab_switch != 'tab2':
        return (None,) * 8

    from .figures.tab2 import get_figure0, get_figure1, get_figure2, get_figure3, get_gauge_color, get_gauge_value

    name0, df0, len0, total_usage0, scores0 = cache.get(content_uuid[0])
    name1, df1, len1, total_usage1, scores1 = cache.get(content_uuid[1])

    # expand the frame when the n_intervals is out of bound
    if n_intervals >= len0:
        df0 = expand(df0, len1)
        total_usage0 = expand(total_usage0, len1)
        scores0 = expand(scores0, len1)
    if n_intervals >= len1:
        df1 = expand(df1, len0)
        total_usage1 = expand(total_usage1, len0)
        scores1 = expand(scores1, len0)

    return get_figure0(n_intervals, df0, df1, name0, name1, show_delta), \
        get_figure1(n_intervals, [df0, df1][len0 < len1]), \
        get_figure2(n_intervals, total_usage0, total_usage1, name0, name1), \
        get_figure3(n_intervals, df0, df1, scores0, scores1, name0, name1)


def update_card_title(title):
    return title


def update_slider_max(max_intervals):
    log.debug('update_slider_max')
    return max_intervals - 1


def update_slider(slider_value):
    log.debug(f'>>> update_slider: {slider_value}')
    return f'{slider_value}'


# Callbacks for stopping interval update
def set_interval(stop_n_clicks, reset_n_clicks, n_intervals, content,
                 current, slider_value, interval_done, max_intervals):
    log.debug('>>> set_interval')
    ctx = callback_context

    if not ctx.triggered:
        input_id = 'reset'
    else:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0].split('-')[1]

    log.debug(f'??? ctx: {input_id}')

    def pause_callback():
        if stop_n_clicks == 0:
            return True, 'pause', slider_value, interval_done
        return not current, 'pause' if current else 'start', slider_value, interval_done

    def interval_callback(_slider_value):
        if _slider_value + 1 == max_intervals:
            return True, 'start', _slider_value, False
        if not current and not interval_done:
            _slider_value += 1
        return current, 'start' if current else 'pause', _slider_value, not interval_done

    def reset_callback():
        return True, 'start', 0, True

    callbacks = dict(
        pause=pause_callback,
        interval=interval_callback
    )

    callbacks_kwargs = dict(
        interval=dict(_slider_value=slider_value)
    )

    callback_func = callbacks.get(input_id, reset_callback)
    kwargs = callbacks_kwargs.get(input_id, dict())
    return callback_func(**kwargs)


for tab in ('tab1-', 'tab2-'):
    app.callback(
        Output(tab + 'card-title', 'children'),
        [Input(tab + 'card-title-data', 'data')]
    )(update_card_title)

    app.callback(
        Output(tab + 'slider', 'max'),
        [Input(tab + 'max-intervals', 'data')]
    )(update_slider_max)

    app.callback(
        Output(tab + 'slider-output-container', 'children'),
        [Input(tab + 'slider', 'value')]
    )(update_slider)

    app.callback(
        [
            Output(tab + 'interval-component', 'disabled'),
            Output(tab + 'pause-button', 'children'),
            Output(tab + 'slider', 'value'),
            Output(tab + 'interval-done', 'data')
        ],
        [
            Input(tab + 'pause-button', 'n_clicks'),
            Input(tab + 'reset-button', 'n_clicks'),
            Input(tab + 'interval-component', 'n_intervals'),
            Input(tab + 'data-frame', 'data'),
        ],
        [
            State(tab + 'interval-component', 'disabled'),
            State(tab + 'slider', 'value'),
            State(tab + 'interval-done', 'data'),
            State(tab + 'max-intervals', 'data')
        ],
    )(set_interval)
