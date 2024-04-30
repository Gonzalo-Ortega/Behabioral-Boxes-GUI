from nicegui import app, native, ui
from controllers.main_controller import MainController
from controllers import sound_controller
from controllers import valves_controller
from controllers import video_controller

import mat73
import pickle
import os
import shutil

PORTS_MATRIX = []
PORTS_MATRIX_DIR = 'OutputMatrixPorts12Batch.mat'
# C:/Users/DALMAU-PC2/AppData/Local/Programs/Python/Python312


def display_calibration_tab(data):
    ui.number(label='Reward time', value=0.5, min=0, format='%.3f', step=0.001).bind_value(data, 'reward_time')
    ui.markdown('Compensation factor: ')
    f = valves_controller.compensationFactor
    with ui.row():
        for i in range(0, 7):
            ui.number(label=str(i + 1), value=f[i], min=0, format='%.2f', step=0.01)
    ui.markdown('Active ports: ')
    with ui.row():
        for i in range(0, 7):
            ui.switch(str(i + 1), value=True)


def display_experiment_tab(data):
    with ui.row():
        ui.number(label='Experiment day', value=1, min=1, format='%.0f').bind_value(data, 'exp_day')
        ui.number(label='Animal Number', value=1, min=1, format='%.0f').bind_value(data, 'animal_number')
        ui.select({1: 'Stage 1', 1.2: 'Stage 1.2', 2: 'Stage 2', 2.2: 'Stage 2.2', 3: 'Stage 3', 4: 'Stage 4'},
                  label='Stage', value=3).bind_value(data, 'stage')
        ui.number(label='Drug type', value=0, min=0, format='%.0f').bind_value(data, 'drug_type')
        ui.number(label='Dose', value=0.000, min=0, format='%.3f', step=0.001).bind_value(data, 'dose')
    with ui.row():
        with ui.grid(columns=2):
            ui.markdown('Correct port:')
            with ui.card():
                ui.label('').bind_text(data, 'correct_port')
            ui.markdown('Yesterday\'s port:')
            with ui.card():
                ui.label('').bind_text(data, 'yesterdays_port')
        ui.upload(label='Upload a custom port matrix:', max_files=1, on_upload=lambda file: update_port_matrix(file)).tooltip(
            'Port matrix is used to compute today and yesterday\'s correct ports')


def display_gui():
    global PORTS_MATRIX
    PORTS_MATRIX = mat73.loadmat(PORTS_MATRIX_DIR)
    data = load_data()

    ui.markdown('### Behavioral Boxes GUI')
    ui.markdown('Enter the desired configuration and select the running mode:')
    with ui.row():
        with ui.link(target='https://github.com/Gonzalo-Ortega/Behavioral-Boxes-GUI'):
            ui.image('mice.png').classes('w-20').tooltip('GUI source code!')
        with ui.card():
            with ui.row():
                ui.select({1: 'Box 1', 2: 'Box 2', 3: 'Box 3'}, label='Box', value=1).bind_value(data, 'box')
                ui.select({0: 'Calibrate', 1: 'Train', 2: 'Recall'}, label='Mode', value=1).bind_value(data, 'mode')
        with ui.column():
            ui.button('Save', icon='save', on_click=lambda: save_data(data)).tooltip(
                'Save changed data before running')
            ui.button('Run', icon='play_circle_outline', on_click=lambda: run_experiment(data)).tooltip(
                'Run selected mode')
    ui.separator()
    with ui.tabs() as tabs:
        calibrate = ui.tab('Calibration').tooltip(
            'Change parameters for vale\'s calibration')
        experiment = ui.tab('Train or recall').tooltip(
            'Change parameters for train or recall modes')
    with ui.tab_panels(tabs, value=experiment):
        with ui.tab_panel(calibrate):
            with ui.card():
                display_calibration_tab(data)
        with ui.tab_panel(experiment):
            with ui.card():
                display_experiment_tab(data)


def update_port_matrix(file):
    global PORTS_MATRIX
    PORTS_MATRIX = mat73.loadmat(file.content)
    ui.notify(f'Ports matrix changed to: {file.name}')


def update_ports(data):
    day = int(data['exp_day'])
    animal = int(data['animal_number'])
    if day > len(PORTS_MATRIX['A']):
        ui.notify('Experiment day out of range.')
        data['correct_port'] = '-'
        data['yesterdays_port'] = '-'
    elif animal > len(PORTS_MATRIX['A'][day - 1]):
        ui.notify('Animal number out of range.')
        data['correct_port'] = '-'
        data['yesterdays_port'] = '-'
    else:
        data['correct_port'] = PORTS_MATRIX['A'][day - 1][animal - 1]
        if data['exp_day'] > 1:
            data['yesterdays_port'] = PORTS_MATRIX['A'][day - 2][animal - 1]
        else:
            data['yesterdays_port'] = 'First day'


def save_data(data):
    if data['mode'] == 0:
        valves_controller.timeLengthWReward = data['reward_time']
        valves_controller.compensationFactor = data['compensation_factor']
        ui.notify(data['compensation_factor'])

    elif data['mode'] == 1 or data['mode'] == 2:
        update_ports(data)
        ui.notify('Saved')
    else:
        ui.notify('ERROR')

    with open('data.pkl', 'wb') as file:
        pickle.dump(data, file)


def load_data() -> dict:
    if os.path.exists('data.pkl'):
        with open('data.pkl', 'rb') as file:
            data = pickle.load(file)
    else:
        data = {'box': 1, 'mode': 1, 'correct_port': '-', 'yesterdays_port': '-',
                'exp_day': 1, 'animal_number': 1, 'stage': 3, 'drug_type': 0, 'dose': 0.00,
                'reward_time': 0.5, 'active_ports': [False] * 8,
                'compensation_factor': [1.25, 2., .83, 2.9, 2.03, .83, 2.63, .8]}
    return data


def run_experiment(data):
    if data['mode'] == 0:
        valves_controller.configure(data['box'])
        ui.notify('Ready to calibrate!')
    else:
        main_controller = MainController(data)
        sound_controller.configure(data['box'])
        video_controller.configure(data['box'], data['mode'])
        if data['mode'] == 1:
            main_controller.run_experiment()
            ui.notify('Train mode!')
        elif data['mode'] == 2:
            main_controller.run_experiment()
            ui.notify('Recall mode!')
        else:
            ui.notify('ERROR')


def window_config():
    ui.page_title('Behavioral Boxes')

    app.native.window_args['resizable'] = True
    app.native.start_args['debug'] = False
    # ui.run(native=True, window_size=(500, 500), fullsound_controllerreen=False, reload=False, port=native.find_open_port())
    ui.run()


if __name__ in {'__main__', '__mp_main__'}:
    display_gui()
    window_config()
