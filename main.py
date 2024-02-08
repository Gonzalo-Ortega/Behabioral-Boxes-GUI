from nicegui import app, native, ui
from controllers import main_controller
from controllers import sound_controller
from controllers import valves_controller
from controllers import video_controller

import mat73

PORTS_MATRIX_DIR = 'OutputMatrixPorts12Batch.mat'


def display_calibration_tab(calibration_vars):
    calibration_vars.update(
        {'reward_time': ui.number(label='Reward time', value=0.5, min=0, format='%.3f', precision=3)})

    ui.markdown('Compensation factor: ')
    calibration_config = ui.row()
    f = valves_controller.compensationFactor
    with calibration_config:
        calibration_vars.update({'compensation_factor': (
            ui.number(label='1', value=f[0], min=0, format='%.2f', precision=2),
            ui.number(label='2', value=f[1], min=0, format='%.2f', precision=2),
            ui.number(label='3', value=f[2], min=0, format='%.2f', precision=2),
            ui.number(label='4', value=f[3], min=0, format='%.2f', precision=2),
            ui.number(label='5', value=f[4], min=0, format='%.2f', precision=2),
            ui.number(label='6', value=f[5], min=0, format='%.2f', precision=2),
            ui.number(label='7', value=f[6], min=0, format='%.2f', precision=2),
            ui.number(label='8', value=f[7], min=0, format='%.2f', precision=2))})

    ui.markdown('Active ports: ')
    calibration_vars.update({'active_ports': ui.row()})
    with calibration_vars['active_ports']:
        ui.switch('1', value=True)
        ui.switch('2', value=True)
        ui.switch('3', value=True)
        ui.switch('4', value=True)
        ui.switch('5', value=True)
        ui.switch('6', value=True)
        ui.switch('7', value=True)
        ui.switch('8', value=True)


def display_experiment_tab(mode_vars, experiment_vars):
    mode_vars.update({'config': ui.toggle({1: 'Train', 2: 'Recall'}, value=1)})
    with ui.row():
        experiment_vars.update({'exp_day': ui.number(label='Experiment day', value=1, min=1, format='%.0f'),
                                'animal_number': ui.number(label='Animal Number', value=1, min=1,
                                                           format='%.0f'),
                                'stage': ui.select({1: 'Stage 1', 2: 'Stage 2', 3: 'Stage 3'}, label='Stage',
                                                   value=3),
                                'drug_type': ui.number(label='Drug type', value=0, min=0, format='%.0f'),
                                'dose': ui.number(label='Dose', value=0.000, min=0, format='%.3f',
                                                  precision=3)})
    experiment_vars.update({'correct_port': ui.label('Correct port: '),
                            'yesterday_port': ui.label('Yesterday port: ')})


def display_gui():
    ui.markdown('## Behavioral Boxes GUI')
    ui.markdown('### Enter the desired configuration and select the running mode:')

    experiment_vars = {}
    calibration_vars = {}

    with ui.row():
        mode_vars = {'box': ui.select({1: 'Box 1', 2: 'Box 2', 3: 'Box 3'}, label='Box', value=1)}
        ui.button('Save', on_click=lambda: save_values(mode_vars, experiment_vars, calibration_vars))
        ui.button('Run', on_click=lambda: run_experiment(mode_vars, experiment_vars, calibration_vars))

    mode_vars.update({'mode': ui.tabs()})
    with mode_vars['mode'] as tabs:
        calibrate = ui.tab('Calibration')
        experiment = ui.tab('Train and recall')
        mode_vars['mode'].value = 'Train and recall'
    with ui.tab_panels(tabs, value=experiment):
        with ui.tab_panel(calibrate):
            display_calibration_tab(calibration_vars)
        with ui.tab_panel(experiment):
            display_experiment_tab(mode_vars, experiment_vars)


def update_ports(option_vars):
    port_matrix = mat73.loadmat(PORTS_MATRIX_DIR)
    correct_port = port_matrix['A'][main_controller.ExpDay - 1][main_controller.AnimalNumber - 1]
    option_vars['correct_port'].text = 'Correct port: ' + str(correct_port)
    if main_controller.ExpDay > 1:
        yesterday_port = port_matrix['A'][main_controller.ExpDay - 2][main_controller.AnimalNumber - 1]
    else:
        yesterday_port = 'First day'
    option_vars['yesterday_port'].text = 'Yesterday port: ' + str(yesterday_port)


def save_values(mode_vars, experiment_vars, calibration_vars):
    mode = mode_vars['mode'].value
    if mode == 'Calibration':
        valves_controller.timeLengthWReward = calibration_vars['reward_time'].value

        compensation_factor = [0] * 8
        for i in range(0, len(compensation_factor)):
            compensation_factor[i] = calibration_vars['compensation_factor'][i].value
        valves_controller.compensationFactor = compensation_factor

        ui.notify(compensation_factor)

    elif mode == 'Train and recall':
        main_controller.ExpDay = int(experiment_vars['exp_day'].value)
        main_controller.AnimalNumber = int(experiment_vars['animal_number'].value)
        main_controller.StageLevel = experiment_vars['stage'].value
        main_controller.DrugType = experiment_vars['drug_type'].value
        main_controller.Dose = experiment_vars['dose'].value
        update_ports(experiment_vars)
        ui.notify('Saved')
    else:
        ui.notify('ERROR')


def run_experiment(mode_vars, experiment_vars, calibration_vars):
    mode = mode_vars['mode'].value
    box = mode_vars['box'].value
    config = mode_vars['config'].value
    if mode == 'Calibration':
        valves_controller.configure(box)
        ui.notify('Ready to calibrate!')
    else:
        sound_controller.configure(box)
        main_controller.configure(box, mode)
        video_controller.configure(box, mode)
        if config == 1:
            ui.notify('Train mode!')
        elif config == 2:
            ui.notify('Recall mode!')
        else:
            ui.notify('ERROR')


def window_config():
    app.native.window_args['resizable'] = True
    app.native.start_args['debug'] = False
    # ui.run(native=True, window_size=(500, 500), fullsound_controllerreen=False, reload=False, port=native.find_open_port())
    ui.run()


if __name__ in {'__main__', '__mp_main__'}:
    display_gui()
    window_config()
