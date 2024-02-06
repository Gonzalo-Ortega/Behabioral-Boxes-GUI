from nicegui import app, native, ui
from controllers import main_controller as mc
from controllers import valves_controller as vc

import mat73

PORTS_MATRIX_DIR = 'OutputMatrixPorts12Batch.mat'


def display_gui():
    ui.markdown("## Behavioral Boxes GUI")
    ui.markdown("### Enter the desired configuration and select the running mode:")

    experiment_vars = {}
    calibration_vars = {}

    with ui.row():
        mode_vars = {"box": ui.select({1: 'Box 1', 2: 'Box 2', 3: 'Box 3'}, label='Box', value=1),
                     "mode": ui.select({1: 'Calibrate', 2: 'Test', 3: 'Recall'}, label='Mode', value=2,
                                       on_change=lambda: check_visibilities(experiment_vars, mode_vars, calibration_vars))}
    with ui.row():
        experiment_vars.update({"exp_day": ui.number(label='Experiment day', value=1, min=1, format='%.0f'),
                                "animal_number": ui.number(label='Animal Number', value=1, min=1, format='%.0f'),
                                "stage": ui.select({1: 'Stage 1', 2: 'Stage 2', 3: 'Stage 3'}, label='Stage', value=3),
                                "drug_type": ui.number(label='Drug type', value=0, min=0, format='%.0f'),
                                "dose": ui.number(label='Dose', value=0.000, min=0, format='%.3f', precision=3)})
        calibration_vars = {"reward_time": ui.number(label='Reward time', value=0.5, min=0, format='%.3f', precision=3)}

    experiment_vars.update({"correct_port": ui.label('Correct port: '),
                            "yesterday_port": ui.label('Yesterday port: ')})

    with ui.row():
        ui.button('Save', on_click=lambda: save_values(experiment_vars))
        ui.button('Run', on_click=lambda: run_experiment())


def check_visibilities(experiment_vars, mode_vars, calibration_vars):
    if mode_vars["mode"].value == 1:
        for element in experiment_vars:
            experiment_vars[element].visible = False
        for element in calibration_vars:
            calibration_vars[element].visible = True
    else:
        for element in experiment_vars:
            experiment_vars[element].visible = True
        for element in calibration_vars:
            calibration_vars[element].visible = False


def save_values(option_vars):
    mc.ExpDay = int(option_vars["exp_day"].value)
    mc.AnimalNumber = int(option_vars["animal_number"].value)
    mc.StageLevel = int(option_vars["stage"].value)
    mc.DrugType = int(option_vars["drug_type"].value)
    mc.Dose = int(option_vars["dose"].value)
    update_ports(option_vars)
    ui.notify("Done!")


def update_ports(option_vars):
    port_matrix = mat73.loadmat(PORTS_MATRIX_DIR)
    correct_port = port_matrix["A"][mc.ExpDay - 1][mc.AnimalNumber - 1]
    option_vars["correct_port"].text = "Correct port: " + str(correct_port)
    if mc.ExpDay > 1:
        yesterday_port = port_matrix["A"][mc.ExpDay - 2][mc.AnimalNumber - 1]
    else:
        yesterday_port = "First day"
    option_vars["yesterday_port"].text = "Yesterday port: " + str(yesterday_port)


def run_experiment():
    notification = 'Experiment day: ' + str(mc.ExpDay) + \
                   ' - Animal number: ' + str(mc.AnimalNumber) + \
                   ' - Stage level: ' + str(mc.StageLevel) + \
                   ' - Drug type: ' + str(mc.DrugType) + \
                   ' - Dose: ' + str(mc.Dose)
    ui.notify(notification)


def window_config():
    app.native.window_args['resizable'] = True
    app.native.start_args['debug'] = False
    ui.run(native=True, window_size=(500, 500), fullscreen=False, reload=False, port=native.find_open_port())


if __name__ in {"__main__", "__mp_main__"}:
    display_gui()
    window_config()
