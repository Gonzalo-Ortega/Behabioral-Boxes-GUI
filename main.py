from nicegui import app, native, ui
from controllers import main_controller
from controllers import sound_controller
from controllers import valves_controller
from controllers import video_controller

import mat73

PORTS_MATRIX_DIR = 'OutputMatrixPorts12Batch.mat'


def display_gui():
    ui.markdown("## Behavioral Boxes GUI")
    ui.markdown("### Enter the desired configuration and select the running mode:")

    experiment_vars = {}
    calibration_vars = {}

    with ui.row():
        mode_vars = {"box": ui.select({1: 'Box 1', 2: 'Box 2', 3: 'Box 3'}, label='Box', value=1),
                     "mode": ui.select({0: 'Calibrate', 1: 'Train mode', 2: 'Recall mode'}, label='Mode', value=1,
                                       on_change=lambda: check_visibilities(experiment_vars, mode_vars,
                                                                            calibration_vars))}
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
        ui.button('Save', on_click=lambda: save_values(mode_vars, experiment_vars, calibration_vars))
        ui.button('Run', on_click=lambda: run_experiment(mode_vars, experiment_vars, calibration_vars))

    check_visibilities(experiment_vars, mode_vars, calibration_vars)


def check_visibilities(experiment_vars, mode_vars, calibration_vars):
    if mode_vars["mode"].value == 0:
        for element in experiment_vars:
            experiment_vars[element].visible = False
        for element in calibration_vars:
            calibration_vars[element].visible = True
    else:
        for element in experiment_vars:
            experiment_vars[element].visible = True
        for element in calibration_vars:
            calibration_vars[element].visible = False


def save_values(mode_vars, experiment_vars, calibration_vars):
    mode = mode_vars["mode"].value
    if mode == 0:
        valves_controller.timeLengthWReward = int(calibration_vars["reward_time"].value)
    else:
        main_controller.ExpDay = int(experiment_vars["exp_day"].value)
        main_controller.AnimalNumber = int(experiment_vars["animal_number"].value)
        main_controller.StageLevel = int(experiment_vars["stage"].value)
        main_controller.DrugType = int(experiment_vars["drug_type"].value)
        main_controller.Dose = int(experiment_vars["dose"].value)
        update_ports(experiment_vars)
    ui.notify("Done!")


def update_ports(option_vars):
    port_matrix = mat73.loadmat(PORTS_MATRIX_DIR)
    correct_port = port_matrix["A"][main_controller.ExpDay - 1][main_controller.AnimalNumber - 1]
    option_vars["correct_port"].text = "Correct port: " + str(correct_port)
    if main_controller.ExpDay > 1:
        yesterday_port = port_matrix["A"][main_controller.ExpDay - 2][main_controller.AnimalNumber - 1]
    else:
        yesterday_port = "First day"
    option_vars["yesterday_port"].text = "Yesterday port: " + str(yesterday_port)


def run_experiment(mode_vars, experiment_vars, calibration_vars):
    mode = mode_vars["mode"].value
    box = mode_vars["box"].value
    if mode == 0:  # Calibrate
        valves_controller.configure(box)
        ui.notify("Ready to calibrate!")
    else:
        sound_controller.configure(box)
        main_controller.configure(box, mode)
        video_controller.configure(box, mode)
        if mode == 1:  # Train
            ui.notify("Train mode!")
        elif mode == 2:  # Recall
            ui.notify("Recall mode!")
        else:
            ui.notify("ERROR")


def window_config():
    app.native.window_args['resizable'] = True
    app.native.start_args['debug'] = False
    # ui.run(native=True, window_size=(500, 500), fullsound_controllerreen=False, reload=False, port=native.find_open_port())
    ui.run()

if __name__ in {"__main__", "__mp_main__"}:
    display_gui()
    window_config()
