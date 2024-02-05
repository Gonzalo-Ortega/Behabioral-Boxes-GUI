
from nicegui import app, native, ui


from controllers import main_controller


def display_gui():
    ui.label('Behavioral Boxes GUI')
    mice_number = ui.number(label='Mice Number', value=0, min=0, format='%.0f')
    main_controller.AnimalNumber = int(mice_number.value)

    ui.button('Save', on_click=lambda: ui.notify('Mice number: ' + str(main_controller.AnimalNumber)))


def window_config():
    app.native.window_args['resizable'] = True
    app.native.start_args['debug'] = False
    ui.run(native=True, window_size=(500, 500), fullscreen=False, reload=False, port=native.find_open_port())


if __name__ in {"__main__", "__mp_main__"}:
    display_gui()
    window_config()
