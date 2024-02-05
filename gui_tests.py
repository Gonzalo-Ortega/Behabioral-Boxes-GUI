from tkinter import *
from tkinter import ttk

from kivy.app import App
from kivy.uix.label import Label

from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt6.QtCore import QSize, Qt

from nicegui import app, native, ui
from nicegui.events import ValueChangeEventArguments

import sys

def nice_gui():
    ui.label('Hello NiceGUI!')

    ui.icon('thumb_up')
    ui.markdown('This is **Markdown**.')
    ui.html('This is <strong>HTML</strong>.')
    with ui.row():
        ui.label('CSS').style('color: #888; font-weight: bold')
        ui.label('Tailwind').classes('font-serif')
        ui.label('Quasar').classes('q-ml-xl')
    ui.link('NiceGUI on GitHub', 'https://github.com/zauberzeug/nicegui')

    def show(event: ValueChangeEventArguments):
        name = type(event.sender).__name__
        ui.notify(f'{name}: {event.value}')

    ui.button('Button', on_click=lambda: ui.notify('Click'))
    with ui.row():
        ui.checkbox('Checkbox', on_change=show)
        ui.switch('Switch', on_change=show)
    ui.radio(['A', 'B', 'C'], value='A', on_change=show).props('inline')
    with ui.row():
        ui.input('Text input', on_change=show)
        ui.select(['One', 'Two'], value='One', on_change=show)
    ui.link('And many more...', '/documentation').classes('mt-8')

    app.native.window_args['resizable'] = True
    app.native.start_args['debug'] = False

    ui.run(native=True, window_size=(400, 300), fullscreen=False, reload=False, port=native.find_open_port())


def tk_gui():
    root = Tk()
    frm = ttk.Frame(root, padding=10)
    frm.grid()
    ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
    ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
    root.mainloop()


def kivy_gui():
    class kivy_gui(App):
        def build(self):
            return Label(text='Hello world')


def py_qt_6():
    # Subclass QMainWindow to customize your application's main window
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            self.setWindowTitle("My App")
            button = QPushButton("Press Me!")

            # Set the central widget of the Window.
            self.setCentralWidget(button)

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()


if __name__ in {"__main__", "__mp_main__"}:
    nice_gui()
    #tk_gui()
    #kivy_gui().run()
    #py_qt_6()

