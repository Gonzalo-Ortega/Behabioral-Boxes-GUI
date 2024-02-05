import os
import subprocess
from pathlib import Path
import nicegui

# Step 1: Install PyInstaller
install_cmd = ['python', '-m', 'pip', 'install', 'pyinstaller']
subprocess.call(install_cmd)

cmd = [
    'python',
    '-m', 'PyInstaller',
    'gui_tests.py', # your main file with ui.run()
    '--name', 'testBuild', # name of your app
    '--onefile',
    #'--windowed', # prevent console appearing, only use with ui.run(native=True, ...)
    '--add-data', f'{Path(nicegui.__file__).parent}{os.pathsep}nicegui'
]
subprocess.call(cmd)
