import pyfirmata
import time
from time import sleep

# Configuration variables
compensationFactor = (1.25, 2., .83, 2.9, 2.03, .83, 2.63, .8)
port = 'COM18'

# Length of the open valve during licking
timeLengthWReward = .5


def configure(box):
    global compensationFactor, port
    if box == 1:
        compensationFactor = [1.25, 2., .83, 2.9, 2.03, .83, 2.63, .8]
        port = 'COM18'
    elif box == 2:
        compensationFactor = [2.5, 1.43, 1.25, 1.43, 4., 1.43, 2.5, 1.11]
        port = 'COM15'


def calibrate():
    print("[INFO] Loading Arduino")
    # Associate port and board with pyFirmata
    board = pyfirmata.ArduinoMega(port)
    # Use iterator thread to avoid buf fer overflow
    it = pyfirmata.util.Iterator(board)
    it.start()
    # Define i/o pins (giving roles to pins: i.e. d=digital,7=pin number,i=input)
    port_list = [board.get_pin('d:22:i'), board.get_pin('d:24:i'), board.get_pin('d:26:i'), board.get_pin('d:28:i'),
                 board.get_pin('d:30:i'), board.get_pin('d:32:i'), board.get_pin('d:34:i'), board.get_pin('d:36:i')]

    valve_list = [39, 41, 43, 45, 47, 49, 51, 53]
    led_state_list = [23, 25, 27, 29, 31, 33, 35, 37]

    for i in range(0, (len(led_state_list))):
        board.digital[led_state_list[i]].write(1)

    for j in range(0, 3):
        for i in range(0, (len(port_list))):
            time.sleep(0.5)
            pin = valve_list[i]
            time_length = timeLengthWReward * compensationFactor[i]
            board.digital[pin].write(1)
            sleep(time_length)
            board.digital[pin].write(0)
            print(i + 1)

    board.exit()
