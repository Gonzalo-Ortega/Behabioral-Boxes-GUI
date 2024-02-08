import pyfirmata
import time
from time import sleep

# Configuration variables
compensationFactor = (1.25, 2., .83, 2.9, 2.03, .83, 2.63, .8)
port = 'COM18'

# Global variables
timeLengthWReward = .5  # 1.3 #0.035 #0.025 #0.02 #Length of the open valve during licking

# Rewarded port must be changed every day for environment A and keep constant for environment B
PhysRewardPort = 1  # from 1 to 8 correspond to rewarded port with respect to the arena
GeographRewardPort = 1  # Each of the numbners between [1,2,3,4,5,6,7,8] correspond o to the 8 geographycal orientations['NE','EE','SE','SS','SO','OO','NO','NN']
GeographOrientations = ['NE', 'EE', 'SE', 'SS', 'SO', 'OO', 'NO', 'NN']  # One of the following geafrical orientations respect to the box ['','']
TimeToReachReward = 6  # 10 #Time that the animal has to get the reward


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
    listofPorts = [board.get_pin('d:22:i'), board.get_pin('d:24:i'), board.get_pin('d:26:i'), board.get_pin('d:28:i'),
                   board.get_pin('d:30:i'), board.get_pin('d:32:i'), board.get_pin('d:34:i'), board.get_pin('d:36:i')]

    cameraTriggerPin = board.get_pin('d:12:i')

    listofValves = [39, 41, 43, 45, 47, 49, 51, 53]
    listofPortsStates = [False, False, False, False, False, False, False, False]
    listofLedsStates = [23, 25, 27, 29, 31, 33, 35, 37]

    for i in range(0, (len(listofLedsStates))):
        board.digital[listofLedsStates[i]].write(1)

    for j in range(0, 3):  # If you want to loop 100 times write 100 here instead of 1
        for i in range(0, (len(listofPorts))):  # [(4-1)]:# ## # # ## #in [(3-1)]:
            time.sleep(0.5)

            pin = listofValves[i]
            timeLength = timeLengthWReward * compensationFactor[i]
            board.digital[pin].write(1)
            sleep(timeLength)
            board.digital[pin].write(0)
            print(i + 1)

    board.exit()
