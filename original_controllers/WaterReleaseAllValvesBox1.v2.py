#!/usr/bin/python
#Import required libraries
import pyfirmata
import time
from time import sleep

###################################################################################################
print "[INFO] Loading Arduino"
#Global variables
timeLengthWReward =.5 #1.3 #0.035 #0.025#0.02#Length of the open valve during licking
compensationFactor = (1.25, 2., .83, 2.9, 2.03, .83, 2.63, .8)#(1, 1, 1, 1, 1, 1, 1, 1) # (0.74, 0.79, 0.68, 0.77, 0.88, 0.96, 0.70, 0.79) #( 1, 1, 1, 1, 1, 1, 1)#(1.13, 1.58, 1.33, 1.21, 1.31, 2.25, 1.14, 1.24)#(1, 1, 1, 1, 1, 1, 1, 1)#(1.3, 1.7, 1.3, 1.2, 1.4, 1.3, 1, 2.5) #(1.3, 1.7, 1.3, 1.2, 1.4, 1.3, 1, 2.5)  
## COMPENSATION FACTOR CHANGED THE 20/12/2021
#Rewarded port must be changed every day for environment A and keep constant for environmnet B
PhysRewardPort=1#from 1 to 8 correspond to rewarded port with respect to the arena
GeographRewardPort=1#Each of the numbners between [1,2,3,4,5,6,7,8] correspond o to the 8 geographycal orientations['NE','EE','SE','SS','SO','OO','NO','NN']
GeographOrientations=['NE','EE','SE','SS','SO','OO','NO','NN']#One of the following geafrical orientations respect to the box ['','']
TimeToReachReward=6 #10 #Time that the animal has to get the reward
#Define custom function in Python to perform Blink action
def openWaterPort(pin, timeLength, message): #any oin number from 22 to 53
    #print(message)
    board.digital[pin].write(1)
    sleep(timeLength)   
    board.digital[pin].write(0)

#Associate port and board with pyFirmata
port = 'COM18'#this must be changed for every board that is utilized
board = pyfirmata.ArduinoMega(port)
#Use iterator thread to avoid buf fer overflow
it = pyfirmata.util.Iterator(board)
it.start()
#Define i/o pins (giving roles to pins: i.e. d=digital,7=pin number,i=input)
lickPin1 = board.get_pin('d:22:i') 
lickPin2 = board.get_pin('d:24:i')
lickPin3 = board.get_pin('d:26:i')
lickPin4 = board.get_pin('d:28:i')
lickPin5 = board.get_pin('d:30:i')
lickPin6 = board.get_pin('d:32:i')
lickPin7 = board.get_pin('d:34:i')
lickPin8 = board.get_pin('d:36:i')
listofPorts=[lickPin1,lickPin2,lickPin3,lickPin4,lickPin5,lickPin6,lickPin7,lickPin8]
cameraTriggerPin = board.get_pin('d:12:i')
valvePin1 = 39
valvePin2 = 41
valvePin3 = 43
valvePin4 = 45
valvePin5 = 47
valvePin6 = 49
valvePin7 = 51
valvePin8 = 53
listofValves=[valvePin1, valvePin2, valvePin3, valvePin4, valvePin5, valvePin6, valvePin7, valvePin8]#, outPin2, outPin3, outPin4, outPin5, outPin6, outPin7, outPin8]
#Initializing valve state value
value1=False
value2=False
value3=False
value4=False
value5=False
value6=False
value7=False
value8=False
listofPortsStates=[value1,value2,value3,value4,value5,value6,value7,value8]
#Defining leds Ports
ledsPin1 = 23
ledsPin2 = 25
ledsPin3 = 27
ledsPin4 = 29
ledsPin5 = 31
ledsPin6 = 33
ledsPin7 = 35
ledsPin8 = 37
listofLedsStates=[ledsPin1, ledsPin2, ledsPin3, ledsPin4, ledsPin5, ledsPin6, ledsPin7, ledsPin8]#, ledsPin2, ledsPin3, ledsPin4, ledsPin5, ledsPin6, ledsPin7, ledsPin8]


for i in range(0,(len(listofLedsStates))):
    board.digital[listofLedsStates[i]].write(1)

for j  in range(0,3):#If you want to loop 100 times write 100 here instead of 1
    for i in range (0,(len(listofPorts))):# [(4-1)]:# ## # # ## #in [(3-1)]:
        time.sleep(0.5)      
        openWaterPort(listofValves[i],timeLengthWReward*compensationFactor[i], "Water drop port ")
        print i+1
        
board.exit()