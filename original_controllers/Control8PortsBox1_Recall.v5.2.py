#!/usr/bin/python
#Import required libraries
from termcolor import colored
import pyfirmata
import time
from time import sleep
import random
import pickle
import datetime
import matplotlib.pyplot as plt
import numpy as np, scipy.io
from multiprocessing import Process, Value
import cv2
from mouseTrackerBox1MOG2AdapRecall import runvideo
from PlaySoundEnvA import sine_tone_EnvA
import tables
inputmat=tables.openFile('C:/Users/Dalmau/Documents/Python Scripts/OutputMatrixPorts12Batch.mat')
PortsMatrix=inputmat.root.A
TimesRecallVector=inputmat.root.TimeDelayRecall
AnimalNames =["Mouse4151", "Mouse4152", "Mouse4153", "Mouse4154", "Mouse4155", "Mouse4156", "Mouse4157", "Mouse4158", "Mouse4159", "Mouse4160" ]

########################################################################################################
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
prtLoc = []
refPt = []
cropping = False
 
def set_Env_contour_and_Port_Locat(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that the cropping operation is finished
        refPt.append((x, y))
        cropping = False 
        # draw a circle around the region of interest
        center=np.mean(refPt,0)
        radius=np.sqrt(np.sum(np.square(np.diff(refPt,1))))/2
        cv2.circle(image, (int(round(center[0])),int(round(center[1]))), int(round(radius)), (0, 255, 0), 2)#cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)
    
    global prtLoc    
    # if right mouse button is clicked then port location is recorded    
    if event == cv2.EVENT_RBUTTONDBLCLK:#cv2.EVENT_LBUTTONDBLCLK:
        prtLoc = [(x, y)]
        cv2.circle(image, (prtLoc[0]), 2, (255, 0, 0), 2)#cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

################################################################################
image = []
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] taking one picture of the environment...")
cap = cv2.VideoCapture(0)
time.sleep(3.0)
ret, frame = cap.read()
#frame = cv2.flip(frame,0)
if ret == True:
    # converting fram in a gray scale picture        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (11, 11), 0)
cap.release()    
# Clone it and setup the mouse callback function
image = gray.copy()
cv2.namedWindow("image")
#Determining the countour of the spatial enclosure
cv2.setMouseCallback("image", set_Env_contour_and_Port_Locat) 
# keep looping until the 'q' key is pressed
while len(refPt) < 2:
    # display the image and wait for a keypress
    cv2.imshow("image", gray)
    key = cv2.waitKey(1) & 0xFF
cv2.waitKey(0)
#Save data for next step
center=np.mean(refPt,0)
radius=np.sqrt(np.sum(np.square(np.diff(refPt,1))))/2
print "[INFO] giving values to: Arenas center =", center, ", Arenas radius =", radius 
print "[INFO] port locations =", prtLoc[0]
# Release everything if job is finished
cv2.destroyAllWindows()
###################################################################################################
print "[INFO] Loading Arduino"
#Global variables
toneFreq =10560 #4400 #In herzs IMPORTANT: SPEAKER AMPLITUDE AT 10
toneLength = 1  #in seconds (it doesnt accept fractions of a second)
volume = 1
sample_rate = 22050
extraRewardWCue =0 # with 0 there is not water drop with the cue
extraRewardWStart =0
extraRewardEND = 1 # extra drop at the end of the experimen
NumTrialsWLightCue =0 #Number of trials with light on correct port 
lickTimeWindow= 840 #float('inf') #we might put a time window (5sec)  
timeLengthWTone = 0.040 #length of open valve during the tone
timeLengthWReward = 0.05 #0.025#0.02#Length of the open valve during licking
probReward = 1#In general this should be a small number (i.e. 0.25)
compensationFactor = (1.25, 2., .83, 2.9, 2.03, .83, 2.53, .8) #(0.73, 0.73, 0.67, 0.77, 0.68, 0.86, 0.71, 0.76) #(0.74, 0.79, 0.68, 0.77, 0.88, 0.96, 0.70, 0.79)#(0.7, 0.77, 0.7, 0.7, 0.75, 0.63, 0.64, 1)
timeSampling = 0.05#This could be dictated by the DAQ from minimicroscope
#Rewarded port must be changed every day for environment A and keep constant for environmnet B
AnimalNumber=10
ExpDay=50
PhysRewardPortYesterday = int(PortsMatrix[AnimalNumber - 1][ExpDay - 2]) 
PhysRewardPort = int(PortsMatrix[AnimalNumber -1][ExpDay - 1])#Port from the box that will be located in the physical position called GeographRewardPort 
SecondPhysRewardPort=[PhysRewardPort, PhysRewardPort, PhysRewardPort]#from 1 to 8 correspond to rewarded port with respect to the arena for each of the three rotations
GeographRewardPort=PhysRewardPort #Each of the numbners between [1,2,3,4,5,6,7,8] correspond o to the 8 geographycal orientations['NE','EE','SE','SS','SO','OO','NO','NN']
GeographOrientations=['NE','EE','SE','SS','SO','OO','NO','NN']#One of the following geafrical orientations respect to the box ['','']
TimeToReachReward=4 #10 #Time that the animal has to get the reward
RewNonRewTimeRatio=float(TimesRecallVector[ExpDay-1][0])/10.#10./10. #1./2. #Time Ratio of the experiment until reach rewarded trials
FlagForRecallTest=1#FLAG TO TEST RECALL
FixMinRewTime = 7.#1.5
StageLevel =3
DrugType=0 #0 is no drug, 1 is saline, 2 is muscimol, 3 is saline for CPP, 4 is CPP
Dose =0
#Define custom function in Python to perform Blink action
def openWaterPort(pin, timeLength, message): #any oin number from 22 to 53
    print(message)
    board.digital[pin].write(1)
    sleep(timeLength)   
    board.digital[pin].write(0)

#Associate port and board with pyFirmata
port = 'COM18'#this must be changed for every board that is utilized
board = pyfirmata.ArduinoMega(port)
#Use iterator thread to avoid buffer overflow
it = pyfirmata.util.Iterator(board)
it.start()
time.sleep(1)
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
outPin1 = 39
outPin2 = 41
outPin3 = 43
outPin4 = 45
outPin5 = 47
outPin6 = 49
outPin7 = 51
outPin8 = 53
listofValves=[outPin1, outPin2, outPin3, outPin4, outPin5, outPin6, outPin7, outPin8]#, outPin2, outPin3, outPin4, outPin5, outPin6, outPin7, outPin8]
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
listofLeds=[ledsPin1, ledsPin2, ledsPin3, ledsPin4, ledsPin5, ledsPin6, ledsPin7, ledsPin8]#, ledsPin2, ledsPin3, ledsPin4, ledsPin5, ledsPin6, ledsPin7, ledsPin8]
#Initializing leds state value
listofLedsStates=[value1,value2,value3,value4,value5,value6,value7,value8]
#Initializing the state of the animal
In=False
Water=False
#Defining libraries for data storage
datatimes = {}
datatimes['cueTime'] = []
datatimes['timeLickIncorrectPort'] = []
datatimes['timeLickCorrectPort'] = []
datatimes['IncorrectPortLicked'] = []
datatimes['timeErrorPort'] = []#Licking ports outside the tone
datatimes['ErrorPortLicked'] = []
datatimes['timeErrorCorrectPort']=[]#Licking correct ports outside the tone
datatimes['expStart'] = []
datatimes['waterStart'] = []
datatimes['RewNonRewTimeRatio'] = []
datatimes['FlagForRecallTest'] = []
datatimes['timeDelayCorrectLick']=[]
datatimes['correctPortTimeWindow'] = []
datatimes['timeLickFirstCorrect']= []
datatimes['timeDelayFirstCorrectLick']= []
dataPlot = {}
dataPlotCorrectPort=[]
dataPlotAllCorrectPort=[]
dataPlotIncorrectPort=[]
dataPlotErrorCorrectPort=[]
dataPlotIncorrectInCorrectTrials=[]
dataPlotTrials=[]
dataPlotErrorPort=[]
dataTriggerZone =[]
dataTrajectory =[]
mouseTrajectory = {}
mouseTrajectory['traj'] = []
mouseTrajectory['trigerZone'] = []
mouseTrajectory['trigerTime'] = []
mouseTrajectory['BoxDimentions'] = []
mouseTrajectory['CorrectPortLocation'] = []
mouseTrajectory['BoxDimentions'].append((center,radius,'EnvA'))
mouseTrajectory['CorrectPortLocation'].append((prtLoc))
mouseTrajectory['TriggerZoneRatio']=[]
mouseTrajectory['NumTrialsWLightCue']=[]
mouseTrajectory['Stage']=[]
mouseTrajectory['DrugType']=[]
mouseTrajectory['Dose']=[]
mouseTrajectory['TimeLength']=[]
mouseTrajectory['AcclimatTime']=[]
#Performance score plotting variables..
kk=0
OldLength=0
kkk=0
ww=0
rot=0
freqPlot=3
fig=plt.figure(1,figsize=(13,6))
m_1=[]
#while loop temporal variables
t_init = time.time() #we fix the time of the beggining of the experiment
expLength =10.0 #(in minutes)
aclimTimeLength = .7 #1.6#0.05#1.6 #(in minutes)
t_end = t_init  + 60 *(expLength + aclimTimeLength) # we fix the total length of the experiment
timeFirstEntry = t_init
timeFirstEntry2 = t_init
RewNonRewTimeRatio= random.random() #float(TimesRecallVector[ExpDay-1][0])/10.#10./10. #1./2. #Time Ratio of the experiment until reach rewarded trials
if FlagForRecallTest==0:#Otherwise maintins the value assigned on TimesRecallVector
    RewNonRewTimeRatio=0
FirstCorrect = False
cueTime = t_init
timeLastEntry= t_init
#Size of triggering zone
PercenRadiusDisc= 0.26#0.26
mouseTrajectory['TriggerZoneRatio'].append((PercenRadiusDisc))
mouseTrajectory['Stage'].append((StageLevel))
mouseTrajectory['DrugType'].append((DrugType))
mouseTrajectory['Dose'].append((Dose))
mouseTrajectory['TimeLength'].append((expLength))
mouseTrajectory['AcclimatTime'].append((aclimTimeLength))
TriggeringZoneRadius=radius*PercenRadiusDisc#radius*4/5

#transforming water port location from cartesian to polar coordinates
PrtLocAngle=np.arctan2((1.*prtLoc[0][1]-1.*center[1]),-(1.*prtLoc[0][0]-1.*center[0]))
if PrtLocAngle<0:
    PrtLocAngle=(np.pi+PrtLocAngle)+np.pi
#Generating a random trigerring zone in polar coordinates for the first trial
#if triggering zone is fixed and centered in the middle of the image
RandomTriggZone=1#If it is 0 there is not random zone, it is a fix disc in the middle of the arena. It is is 1, it will be in a random location
ProhAngleProp=0.1#proportion of 360degrees// 0.25 is equal to 90degrees
if RandomTriggZone<1: 
	centerDiskX=center[0]#-np.sin(finalAngle)*randRadius
	centerDiskY=center[1]#+np.cos(finalAngle)*randRadius
else:
	randRadius=random.random()*(radius-TriggeringZoneRadius)
	randAngle=random.random()*2*np.pi*(1-ProhAngleProp)# substracting the 10% of the total circuference to avoid trigger zones close to the water port
	finalAngle=ProhAngleProp*np.pi+PrtLocAngle+randAngle
	centerDiskX=center[0]-np.cos(finalAngle)*randRadius
	centerDiskY=center[1]+np.sin(finalAngle)*randRadius
mouseTrajectory['trigerZone'].append((centerDiskX,centerDiskY)) 
#passing the variables of the trigger zone to the video player
XTA=Value('i',0)
YTA=Value('i',0)
RTA=Value('i',0)
#dataTrajectory=Array('i', range(2))
XTA.value=int(round(centerDiskX))
YTA.value=int(round(centerDiskY))
RTA.value=int(round(TriggeringZoneRadius))
Xmean=Value('i',prtLoc[0][0])
Ymean=Value('i',prtLoc[0][1])
Xport=Value('i',prtLoc[0][0])
Yport=Value('i',prtLoc[0][1])
Xcirc=Value('i',int(center[0]))
Ycirc=Value('i',int(center[1]))
Rcirc=Value('i',int(round(radius)))
running=Value('i',1)
isRecording=Value('i',1)
TrajLastTrial=[]

#MAIN LOOP STARTS HERE##############################################################
#Video recording variables....
p1=Process(target=runvideo, args=(running, isRecording, Xmean, Ymean, XTA, YTA, RTA, Xport, Yport, Xcirc, Ycirc, Rcirc, ))#Process(target=runvideo, args=(running, Xmean, Ymean, XTA, YTA, RTA, TrajLastTrial, ))
p1.start()

print "[INFO] Behavioral experiment starts"
print "[INFO] Generating random triggering zone"
datatimes['expStart'].append(time.time())
datatimes['GeographPort']=GeographOrientations[GeographRewardPort]
datatimes['correctPortTimeWindow'].append((PhysRewardPort, TimeToReachReward, GeographRewardPort))
datatimes['RewNonRewTimeRatio'].append((RewNonRewTimeRatio))
datatimes['FlagForRecallTest'].append((FlagForRecallTest))
mouseTrajectory['NumTrialsWLightCue'].append((NumTrialsWLightCue))
previosrecord=time.time()
TimeToTriggANewTrial=min(TimeToReachReward,10)
OriginalTime=TimeToReachReward
if extraRewardWStart > 0:
    for i in range(0,(len(listofValves))):
        openWaterPort(listofValves[i-1], timeLengthWTone*compensationFactor[i-1], "Water drop port ")
pinLight=listofLeds[PhysRewardPort-1]
board.digital[pinLight].write(0)
Aux8 = []
AuxAccum = []
FirstTrialWithWater=1
FirstTrialWithAvailWater=0
TimeRead=.1
while time.time() <= (t_end +0.001):
    #Setting the flag for light on the correct port    
    if (len(datatimes['cueTime'])< NumTrialsWLightCue):
        extraLedCue = 1
    else:
        extraLedCue = 0
    #Setting the first trial number with water available
    if (((t_init+ 60*((expLength-FixMinRewTime)*RewNonRewTimeRatio+aclimTimeLength))<time.time()) and (FirstTrialWithAvailWater==0)):
        FirstTrialWithAvailWater=len(datatimes['cueTime'])
    #Checkin if animals enters trigger zone
    if ((t_init + aclimTimeLength*60.)<time.time()) and (np.sqrt(np.square(Xmean.value-centerDiskX)+np.square(Ymean.value-centerDiskY))<TriggeringZoneRadius) and ((timeLastEntry+TimeToTriggANewTrial)<time.time()):
        timeLastEntry=time.time()        
        In=True
        TrajLastTrial=[]
    #Storing the X,Y,t vector with the boolean variable of been in or out the trigger zone    
    dataTrajectory.append((Xmean.value,Ymean.value,time.time(),In))
    TrajLastTrial.append((Xmean.value,Ymean.value))
    if((previosrecord+0.01)<time.time()):
        previosrecord=time.time()
        mouseTrajectory['traj'].append((Xmean.value,Ymean.value,time.time()))
    # Loop that counts licks during the interval between the incorrect port is licked and the next trial is started when the animal steps on the triggering zone
    while (In is False) and (time.time() < t_end) and (time.time()<(cueTime+lickTimeWindow)):#will run until the next trial starts, just to check animal licking between trials            
        if((previosrecord+0.01)<time.time()):
            previosrecord=time.time()
            mouseTrajectory['traj'].append((Xmean.value,Ymean.value,time.time()))        
        for i in range(0,(len(listofPorts))):#Loop over all the water ports
            listofPortsStates[i]=False
            listofPortsStates[i]=listofPorts[i].read()#should be "i" in [i], when testing from all ports               
            if (listofPortsStates[i] is True) and ((timeFirstEntry + timeSampling)<time.time()):#t doesnt count continously licks but every ~timeSamplig time length
                timeFirstEntry=time.time()
                datatimes['timeErrorPort'].append(time.time())
                datatimes['ErrorPortLicked'].append(i+1)
                print 'Licking port outside the time window =',i+1, time.time()
                if (listofPortsStates[PhysRewardPort-1] is True):
                   datatimes['timeErrorCorrectPort'].append(time.time())
        if  ((t_init + aclimTimeLength*60.)<time.time()) and (np.sqrt(np.square(Xmean.value-centerDiskX)+np.square(Ymean.value-centerDiskY))<TriggeringZoneRadius) and ((timeLastEntry+TimeToTriggANewTrial)<time.time()) :
            timeLastEntry=time.time()                 
            In=True
        TimeToTriggANewTrial=min(TimeToReachReward,10)
    #Loading error licks before first trial    
    if((len(datatimes['cueTime'])<2)):#(0<len(datatimes['cueTime'])) and 
        Aux11=np.array(datatimes['ErrorPortLicked'][0:(len(datatimes['ErrorPortLicked']))],dtype='int')
        Aux111=np.abs(np.diff(Aux11))
        Aux1111=np.where(Aux111>0)
        Aux11111=Aux11[Aux1111[0][0:-1]]
    #Saving mouse rajectory
    mouseTrajectory['traj'].append((Xmean.value,Ymean.value,time.time()))            
    
    #Trial starts with the tone + small water drop to cue the port
    toneLength=TimeToReachReward
    p = Process(target=sine_tone_EnvA, args=(toneFreq, toneLength, volume, sample_rate))
    p.start()    
    #Trial starts with tone and light on the correct port
    if extraLedCue > 0:    
        board.digital[pinLight].write(1)
    datatimes['cueTime'].append(time.time())
    cueTime=time.time()
    print 'cueTime =', time.time()
    #Small water drop at correct port to indicate animal where the reward is obtained (useful for the beggining of the trayning)
    if extraRewardWCue > 0:
        openWaterPort(listofValves[PhysRewardPort-1], timeLengthWTone*compensationFactor[PhysRewardPort-1], "Water drop port ")
    #Test if animal licks incorrect port it detects correct port  lick and jump to the next trial
    listofPortsStates[PhysRewardPort-1]=False #listofPorts[PhysRewardPort-1].read()#should be "i" here   
    time.sleep(TimeRead)    
    while (time.time() < t_end) and (listofPortsStates[PhysRewardPort-1] is False) and (time.time()<(cueTime+TimeToReachReward)):
        if((previosrecord+0.01)<time.time()):
            previosrecord=time.time()
            mouseTrajectory['traj'].append((Xmean.value,Ymean.value,time.time()))
        #detection of lick on port 2 while animal needs to lick on port 1 to get the reward    
        for i in range(0,(len(listofPorts))):
            listofPortsStates[i]=False
            listofPortsStates[i]=listofPorts[i].read()#should be "i" here
            if (not(i==(PhysRewardPort-1))) and (listofPortsStates[i] is True) and ((timeFirstEntry + timeSampling)<time.time()):#should be "i" here
                timeFirstEntry=time.time()
                datatimes['timeLickIncorrectPort'].append(time.time())
                datatimes['IncorrectPortLicked'].append(i+1)
                print 'incorrect port =', i+1, time.time()
        listofPortsStates[PhysRewardPort-1]=listofPorts[PhysRewardPort-1].read()
        time.sleep(TimeRead)
        if (listofPortsStates[PhysRewardPort-1] is True) and ((timeFirstEntry2 + timeSampling)<time.time()):
            timeFirstEntry2=time.time()
            datatimes['timeLickCorrectPort'].append(time.time())
            datatimes['timeDelayCorrectLick'].append(time.time()-cueTime)
            mouseTrajectory['trigerTime'].append(time.time())
            print 'correct port =', PhysRewardPort, time.time()
            if ((t_init+ 60*((expLength-FixMinRewTime)*RewNonRewTimeRatio+aclimTimeLength))<time.time()):            
                if Water is False:
                    if ((t_init+ 60.0*(((expLength-FixMinRewTime)*RewNonRewTimeRatio)+aclimTimeLength))<time.time()):                        
                        datatimes['waterStart'].append((time.time(),1))# Second variable equal to 1 if time is more than RewNonRewTimeRatio of the experiment                    
                    else:
                        datatimes['waterStart'].append((time.time(),2))# Second variable equal to 2 if trial number is larger than RewNonRewMinTrials
                    Water=True
                    FirstTrialWithWater=len(datatimes['cueTime'])
                    print colored('[INFO] WATER IS DELIVERED FROM NOW ON','blue')
                    #dataPlotCorrectPort=[]
                openWaterPort(listofValves[PhysRewardPort-1], timeLengthWReward*compensationFactor[PhysRewardPort-1], "[INFO] Water is been delivered")                
                p.terminate()
                board.digital[pinLight].write(0)
            if (FirstCorrect is False):
                datatimes['timeLickFirstCorrect'].append(time.time())
                datatimes['timeDelayFirstCorrectLick'].append(time.time()-cueTime)
                FirstCorrect = True
        #Reset Port state to exit loop    
        if (time.time()<(t_init+ 60*((expLength-FixMinRewTime)*RewNonRewTimeRatio+aclimTimeLength))):
            listofPortsStates[PhysRewardPort-1]=False
    board.digital[pinLight].write(0)    
    #Generates a Random trigger zone after the trial is finished
    if RandomTriggZone<1: 
        centerDiskX=center[0]#-np.sin(finalAngle)*randRadius
        centerDiskY=center[1]#+np.cos(finalAngle)*randRadius
    else:
        randRadius=r = (radius-TriggeringZoneRadius)*np.sqrt(random.random())#random.random()*(radius-TriggeringZoneRadius)
        randAngle=random.random()*2*np.pi*(1-ProhAngleProp)# substracting the 10% of the total circuference to avoid trigger zones close to the water port
        finalAngle=ProhAngleProp*np.pi+PrtLocAngle+randAngle
        centerDiskX=center[0]-np.cos(finalAngle)*randRadius
        centerDiskY=center[1]+np.sin(finalAngle)*randRadius
    XTA.value=int(round(centerDiskX))
    YTA.value=int(round(centerDiskY))
    #dataTriggerZone.append((centerDiskX, centerDiskY))
    mouseTrajectory['trigerZone'].append((centerDiskX,centerDiskY)) 
    print "[INFO] Generating a new random triggering zone"
    TimeToTriggANewTrial=min(TimeToReachReward,0)        
    #Re-setting the animal in/out state variable
    In=False
    FirstCorrect=False
    
    #Adding trials with water
    if (Water is True ) and (OldLength<len(datatimes['timeDelayFirstCorrectLick'])):
        OldLength=len(datatimes['timeDelayFirstCorrectLick'])
        ww=ww+1  
    #Plotting partial results
    dataPlotTrials.append(len(datatimes['cueTime']))#Total number of trials
    dataPlotErrorPort.append(len(datatimes['timeErrorPort']))#number of trials where lick was outside the licking time window
    dataPlotCorrectPort.append(len(datatimes['timeLickFirstCorrect']))#number of trials where one lick was achieved during the tone (only the first)
    dataPlotAllCorrectPort.append(len(datatimes['timeLickCorrectPort']))#number of licks were achieved during the tone
    dataPlotIncorrectPort.append(len(datatimes['timeLickIncorrectPort']))#number of lickes in the incorrect port
    dataPlotErrorCorrectPort.append(len(datatimes['timeErrorCorrectPort']))#number of trials where lick was outside the licking time window
    #dataPlotIncorrectInCorrectTrials.append(len(datatimes['IncorrectPortLicked']));
    if 0<len(datatimes['timeErrorCorrectPort']):        
        XX=[]
        XX=[x for x in datatimes['timeErrorCorrectPort'] if datatimes['cueTime'][-1] <= x < (datatimes['cueTime'][-1] + TimeToReachReward)]                       
        if 0<len(XX):
            Aux8.extend([PhysRewardPort]*len(XX))
    Aux0=np.array(dataPlotTrials,dtype='float')#Accumulative sum of trials
    Aux00=np.array(dataPlotErrorPort,dtype='float')#Accumulative sum of licks outsie the licking time window
    Aux1=np.array(dataPlotCorrectPort,dtype='float')#Accumulative sum of correct trials
    Aux2=np.array(dataPlotIncorrectPort,dtype='float')#Accumulative sum of licks before getting to the correct port  
    Aux6=np.array(dataPlotErrorCorrectPort,dtype='float')#Accumulative sum of licks in correct port outside the tone (time-window)
    Aux5=datatimes['timeDelayFirstCorrectLick'][0:(len(datatimes['timeDelayFirstCorrectLick']))]#Reaction time
    Aux88=datatimes['IncorrectPortLicked'][0:(len(datatimes['IncorrectPortLicked']))]#Index of the incorrect port licked        
    Aux99=datatimes['ErrorPortLicked'][0:(len(datatimes['ErrorPortLicked']))]#Index of the incorrect port licked       
    if (0<len(datatimes['timeLickCorrectPort'])) and (datatimes['cueTime'][-1]<datatimes['timeLickCorrectPort'][-1]):
        Aux7=datatimes['IncorrectPortLicked'][0:(len(datatimes['IncorrectPortLicked']))]#np.array(dataPlotIncorrectInCorrectTrials,dtype='float')#Number of incorrect licks during correct trials
        Aux77=np.abs(np.diff(Aux7))
        Aux777=np.where(Aux77>0)
        Aux7777=np.size(Aux777)#Number of incorrect ports, before correct port
        AuxAccum.append(Aux7777)    
        kkk=kkk+1
    XYcoord=[]        
    XYcoord=np.array(TrajLastTrial,dtype='float')#XYcoord=np.array(dataTrajectory,dtype='float')
    DistribRate=[]
    DistribRate, xedges, yedges=np.histogram2d(XYcoord[:,0],XYcoord[:,1], bins=10)
    FirRatDens=[];
    FirRatDens=np.divide(DistribRate,(np.nansum(np.nansum(DistribRate))));
    m_1=np.append(m_1,np.nansum(np.nansum(FirRatDens*FirRatDens)));         
    if (1<len(datatimes['cueTime'])) and (np.mod(len(datatimes['cueTime']),freqPlot)==0):# and (1<len(datatimes['timeErrorPort'])):        
        kk=kk+1
        if 1<kk:
            x = np.ones(len(Aux0))
            x2 = np.ones(len(Aux5))
            yCTT = (1./8.)*x;
            yRT = 6*x2;
            ax4 = fig.add_subplot(241)
            ax4.cla()
            ax4.set_title('Inverse of coverage', fontsize=10)
            ax4.set_ylim([0,1])            
            for tick in ax4.xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            for tick in ax4.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)            
            ax4.plot((np.arange(len(m_1))+1),m_1,'go-')
            ax3 = fig.add_subplot(242)
            ax3.cla()
            for tick in ax3.xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            for tick in ax3.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            ax3.set_title('Reaction time', fontsize=10)                        
            ax3.plot((np.arange(len(Aux5))+1),Aux5,'bo-',(np.arange(len(Aux5))),yRT,'b--')
            ax3.text(2,1,'Mean React. Time = {:.2f}'.format(np.mean(Aux5)), fontsize=8)                
            ax5 = fig.add_subplot(243)
            if 1<kkk:               
                ax5.cla()#ax5.set_ylim([0,np.divide(AuxAccum,Aux0[-1])])                 
                for tick in ax5.xaxis.get_major_ticks():
                    tick.label.set_fontsize(8)
                for tick in ax5.yaxis.get_major_ticks():
                    tick.label.set_fontsize(8) 
                ax5.set_title('# Incorrect ports|Correct trials', fontsize=10)                               
                ax5.plot((np.arange(len(AuxAccum))+1),np.divide(AuxAccum,Aux0[-1]),'mo-')
                ax5.text(3.5,0.05,'#Incorr|Corr. = {:.2f}'.format(np.mean(np.divide(AuxAccum,Aux0[-1]))), fontsize=8)
            ax1 = fig.add_subplot(245)
            ax1.cla()
            for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            ax1.set_ylim([0,1.1])
            ax1.set_title('Accum Perform. (CT/TT)', fontsize=10)                                     
            ax1.plot((np.arange(len(Aux1))+1),Aux1/Aux0,'ko-',(np.arange(len(yCTT))+1),yCTT,'k--')            
            ax1.text(3.5,0.2,'Mean Perform. = {:.2f}'.format(np.mean(Aux1/Aux0)), fontsize=8)            
            ax2 = fig.add_subplot(246)
            ax2.cla()
            for tick in ax2.xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            for tick in ax2.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)            
            ax2.axis([0,(len(Aux00)+1),0,3.1*(max(max(Aux00/7),max(Aux6))+1.0)])  
            ax2.set_title('Accum: #Errors, #Persistant', fontsize=10)                                      
            ax2.plot((np.arange(len(Aux00))+1),Aux00/7,'ro--',(np.arange(len(Aux6))+1),Aux6,'ko:')
            ax2.plot([FirstTrialWithAvailWater, FirstTrialWithAvailWater], [0, max(max(Aux00/7),max(Aux6))], ':b', lw=0.5)
            ax2.plot([FirstTrialWithWater, FirstTrialWithWater], [0, max(max(Aux00/7),max(Aux6))], ':g', lw=0.5)
            ax2.text(4,200,'#Trials w/H2O = {:.2f}'.format(ww), fontsize=8)                
            ax22 = fig.add_subplot(247)            
            ax22.cla()
            for tick in ax22.xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            for tick in ax22.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            #ax22.axis([0,kk+1,0,1.1*max(Aux2)])  
            ax22.set_xlim([-4.5,5.5])    
            ax22.set_title('Histogram of Recall before water', fontsize=10)
            if(FirstTrialWithWater<2):
                if(0<len(Aux5)):
                    AuxMem=np.concatenate((Aux8,Aux88,Aux99,np.ones(len(Aux5))*PhysRewardPort))
                else:
                    AuxMem=np.concatenate((Aux8,Aux88,Aux99))
            else:
                XX00=[yy for yy in datatimes['timeLickFirstCorrect'] if yy < (np.array(datatimes['waterStart'])[0,0])]
                AuxFirstCorr=np.array(np.ones(len(XX00))*PhysRewardPort,dtype='float')
                XX0=[y for y in datatimes['timeLickCorrectPort'] if y < (np.array(datatimes['waterStart'])[0,0])]
                AuxCorr=np.array(np.ones(len(XX0))*PhysRewardPort,dtype='float')
                XX1=[z for z in datatimes['timeLickIncorrectPort'] if z < (np.array(datatimes['waterStart'])[0,0])]
                AuxInc=np.array(datatimes['IncorrectPortLicked'])[:(len(XX1))]
                XX2=[w for w in datatimes['timeErrorCorrectPort'] if w < (np.array(datatimes['waterStart'])[0,0])]
                AuxErrCorr=np.array(np.ones(len(XX2))*PhysRewardPort,dtype='float')
                XX3=[zz for zz in datatimes['timeErrorPort'] if zz < (np.array(datatimes['waterStart'])[0,0])]
                AuxErr=np.array(datatimes['ErrorPortLicked'])[:(len(XX3))]
                AuxMem=np.concatenate((AuxCorr,AuxInc,AuxErrCorr,AuxErr))#AuxMem=np.concatenate((AuxFirstCorr,AuxCorr,AuxInc,AuxErrCorr,AuxErr))
            AuxMemMem=[]            
            AuxMemMem=np.array(np.mod(AuxMem-PhysRewardPort,8))            
            AUX7=[]
            AUX7=np.where(AuxMemMem==7)
            AuxMemMem[AUX7[0][:]]=-1
            AUX6=[]
            AUX6=np.where(AuxMemMem==6)
            AuxMemMem[AUX6[0][:]]=-2
            AUX5=[]
            AUX5=np.where(AuxMemMem==5)
            AuxMemMem[AUX5[0][:]]=-3
            if 0<len(AuxMemMem):            
                ax22.hist(AuxMemMem,bins=(-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5),color='blue')
            ax33 = fig.add_subplot(244)            
            ax33.cla()       
            for tick in ax33.xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            for tick in ax33.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)            
            ax33.set_xlim([-4.5,5.5])    
            ax33.set_title('Histogram of Recall before first trial', fontsize=10)    
            Aux=[]
            Aux=np.array(np.mod(Aux11-PhysRewardPort,8))            
            AUX7=[]            
            AUX7=np.where(Aux==7)
            Aux[AUX7[0][:]]=-1
            AUX6=[]
            AUX6=np.where(Aux==6)
            Aux[AUX6[0][:]]=-2
            AUX5=[]
            AUX5=np.where(Aux==5)
            Aux[AUX5[0][:]]=-3
            ax33.hist(Aux,bins=(-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5),color='green')
            ax44 = fig.add_subplot(248)            
            ax44.cla()       
            for tick in ax44.xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            for tick in ax44.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)            
            ax44.set_xlim([-4.5,5.5])    
            ax44.set_title('Histogram licks On+Off Tone', fontsize=10)    
            AUX=[]
            AUX=np.array(np.mod(np.concatenate((Aux8,Aux88,Aux99))-PhysRewardPort,8))            
            AUX7=[]
            AUX7=np.where(AUX==7)
            AUX[AUX7[0][:]]=-1
            AUX6=[]
            AUX6=np.where(AUX==6)
            AUX[AUX6[0][:]]=-2
            AUX5=[]
            AUX5=np.where(AUX==5)
            AUX[AUX5[0][:]]=-3
            ax44.hist(AUX,bins=(-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5),color='grey')                 
            
            plt.show()
            plt.pause(0.0001)
#if extraRewardEND > 0:
#	 openWaterPort(listofValves[PhysRewardPort-1], timeLengthWReward*compensationFactor[PhysRewardPort-1], "[INFO] Water is been delivered")    
                        
#video recording release                
running.value=0
#Release the board
board.digital[pinLight].write(0)
board.exit()
#Saving data time stems
data={}
timeEndExp = time.time()
data['timestamp'] = time.time()
data['datatimes'] = datatimes
data['mouseTrajectory'] = mouseTrajectory

#building the time stem of the end of the experiment as a string
valueTime = datetime.datetime.fromtimestamp(timeEndExp)
path=  'C:/Users/Dalmau/Documents/Python Scripts/DataOutput/'
animalName = AnimalNames[AnimalNumber-1] #sys.argv[1]
f_myfile = open(path+animalName+valueTime.strftime('_%Y-%m-%d_%H-%M-%S_rec.dat'), 'wb')

#dump all the data into a dictionary
pickle.dump(data, f_myfile)
f_myfile.close()
#Save figure
fig.savefig(path+animalName+valueTime.strftime('_%Y-%m-%d_%H-%M-%S_rec'))
scipy.io.savemat(path+animalName+valueTime.strftime('_%Y-%m-%d_%H-%M-%S_rec.mat'), mdict={'data': data})
p1.terminate
p.terminate