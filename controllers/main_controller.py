# Import required libraries
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
from controllers import video_controller
from controllers.sound_controller import sine_tone
import tables


class MainController:

    def __init__(self, data):
        box = data['box']
        selected_mode = data['mode']
        stage = data['stage']

        input_mat = tables.open_file('OutputMatrixPorts12Batch.mat')
        self.PortsMatrix = input_mat.root.A
        self.TimesRecallVector = input_mat.root.TimeDelayRecall

        self.AnimalNames = ["Mouse4103", "Mouse4104", "Mouse4109", "Mouse4110", "Mouse4111", "Mouse4112", "Mouse4113",
                            "Mouse4114", "Mouse4115", "Mouse4117", "Mouse4118", "Mouse4119", "Mouse4120", "Mouse4121",
                            "Mouse4122", "Mouse4123", "Mouse4124", "Mouse4125", "Mouse4126", "Mouse4127", "Mouse4128"]

        # Initialize the list of reference points and a boolean indicating whether cropping is performed
        self.prtLoc = []
        self.refPt = []
        self.cropping = False
        self.image = []
        self.PercenRadiusDisc = 0.23

        # General variables:
        self.ExpDay = int(data['exp_day'])
        self.AnimalNumber = int(data['animal_number'])
        self.StageLevel = int(data['stage'])
        self.DrugType = int(data['drug_type'])
        self.Dose = data['dose']

        # Box variables:
        if box == 1:
            self.video_num = 0
            self.audio_output_index = 3
            self.toneFreq = 10560
            self.compensationFactor = (1.25, 2., .83, 2.9, 2.03, .83, 2.73, .8)
            self.port = 'COM1'

        elif box == 2:
            self.video_num = 1
            self.audio_output_index = 1
            self.toneFreq = 7040
            self.compensationFactor = (2.5, 1.43, 1.25, 1.43, 1.5, 1.43, 2.5, 1.11)
            self.port = 'COM15'

        elif box == 3:
            self.video_num = 0
            self.audio_output_index = 2
            self.toneFreq = 4400
            self.compensationFactor = (.91, .83, 2.91, 2.83, 1.43, .83, 1.43, 1.71)
            self.port = 'COM3'

        # Mode variables:
        if selected_mode == 1:
            self.mode = 'Train'
            self.file_name = '_%Y-%m-%d_%H-%M-%S_trn.avi'
            self.NumOfRotations = 1
        else:
            self.mode = 'Recall'
            self.file_name = '_%Y-%m-%d_%H-%M-%S_rec.avi'
            self.NumOfRotations = 2

        # Stage variables:
        if stage == 1:
            self.TimeToReachReward = 110
            self.NumTrialsWLightCue = 1000
            self.MinNumTrials = 30
            self.extraRewardWCue = True
            self.extraRewardWStart = True

        elif stage == 1.2:
            self.TimeToReachReward = 10
            self.NumTrialsWLightCue = 2
            self.MinNumTrials = 30
            self.extraRewardWCue = False
            self.extraRewardWStart = False
            self.FlagRandDeliver = False

        elif stage == 2:
            self.TimeToReachReward = 4
            self.NumTrialsWLightCue = 0
            self.MinNumTrials = 30
            self.extraRewardWCue = False
            self.extraRewardWStart = False
            self.FlagRandDeliver = False

        elif stage == 2.2:
            self.TimeToReachReward = 4
            self.NumTrialsWLightCue = 0
            self.MinNumTrials = 20
            self.extraRewardWCue = False
            self.extraRewardWStart = False
            self.FlagRandDeliver = True

        elif stage == 3:
            self.TimeToReachReward = 4
            self.NumTrialsWLightCue = 0
            self.MinNumTrials = 30
            self.MinPerf = 0.95
            self.extraRewardWCue = False
            self.extraRewardWStart = False
            self.FlagRandDeliver = True

        elif stage == 4:
            self.TimeToReachReward = 4
            self.NumTrialsWLightCue = 0
            self.MinNumTrials = 30
            self.extraRewardWCue = False
            self.extraRewardWStart = False
            self.FlagRandDeliver = True

    # Define custom function in Python to perform Blink action (opening water port)
    def _open_water_port(self, pin, timeLength, message):  # any oin number from 22 to 53
        print(f'[PORT] {message}')
        self.board.digital[pin].write(1)
        sleep(timeLength)
        self.board.digital[pin].write(0)

    def run_experiment(self):
        def set_env_contour_and_port_locat(event, x, y, flags, param):  # All arguments required
            # Ff the left mouse button was clicked, record the starting (x, y)
            # coordinates and indicate that self.cropping is being performed

            if event == cv2.EVENT_LBUTTONDOWN:
                print('[EVENT] Button down')
                self.refPt = [(x, y)]
                self.cropping = True

            # Check to see if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                print('[EVENT] Button up')
                # record the ending (x, y) coordinates and indicate that the self.cropping operation is finished
                self.refPt.append((x, y))
                self.cropping = False
                # draw a circle around the region of interest
                center = np.mean(self.refPt, 0)
                radius = np.sqrt(np.sum(np.square(np.diff(self.refPt, 1)))) / 2
                cv2.circle(self.image, (int(round(center[0])), int(round(center[1]))), int(round(radius)), (0, 255, 0),
                           2)  # cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
                cv2.imshow("self.image", self.image)
            # if right mouse button is clicked then port location is recorded
            if event == cv2.EVENT_RBUTTONDBLCLK:  # cv2.EVENT_LBUTTONDBLCLK:
                print('[EVENT] Right doble click')
                self.prtLoc = [(x, y)]
                cv2.circle(self.image, (self.prtLoc[0]), 2, (255, 0, 0),
                           2)  # cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
                cv2.imshow("self.image", self.image)

        # initialize the video stream and allow the camera sensor to warmup
        print(f'[INFO] Taking one picture of the environment {self.video_num}')
        cap = cv2.VideoCapture(self.video_num, cv2.CAP_DSHOW)
        time.sleep(3.0)
        ret, frame = cap.read()
        # frame = cv2.flip(frame,0)
        if ret:
            # converting fram in a gray scale picture
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray = cv2.GaussianBlur(gray, (11, 11), 0)
        cap.release()
        # Clone it and set up the mouse callback function
        self.image = gray.copy()
        cv2.namedWindow("self.image")
        # Determining the countour of the spatial enclosure
        cv2.setMouseCallback("self.image", set_env_contour_and_port_locat)
        # keep looping until the 'q' key is pressed
        while len(self.refPt) < 2:
            # display the self.image and wait for a keypress
            cv2.imshow("self.image", gray)
            key = cv2.waitKey(1) & 0xFF
        cv2.waitKey(0)
        # Save data for next step
        center = np.mean(self.refPt, 0)
        radius = np.sqrt(np.sum(np.square(np.diff(self.refPt, 1)))) / 2
        print("[INFO] Giving values to: Arenas center =", center, ", Arenas radius =", radius)
        print("[INFO] Port locations =", self.prtLoc[0])
        # Release everything if job is finished
        cv2.destroyAllWindows()
        ###################################################################################################
        print("[INFO] Loading Arduino")
        # Global variables
        volume = 1
        sample_rate = 22050
        lickTimeWindow = 840  # float('inf') #we might put a time window (5sec)
        timeLengthWTone = 0.040  # length of open valve during the tone
        timeLengthWReward = 0.085  # 0.025#0.02#Length of the open valve during licking
        timeSampling = 0.05  # This could be dictated by the DAQ from mini-microscope

        # Rewarded self.port must be changed every day for environment A and keep constant for environment B

        PhysRewardPortYesterday = int(self.PortsMatrix[self.AnimalNumber - 1][self.ExpDay - 2])

        # Port from the box that will be located in the physical position called GeographRewardPort
        PhysRewardPort = int(self.PortsMatrix[self.AnimalNumber - 1][self.ExpDay - 1])
        self.PortsMatrix.close()

        SecondPort = random.randrange(1, 8)

        # From 1 to 8 correspond to rewarded self.port with respect to the arena for each of the three rotations
        SecondPhysRewardPort = [SecondPort, SecondPort, SecondPort]

        # Each of the numbners between [1,2,3,4,5,6,7,8] correspond o to the 8 geographical orientations
        GeographRewardPort = PhysRewardPort

        # One of the following generically orientations respect to the box ['','']
        GeographOrientations = ['NE', 'EE', 'SE', 'SS', 'SO', 'OO', 'NO', 'NN']

        # ONLY FOR RECALL
        FlagForRecallTest = 1
        FixMinRewTime = 7.

        # Three minutes before the end of the experiment with no reward:
        TimeBefEndNoRew = 0

        # Extended time to reach reward in case animals do not reach the rewarded port in the self.TimeToReachReward time:
        ExtendedTime = 20

        # Minutes after star that if the performance is below 10% the animals get help:
        EmergencyTime = 10

        # Associate port and board with pyFirmata:
        self.board = pyfirmata.ArduinoMega(self.port)
        print('[INFO] Arduino ready, plug cable')

        # Use iterator thread to avoid buffer overflow:
        it = pyfirmata.util.Iterator(self.board)
        it.start()
        time.sleep(1)

        # Define i/o pins (giving roles to pins: i.e. d=digital, 7=pin number, i=input):
        listofPorts = [self.board.get_pin('d:22:i'), self.board.get_pin('d:24:i'), self.board.get_pin('d:26:i'),
                       self.board.get_pin('d:28:i'),
                       self.board.get_pin('d:30:i'), self.board.get_pin('d:32:i'), self.board.get_pin('d:34:i'),
                       self.board.get_pin('d:36:i')]

        listofValves = [39, 41, 43, 45, 47, 49, 51, 53]
        listofPortsStates = [False, False, False, False, False, False, False, False]
        listofLeds = [23, 25, 27, 29, 31, 33, 35, 37]

        # Initializing the state of the animal
        In = False
        Water = False

        # Defining libraries for data storage
        datatimes = {'cueTime': [], 'timeLickIncorrectPort': [], 'timeLickCorrectPort': [], 'IncorrectPortLicked': [],
                     'timeErrorPort': [], 'ErrorPortLicked': [], 'timeErrorCorrectPort': [], 'expStart': [],
                     'waterStart': [], 'RewNonRewTimeRatio': [], 'FlagForRecallTest': [], 'timeDelayCorrectLick': [],
                     'correctPortTimeWindow': [], 'trialsWithHelp': [], 'TimeSecondPort': [], 'TimeNoMoreReward': [],
                     'timeLickFirstCorrect': [], 'timeDelayFirstCorrectLick': []}

        if self.mode == 'Recall':
            datatimes.update({'trialsWithHelp': [], 'TimeSecondPort': [], 'TimeNoMoreReward': []})

        dataPlotAllCorrectPort = []
        dataPlotCorrectPort = []
        dataPlotIncorrectPort = []
        dataPlotErrorCorrectPort = []
        dataPlotTrials = []
        dataPlotErrorPort = []
        dataTrajectory = []

        mouseTrajectory = {'traj': [], 'trigerZone': [], 'trigerTime': [], 'BoxDimentions': [(center, radius, 'EnvA')],
                           'CorrectPortLocation': [self.prtLoc], 'TriggerZoneRatio': [], 'self.NumTrialsWLightCue': [],
                           'Stage': [],
                           'self.DrugType': [], 'Dose': []}
        if self.mode == 'Train':
            mouseTrajectory.update({'TimeBefEndNoRew': []})
        elif self.mode == 'Recall':
            mouseTrajectory.update({'TimeLength': [], 'AcclimatTime': []})

        # Performance score plotting variables.
        kk = 0
        kkk = 0
        ww = 0
        rot = 0
        freqPlot = 3
        fig = plt.figure(1, figsize=(13, 6))
        m_1 = []

        # While loop temporal variables
        # We fix the time of the beginning of the experiment
        t_init = time.time()

        # TRAIN MODE
        RotationFreq = 40. / (self.NumOfRotations + 1)  # (in minutes)
        NonRewardPeriodStart = 4

        # Time Ratio of the experiment until reach rewarded trials
        RewNonRewTimeRatio = random.random()

        # Otherwise maintains the value assigned on self.TimesRecallVector:
        if self.FlagRandDeliver:
            RewNonRewTimeRatio = 0
        FirstCorrect = False
        aclimTimeLength = 0.1  # In minutes
        SleepTime = 0  # In seconds

        if self.mode == 'Train':
            expLength = 10.0  # (in minutes)
            t_end = t_init + 60 * expLength + SleepTime * self.NumOfRotations
        elif self.mode == 'Recall':
            expLength = aclimTimeLength + RotationFreq * (self.NumOfRotations + 1)
            t_end = t_init + 60 * (expLength + aclimTimeLength)

        timeFirstEntry = t_init
        timeFirstEntry2 = t_init
        cueTime = t_init
        timeLastEntry = t_init
        timeLastRotation = t_init

        # Size of triggering zone
        mouseTrajectory['TriggerZoneRatio'].append(self.PercenRadiusDisc)
        mouseTrajectory['Stage'].append(self.StageLevel)
        mouseTrajectory['self.DrugType'].append(self.DrugType)

        if self.mode == 'Train':
            mouseTrajectory['TimeBefEndNoRew'].append(TimeBefEndNoRew)
        elif self.mode == 'Recall':
            mouseTrajectory['Dose'].append((self.Dose))
            mouseTrajectory['TimeLength'].append(expLength)
            mouseTrajectory['AcclimatTime'].append(aclimTimeLength)

        TriggeringZoneRadius = radius * self.PercenRadiusDisc

        # Transforming water port location from cartesian to polar coordinates:
        self.prtLocAngle = np.arctan2((1. * self.prtLoc[0][1] - 1. * center[1]),
                                      -(1. * self.prtLoc[0][0] - 1. * center[0]))
        if self.prtLocAngle < 0:
            self.prtLocAngle = (np.pi + self.prtLocAngle) + np.pi

        # Generating a random triggering zone in polar coordinates for the first trial
        # if triggering zone is fixed and centered in the middle of the self.image

        # If it is 0 there is not random zone, it is a fix disc in the middle of the arena.
        # It is is 1, it will be in a random location.
        RandomTriggZone = 1

        # Proportion of 360degrees// 0.25 is equal to 90degrees
        ProhAngleProp = 0.1
        if RandomTriggZone < 1:
            centerDiskX = center[0]
            centerDiskY = center[1]
        else:
            randRadius = random.random() * (radius - TriggeringZoneRadius)

            # Subtracting the 10% of the total circumference to avoid trigger zones close to the water port
            randAngle = random.random() * 2 * np.pi * (1 - ProhAngleProp)
            finalAngle = ProhAngleProp * np.pi + self.prtLocAngle + randAngle
            centerDiskX = center[0] - np.cos(finalAngle) * randRadius
            centerDiskY = center[1] + np.sin(finalAngle) * randRadius

        mouseTrajectory['trigerZone'].append((centerDiskX, centerDiskY))

        # passing the variables of the trigger zone to the video player
        XTA = Value('i', 0)
        YTA = Value('i', 0)
        RTA = Value('i', 0)

        XTA.value = int(round(centerDiskX))
        YTA.value = int(round(centerDiskY))
        RTA.value = int(round(TriggeringZoneRadius))
        Xmean = Value('i', self.prtLoc[0][0])
        Ymean = Value('i', self.prtLoc[0][1])
        Xport = Value('i', self.prtLoc[0][0])
        Yport = Value('i', self.prtLoc[0][1])
        Xcirc = Value('i', int(center[0]))
        Ycirc = Value('i', int(center[1]))
        Rcirc = Value('i', int(round(radius)))
        running = Value('i', 1)
        isRecording = Value('i', 1)
        TrajLastTrial = []

        # MAIN LOOP STARTS HERE##############################################################
        # Video recording variables....
        p1 = Process(target=video_controller.run_video,
                     args=(running, self.video_num, self.file_name, isRecording, Xmean, Ymean, XTA, YTA, RTA, Xport, Yport, Xcirc, Ycirc, Rcirc))
        p1.start()

        print("[INFO] Behavioral experiment starts")
        print("[INFO] Generating random triggering zone")

        datatimes['expStart'].append(time.time())
        datatimes['GeographPort'] = GeographOrientations[GeographRewardPort - 1]
        datatimes['correctPortTimeWindow'].append((PhysRewardPort, self.TimeToReachReward, GeographRewardPort))
        datatimes['RewNonRewTimeRatio'].append(RewNonRewTimeRatio)

        if self.mode == 'Train':
            datatimes['correctPortTimeWindow'].append(SecondPhysRewardPort)
            datatimes['FlagForRecallTest'].append(self.FlagRandDeliver)

        if self.mode == 'Recall':
            datatimes['FlagForRecallTest'].append(FlagForRecallTest)

        mouseTrajectory['self.NumTrialsWLightCue'].append(self.NumTrialsWLightCue)
        previosrecord = time.time()
        TimeToTriggANewTrial = min(self.TimeToReachReward, 10)
        OriginalTime = self.TimeToReachReward
        if self.extraRewardWStart and self.ExpDay == 1:
            for i in range(0, (len(listofValves))):
                self._open_water_port(listofValves[i - 1], timeLengthWTone * self.compensationFactor[i - 1],
                                      "Water drop port ")
        pinLight = listofLeds[PhysRewardPort - 1]
        self.board.digital[pinLight].write(0)
        Aux8 = []
        AuxAccum = []

        # ONLY TRAIN
        AccumPerf = ([0] * self.MinNumTrials)

        FirstTrialWithWater = 1

        # ONLY RECALL
        FirstTrialWithAvailWater = 0

        TimeRead = .1
        while time.time() <= (t_end + 0.001) and (self.mode == 'Recall' or
                                                  (self.mode == 'Train' and (
                                                          np.mean(AccumPerf[-self.MinNumTrials:]) < self.MinPerf))):
            # Stopping the code for the experimenter to rotate the environment
            if self.mode == 'train' and (timeLastRotation + 60.0 * (aclimTimeLength + RotationFreq)) < time.time():
                print(colored(
                    '[INFO] Turn off the light Rotate environment in (SleepTime) seconds, then turn on the light again',
                    'green'))
                isRecording.value = 0
                PhysRewardPort = SecondPhysRewardPort[rot]
                rot = rot + 1
                time.sleep(SleepTime)
                timeLastRotation = time.time()
                aclimTimeLength = 0
                isRecording.value = 1
                datatimes['TimeSecondPort'].append(time.time())

            if (np.mean(AccumPerf[-self.MinNumTrials:]) > self.MinPerf) and (rot < 1):
                print('[INFO] Second Port', SecondPort)
                PhysRewardPort = SecondPhysRewardPort[rot]
                rot = rot + 1
                datatimes['TimeSecondPort'].append(time.time())

            # Setting the flag for light on the correct port
            extraLedCue = len(datatimes['cueTime']) < self.NumTrialsWLightCue

            # Setting the first trial number with water available
            if (self.mode == 'Recall'
                    and ((t_init + 60 * (
                            (expLength - FixMinRewTime) * RewNonRewTimeRatio + aclimTimeLength)) < time.time()
                         and FirstTrialWithAvailWater == 0)):
                FirstTrialWithAvailWater = len(datatimes['cueTime'])

            # Checkin if animals enters trigger zone
            if (((t_init + aclimTimeLength * 60.) < time.time())
                    and (np.sqrt(np.square(Xmean.value - centerDiskX) + np.square(
                        Ymean.value - centerDiskY)) < TriggeringZoneRadius)
                    and ((timeLastEntry + TimeToTriggANewTrial) < time.time())):
                timeLastEntry = time.time()
                In = True
                TrajLastTrial = []

            # Storing the X,Y,t vector with the boolean variable of been in or out the trigger zone
            dataTrajectory.append((Xmean.value, Ymean.value, time.time(), In))
            TrajLastTrial.append((Xmean.value, Ymean.value))
            if (previosrecord + 0.01) < time.time():
                previosrecord = time.time()
                mouseTrajectory['traj'].append((Xmean.value, Ymean.value, time.time()))

            # Loop that counts licks during the interval between the incorrect port is licked and the next trial is
            # started when the animal steps on the triggering zone.
            # Will run until the next trial starts, just to check animal licking between trials.
            while not In and time.time() < t_end and time.time() < (cueTime + lickTimeWindow):
                if (previosrecord + 0.01) < time.time():
                    previosrecord = time.time()
                    mouseTrajectory['traj'].append((Xmean.value, Ymean.value, time.time()))
                for i in range(0, (len(listofPorts))):  # Loop over all the water ports
                    listofPortsStates[i] = False
                    listofPortsStates[i] = listofPorts[i].read()  # should be "i" in [i], when testing from all ports
                    if (listofPortsStates[i] is True) and ((
                                                                   timeFirstEntry + timeSampling) < time.time()):  # t doesnt count continously licks but every ~timeSamplig time length
                        timeFirstEntry = time.time()
                        datatimes['timeErrorPort'].append(time.time())
                        datatimes['ErrorPortLicked'].append(i + 1)
                        print('[ANIMAL] Licking port outside the time window =', i + 1, time.time())
                        if (listofPortsStates[PhysRewardPort - 1] is True):
                            datatimes['timeErrorCorrectPort'].append(time.time())
                if ((t_init + aclimTimeLength * 60.) < time.time()
                        and (np.sqrt(np.square(Xmean.value - centerDiskX) + np.square(
                            Ymean.value - centerDiskY)) < TriggeringZoneRadius)
                        and (timeLastEntry + TimeToTriggANewTrial) < time.time()):
                    timeLastEntry = time.time()
                    In = True
                TimeToTriggANewTrial = min(self.TimeToReachReward, 10)

            # Loading error licks before first trial
            if len(datatimes['cueTime']) < 2:
                Aux11 = np.array(datatimes['ErrorPortLicked'][0:(len(datatimes['ErrorPortLicked']))], dtype='int')
                Aux111 = np.abs(np.diff(Aux11))
                Aux1111 = np.where(Aux111 > 0)
                Aux11111 = Aux11[Aux1111[0][0:-1]]

            # Saving mouse trajectory
            mouseTrajectory['traj'].append((Xmean.value, Ymean.value, time.time()))

            # Trial starts with the tone + small water drop to cue the self.port
            toneLength = self.TimeToReachReward
            print('[AUDIO] Start process')
            p = Process(target=sine_tone, args=(self.audio_output_index, self.toneFreq, toneLength, volume, sample_rate))
            p.start()

            # Trial starts with tone and light on the correct self.port
            if extraLedCue:
                self.board.digital[pinLight].write(1)
            datatimes['cueTime'].append(time.time())
            cueTime = time.time()
            print('[INFO] cueTime =', time.time())

            # Small water drop at correct self.port to indicate animal where the reward is obtained
            # (useful for the beginning of the training)
            if self.extraRewardWCue:
                self._open_water_port(listofValves[PhysRewardPort - 1],
                                      timeLengthWTone * self.compensationFactor[PhysRewardPort - 1],
                                      "Water drop port ")

            # Test if animal licks self.port 2 until it detects self.port 1 lick and jump to the next trial
            listofPortsStates[PhysRewardPort - 1] = False
            time.sleep(TimeRead)
            while (time.time() < t_end) and not listofPortsStates[PhysRewardPort - 1] and (
                    time.time() < (cueTime + self.TimeToReachReward)):
                if (previosrecord + 0.01) < time.time():
                    previosrecord = time.time()
                    mouseTrajectory['traj'].append((Xmean.value, Ymean.value, time.time()))

                # Detection of lick on self.port 2 while animal needs to lick on self.port 1 to get the reward
                for i in range(0, (len(listofPorts))):
                    listofPortsStates[i] = False
                    listofPortsStates[i] = listofPorts[i].read()  # should be "i" here
                    if (not (i == (PhysRewardPort - 1))) and (listofPortsStates[i] is True) and (
                            (timeFirstEntry + timeSampling) < time.time()):  # should be "i" here
                        timeFirstEntry = time.time()
                        datatimes['timeLickIncorrectPort'].append(time.time())
                        datatimes['IncorrectPortLicked'].append(i + 1)
                        print('incorrect port =', i + 1, time.time())
                listofPortsStates[PhysRewardPort - 1] = listofPorts[PhysRewardPort - 1].read()
                time.sleep(TimeRead)
                if (listofPortsStates[PhysRewardPort - 1] is True) and ((timeFirstEntry2 + timeSampling) < time.time()):
                    timeFirstEntry2 = time.time()
                    datatimes['timeLickCorrectPort'].append(time.time())
                    datatimes['timeDelayCorrectLick'].append(time.time() - cueTime)
                    mouseTrajectory['trigerTime'].append(time.time())
                    print('correct port =', PhysRewardPort, time.time())

                    if self.mode == 'Train':
                        period = NonRewardPeriodStart
                    elif self.mode == 'Recall':
                        period = expLength - FixMinRewTime

                    if (t_init + 60 * (period * RewNonRewTimeRatio + aclimTimeLength)) < time.time():
                        if not Water:
                            if (t_init + 60.0 * ((period * RewNonRewTimeRatio) + aclimTimeLength)) < time.time():
                                # Second variable equal to 1 if time is more than RewNonRewTimeRatio of the experiment
                                datatimes['waterStart'].append((time.time(), 1))
                            else:
                                # Second variable equal to 2 if trial number is larger than RewNonRewMinTrials
                                datatimes['waterStart'].append((time.time(), 2))
                                if self.mode == 'train':
                                    # Second variable equal to 1 if time is more than RewNonRewTimeRatio of the
                                    # experiment
                                    datatimes['TimeNoMoreReward'].append((time.time(), 1))
                            Water = True
                            FirstTrialWithWater = len(datatimes['cueTime'])
                            print(colored('[INFO] WATER IS DELIVERED FROM NOW ON', 'blue'))
                        if Water:
                            if (t_init + 60 * (expLength + aclimTimeLength - TimeBefEndNoRew)) < time.time():
                                # Second variable equal to 2 if trial number is larger than RewNonRewMinTrials
                                datatimes['TimeNoMoreReward'].append((time.time(), 0))
                                Water = False
                                print(colored('[INFO] END OF REWARDED PERIOD', 'red'))
                        # dataPlotCorrectPort=[]
                        if Water:
                            self._open_water_port(listofValves[PhysRewardPort - 1],
                                                  timeLengthWReward * self.compensationFactor[PhysRewardPort - 1],
                                                  "Water is been delivered")
                            p.terminate()
                            self.board.digital[pinLight].write(0)
                    if not FirstCorrect:
                        datatimes['timeLickFirstCorrect'].append(time.time())
                        datatimes['timeDelayFirstCorrectLick'].append(time.time() - cueTime)
                        FirstCorrect = True

                # Reset Port state to exit loop
                if time.time() < (t_init + 60 * (NonRewardPeriodStart * RewNonRewTimeRatio + aclimTimeLength)):
                    listofPortsStates[PhysRewardPort - 1] = False

            self.board.digital[pinLight].write(0)
            # Save the first lick on the correct self.port for all trials

            # Generates a Random trigger zone after the trial is finished
            if RandomTriggZone < 1:
                centerDiskX = center[0]
                centerDiskY = center[1]
            else:
                randRadius = (radius - TriggeringZoneRadius) * np.sqrt(random.random())
                # Subtracting the 10% of the total circumference to avoid trigger zones close to the water port
                randAngle = random.random() * 2 * np.pi * (1 - ProhAngleProp)
                finalAngle = ProhAngleProp * np.pi + self.prtLocAngle + randAngle
                centerDiskX = center[0] - np.cos(finalAngle) * randRadius
                centerDiskY = center[1] + np.sin(finalAngle) * randRadius
            XTA.value = int(round(centerDiskX))
            YTA.value = int(round(centerDiskY))
            mouseTrajectory['trigerZone'].append((centerDiskX, centerDiskY))
            print("[INFO] Generating a new random triggering zone")
            TimeToTriggANewTrial = min(self.TimeToReachReward, 0)

            # Re-setting the animal in/out state variable
            In = False
            FirstCorrect = False

            # Plotting partial results

            # Total number of trials:
            dataPlotTrials.append(len(datatimes['cueTime']))
            # Number of trials where lick was outside the licking time window:
            dataPlotErrorPort.append(len(datatimes['timeErrorPort']))
            # Number of trials where one lick was achieved during the tone (only the first):
            dataPlotCorrectPort.append(len(datatimes['timeLickFirstCorrect']))
            # Number of licks were achieved during the tone:
            dataPlotAllCorrectPort.append(len(datatimes['timeLickCorrectPort']))
            # Number of lickes in the incorrect self.port:
            dataPlotIncorrectPort.append(len(datatimes['timeLickIncorrectPort']))
            # Number of trials where lick was outside the licking time window:
            dataPlotErrorCorrectPort.append(len(datatimes['timeErrorCorrectPort']))

            if self.mode == 'Train':
                cote = 1
            elif self.mode == 'Recall':
                cote = 0
            if len(datatimes['timeErrorCorrectPort']) > cote:
                XX = [x for x in datatimes['timeErrorCorrectPort'] if
                      datatimes['cueTime'][-1] <= x < (datatimes['cueTime'][-1] + self.TimeToReachReward)]
                if 0 < len(XX):
                    Aux8.extend([PhysRewardPort] * len(XX))

            # Accumulative sum of trials
            Aux0 = np.array(dataPlotTrials, dtype='float')
            # Accumulative sum of licks outsie the licking time window
            Aux00 = np.array(dataPlotErrorPort, dtype='float')
            # Accumulative sum of correct trials
            Aux1 = np.array(dataPlotCorrectPort, dtype='float')

            if self.mode == 'Train':
                if len(Aux0) < (self.MinNumTrials + 1):
                    AccumPerf = ([0] * self.MinNumTrials)
                else:
                    if datatimes['cueTime'][-1] < datatimes['timeLickCorrectPort'][-1]:
                        AccumPerf.append(1.0)
                    else:
                        AccumPerf.append(0.0)

            # Accumulative sum of licks in correct self.port outside the tone (time-window):
            Aux6 = np.array(dataPlotErrorCorrectPort, dtype='float')
            # Reaction time:
            Aux5 = datatimes['timeDelayFirstCorrectLick'][0:(len(datatimes['timeDelayFirstCorrectLick']))]
            # Index of the incorrect self.port licked:
            Aux88 = datatimes['IncorrectPortLicked'][0:(len(datatimes['IncorrectPortLicked']))]
            # Index of the incorrect self.port licked:
            Aux99 = datatimes['ErrorPortLicked'][0:(len(datatimes['ErrorPortLicked']))]

            if (0 < len(datatimes['timeLickCorrectPort'])) and (
                    datatimes['cueTime'][-1] < datatimes['timeLickCorrectPort'][-1]):
                # Number of incorrect licks during correct trials:
                Aux7 = datatimes['IncorrectPortLicked'][0:(len(datatimes['IncorrectPortLicked']))]
                Aux77 = np.abs(np.diff(Aux7))
                Aux777 = np.where(Aux77 > 0)
                # Number of incorrect ports, before correct self.port
                Aux7777 = np.size(Aux777)
                AuxAccum.append(Aux7777)
                kkk = kkk + 1

            XYcoord = np.array(TrajLastTrial, dtype='float')
            DistribRate, xedges, yedges = np.histogram2d(XYcoord[:, 0], XYcoord[:, 1], bins=10)
            FirRatDens = np.divide(DistribRate, (np.nansum(np.nansum(DistribRate))))
            m_1 = np.append(m_1, np.nansum(np.nansum(FirRatDens * FirRatDens)))

            if (1 < len(datatimes['cueTime'])) and (np.mod(len(datatimes['cueTime']), freqPlot) == 0):
                kk = kk + 1
                if self.mode == 'train' and kk == 1:
                    Aux4 = 1. * dataPlotIncorrectPort[0]
                    Aux04 = 1. * dataPlotErrorPort[0]
                    if Aux4 == 0:
                        Aux4 = 1
                    if Aux04 == 0:
                        Aux04 = 1
                if 1 < kk:
                    x = np.ones(len(Aux0))
                    x2 = np.ones(len(Aux5))
                    yCTT = (1. / 8.) * x
                    yRT = 6 * x2

                    ax4 = fig.add_subplot(241)
                    ax4.cla()
                    ax4.set_title('Inverse of coverage', fontsize=10)
                    ax4.set_ylim([0, 1])
                    for tick in ax4.xaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    for tick in ax4.yaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    ax4.plot((np.arange(len(m_1)) + 1), m_1, 'go-')
                    ax3 = fig.add_subplot(242)
                    ax3.cla()
                    for tick in ax3.xaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    for tick in ax3.yaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    ax3.set_title('Reaction time', fontsize=10)
                    ax3.plot((np.arange(len(Aux5)) + 1), Aux5, 'bo-', (np.arange(len(Aux5))), yRT, 'b--')
                    ax3.text(2, 1, 'Mean React. Time = {:.2f}'.format(np.mean(Aux5)), fontsize=8)

                    ax5 = fig.add_subplot(243)
                    if 1 < kkk:
                        ax5.cla()
                        for tick in ax5.xaxis.get_major_ticks():
                            tick.label.set_fontsize(8)
                        for tick in ax5.yaxis.get_major_ticks():
                            tick.label.set_fontsize(8)
                        ax5.set_title('# Incorrect ports|Correct trials', fontsize=10)
                        ax5.plot((np.arange(len(AuxAccum)) + 1), np.divide(AuxAccum, Aux0[-1]), 'mo-')
                        ax5.text(3.5, 0.05, '#Incorr|Corr. = {:.2f}'.format(np.mean(np.divide(AuxAccum, Aux0[-1]))),
                                 fontsize=8)

                    ax1 = fig.add_subplot(245)
                    ax1.cla()
                    for tick in ax1.xaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    for tick in ax1.yaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    ax1.set_ylim([0, 1.1])
                    ax1.set_title('Accum Perform. (CT/TT)', fontsize=10)
                    ax1.plot((np.arange(len(Aux1)) + 1), Aux1 / Aux0, 'ko-', (np.arange(len(yCTT)) + 1), yCTT, 'k--')
                    ax1.text(3.5, 0.2, 'Mean Perform. = {:.2f}'.format(np.mean(Aux1 / Aux0)), fontsize=8)
                    if self.mode == 'train' and len(Aux0) > 21:
                        ax1.text(3.5, 0.5, 'Perform. last Trials = {:.2f}'.format(
                            np.mean((Aux1[-20:] - Aux1[-21]) / (Aux0[-20:] - Aux0[-21]))), fontsize=8)

                    ax2 = fig.add_subplot(246)
                    ax2.cla()
                    for tick in ax2.xaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    for tick in ax2.yaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    if self.mode == 'Train':
                        ax2.axis([0, kk + 1, 0, 1.1 * max(max(Aux00), max(Aux6))])
                    elif self.mode == 'Recall':
                        ax2.axis([0, (len(Aux00) + 1), 0, 3.1 * (max(max(Aux00 / 7), max(Aux6)) + 1.0)])
                    ax2.set_title('Accum: #Errors, #Persistant', fontsize=10)
                    if self.mode == 'Train':
                        ax2.plot((np.arange(len(Aux00)) + 1), Aux00 / 7, 'ro--', (np.arange(len(Aux6)) + 1), Aux6,
                                 'ko:')
                    elif self.mode == 'Recall':
                        ax2.plot([FirstTrialWithAvailWater, FirstTrialWithAvailWater],
                                 [0, max(max(Aux00 / 7), max(Aux6))], ':b', lw=0.5)
                        ax2.plot([FirstTrialWithWater, FirstTrialWithWater], [0, max(max(Aux00 / 7), max(Aux6))], ':g',
                                 lw=0.5)
                        ax2.text(4, 200, '#Trials w/H2O = {:.2f}'.format(ww), fontsize=8)

                    ax22 = fig.add_subplot(247)
                    ax22.cla()
                    for tick in ax22.xaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    for tick in ax22.yaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    ax22.set_xlim([-4.5, 5.5])
                    if self.mode == 'Train':
                        ax22.set_title('Histogram of Learning', fontsize=10)
                        if 0 < len(Aux5):
                            AuxMem = np.concatenate((Aux8, Aux88, np.ones(len(Aux5)) * PhysRewardPort))
                        else:
                            AuxMem = np.concatenate((Aux8, Aux88))
                    elif self.mode == 'Recall':
                        ax22.set_title('Histogram of Recall before water', fontsize=10)
                        if FirstTrialWithWater < 2:
                            if 0 < len(Aux5):
                                AuxMem = np.concatenate((Aux8, Aux88, Aux99, np.ones(len(Aux5)) * PhysRewardPort))
                            else:
                                AuxMem = np.concatenate((Aux8, Aux88, Aux99))
                        else:
                            XX00 = [yy for yy in datatimes['timeLickFirstCorrect'] if
                                    yy < (np.array(datatimes['waterStart'])[0, 0])]
                            XX0 = [y for y in datatimes['timeLickCorrectPort'] if
                                   y < (np.array(datatimes['waterStart'])[0, 0])]
                            AuxCorr = np.array(np.ones(len(XX0)) * PhysRewardPort, dtype='float')
                            XX1 = [z for z in datatimes['timeLickIncorrectPort'] if
                                   z < (np.array(datatimes['waterStart'])[0, 0])]
                            AuxInc = np.array(datatimes['IncorrectPortLicked'])[:(len(XX1))]
                            XX2 = [w for w in datatimes['timeErrorCorrectPort'] if
                                   w < (np.array(datatimes['waterStart'])[0, 0])]
                            AuxErrCorr = np.array(np.ones(len(XX2)) * PhysRewardPort, dtype='float')
                            XX3 = [zz for zz in datatimes['timeErrorPort'] if
                                   zz < (np.array(datatimes['waterStart'])[0, 0])]
                            AuxErr = np.array(datatimes['ErrorPortLicked'])[:(len(XX3))]
                            AuxMem = np.concatenate((AuxCorr, AuxInc, AuxErrCorr, AuxErr))
                    AuxMemMem = np.array(np.mod(AuxMem - PhysRewardPort, 8))
                    AUX7 = np.where(AuxMemMem == 7)
                    AuxMemMem[AUX7[0][:]] = -1
                    AUX6 = np.where(AuxMemMem == 6)
                    AuxMemMem[AUX6[0][:]] = -2
                    AUX5 = np.where(AuxMemMem == 5)
                    AuxMemMem[AUX5[0][:]] = -3
                    if 0 < len(AuxMemMem):
                        ax22.hist(AuxMemMem, bins=(-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5), color='blue')

                    ax33 = fig.add_subplot(244)
                    ax33.cla()
                    for tick in ax33.xaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    for tick in ax33.yaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    ax33.set_xlim([-4.5, 5.5])
                    if self.mode == 'Train':
                        ax33.set_title('Histogram of 24 Hrs Recall', fontsize=10)
                        Aux = np.array(np.mod(Aux11 - PhysRewardPortYesterday, 8))
                    elif self.mode == 'Recall':
                        ax33.set_title('Histogram of Recall before first trial', fontsize=10)
                        Aux = np.array(np.mod(Aux11 - PhysRewardPort, 8))
                    AUX7 = np.where(Aux == 7)
                    Aux[AUX7[0][:]] = -1
                    AUX6 = np.where(Aux == 6)
                    Aux[AUX6[0][:]] = -2
                    AUX5 = np.where(Aux == 5)
                    Aux[AUX5[0][:]] = -3
                    ax33.hist(Aux, bins=(-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5), color='green')

                    ax44 = fig.add_subplot(248)
                    ax44.cla()
                    for tick in ax44.xaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    for tick in ax44.yaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    ax44.set_xlim([-4.5, 5.5])
                    ax44.set_title('Histogram licks On+Off Tone', fontsize=10)
                    AUX = np.array(np.mod(np.concatenate((Aux8, Aux88, Aux99)) - PhysRewardPort, 8))
                    AUX7 = np.where(AUX == 7)
                    AUX[AUX7[0][:]] = -1
                    AUX6 = np.where(AUX == 6)
                    AUX[AUX6[0][:]] = -2
                    AUX5 = np.where(AUX == 5)
                    AUX[AUX5[0][:]] = -3
                    ax44.hist(AUX, bins=(-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5), color='grey')

                    plt.show()
                    plt.pause(0.0001)

            # Emergency program behavior for contingency of low performance after 20 trials
            if self.mode == 'Train':
                if ((EmergencyTime * 60 + t_init) < time.time()) and ((Aux1[-1] / Aux0[-1]) < 0.25):
                    self.TimeToReachReward = ExtendedTime
                    datatimes['trialsWithHelp'].append(
                        (kk * freqPlot, ExtendedTime))  # Trial index self.TimeToReachReward was extended to extra time
                    print("[INFO] RUNNING UNDER THE EMERGENCY PROTOCOL")
                    if (self.TimeToReachReward == ExtendedTime) and (0.5 < (Aux1[-1] / Aux0[-1])):
                        self.TimeToReachReward = OriginalTime
                else:
                    self.TimeToReachReward = OriginalTime

        # video recording release
        running.value = 0

        # Release the self.board
        self.board.digital[pinLight].write(0)
        self.board.exit()

        # Saving data time stems
        data = {}
        timeEndExp = time.time()
        data['timestamp'] = time.time()
        data['datatimes'] = datatimes
        data['mouseTrajectory'] = mouseTrajectory

        # Building the time stem of the end of the experiment as a string
        valueTime = datetime.datetime.fromtimestamp(timeEndExp)
        path = 'data_output'

        if self.mode == 'Train':
            name = '_%Y-%m-%d_%H-%M-%S_trn'
        elif self.mode == 'Recall':
            name = '_%Y-%m-%d_%H-%M-%S_rec'

        animalName = self.AnimalNames[self.AnimalNumber - 1]
        f_myfile = open(path + animalName + valueTime.strftime(name + '.dat'), 'wb')

        # Bump all the data into a dictionary
        pickle.dump(data, f_myfile)
        f_myfile.close()

        # Save figure
        fig.savefig(path + animalName + valueTime.strftime(name))
        scipy.io.savemat(path + animalName + valueTime.strftime(name + 'mat'), mdict={'data': data})
        p1.terminate
        p.terminate
