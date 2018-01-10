import os
import cv2
import numpy
import pydot
import threading
import pyautogui
import pynput.mouse
import pynput.keyboard
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .Preprocessing import ImagePreprocessing
from .LeNet import LeNet
from .ErrorCodes import ErrorCodes

class CaptchaSolver(ErrorCodes):     

    # OpenCV mouse click event
    def Click(self, event, x, y, flags, param):
        # Check mouse button
        if(event == cv2.EVENT_LBUTTONDOWN):
            self.refPt = [(x, y)]
        elif(event == cv2.EVENT_LBUTTONUP):
            self.refPt.append((x, y))

            self.Selection = (self.refPt[0][0], self.refPt[0][1], self.refPt[1][0], self.refPt[1][1])
            self.SelectionAvailable = True

    # 
    def On_Press(self, Key):
        # Take a new screenshot
        if(Key.char == "q"):
            self.TakeScreenshot()

    # Thread for keyboard listener
    # Use own threading method because Listener.start() doesn´t work in a class
    def KeyboardThread(self):
        with pynput.keyboard.Listener(
                on_press = self.On_Press,
                ) as listener:listener.join()

    def __init__(self, Width, Height, Epochs, Depth = 1, Batchsize = 32, Bordersize = 8):
        # Print some project informations
        print("+-------------------------------------------------------------------------------")
        print("|            Captcha-Breaker @ Daniel Kampert                                  |")      
        print("| This is a private project for my 'KI & Softcomputing' lecture at HSD germany.|")
        print("| For more informations visit www.github.com/Kampi or write me an E-Mail to    |")
        print("| 'DanielKampert@kampis-elektroecke.de'                                        |")
        print("+------------------------------------------------------------------------------+")

        self.TrainingData = []
        self.TrainingLabel = []
        self.TrainX = []
        self.TrainY = []
        self.TestX = []
        self.TestY = []
        self.Predictions = []
        self.refPt = []
        self.CorrectCounter = 0
        self.LetterCounter = 0
        self.Binarizer = 0
        self.History = 0
        self.Labelcount = 0
        self.Selection = (0, 0)
        self.MouseCoordinatesPressed = (0, 0)
        self.MouseCoordinatesReleased = (0, 0)
        self.LeNet = LeNet()
        self.Mouse = pynput.mouse.Controller()
        self.ImageProcessor = ImagePreprocessing(Width, Height)
        self.ThreadKeyboard = threading.Thread(target = self.KeyboardThread)
        self.Width = Width
        self.Height = Height
        self.Depth = Depth
        self.Epochs = Epochs
        self.Batchsize = Batchsize
        self.Bordersize = Bordersize
        self.SelectionAvailable = False

    def EnableLiveMode(self):
        # Start the mouse thread
        self.ThreadKeyboard.start()

    def DisableLiveMode(self):
        # Stop the mouse thread
        self.ThreadKeyboard.stop()

    def PrintModel(self, OutputPath, ModelName = "Model.png"):
        # Check if the path exist
        if(os.path.exists(OutputPath)):
            # Check if the model is available
            if(self.LeNet != 0):
                print("[INFO] Save visualization to " + OutputPath + "...")
                plot_model(self.LeNet, to_file = OutputPath + "\\" + ModelName, show_shapes = True)
            else:
                print("[ERROR] No model available!")
                return ErrorCodes.NO_MODEL
            return ErrorCodes.UNKNOWN_PATH

        return ErrorCodes.NO_ERROR

    def Report(self, OutputPath = "", ReportName = "Report.txt", PlotName = "Plot.png"):
        # Check if prediction is available
        if(len(self.Predictions) != 0):
            print("[STATUS] Create report for training...")
            Report = classification_report(self.TestY.argmax(axis = 1), self.Predictions.argmax(axis = 1), target_names = self.Binarizer.classes_)
        else:
            return ErrorCodes.NO_DATA

        if(len(OutputPath) != 0):
            print("[INFO] Save report to disk...")
            File = open(OutputPath + "\\" + ReportName, "w")
            File.write(Report)
            File.close()
        else:
            print(Report)

        # Cancel if no training history is available
        if(self.History != 0):
            # Create a new plot
            plt.style.use("ggplot")
            plt.figure(num = "Training")
            plt.plot(numpy.arange(0, self.Epochs), self.History.history["loss"], label = "Train (Loss)")
            plt.plot(numpy.arange(0, self.Epochs), self.History.history["val_loss"], label = "Test (Loss)")
            plt.plot(numpy.arange(0, self.Epochs), self.History.history["acc"], label = "Accuracy")
            plt.plot(numpy.arange(0, self.Epochs), self.History.history["val_acc"], label = "Test (Accuracy)")
            plt.title("Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # Save plot local to disk if the output path isn´t zero
            if(len(OutputPath) != 0):
                plt.savefig(OutputPath + "\\" + PlotName)
            else:
                plt.show()
        else:
            print("[ERROR] No history available!")
            return ErrorCodes.NO_MODEL
        
        return ErrorCodes.NO_ERROR

    def LoadModel(self, InputPath, ModelFileName = "Model", LabelFileName = "Label"):
        ModelPath = InputPath + "\\" + ModelFileName + ".hdf5"
        LabelPath = InputPath + "\\" + LabelFileName

        # Check if the path exist
        if(os.path.exists(InputPath)):
            if(os.path.exists(InputPath)):
                print("[INFO] Load model from disk...")
                self.LeNet = load_model(ModelPath)
                print("[INFO] Load label from disk...")

                # Read label from disk
                LabelFile = open(LabelPath, "r")
                List = []
                List = LabelFile.read().replace("[", "").replace("]", "").replace("'", "").split(" ")
                LabelFile.close()

                # Pass the label to the binariuer
                self.Binarizer = LabelBinarizer().fit(List)
            else:
                print("[ERROR] Unknown input path!")
                return ErrorCodes.UNKNOWN_PATH
        else:
            print("[ERROR] Unknown input path!")
            return ErrorCodes.UNKNOWN_PATH

        # Reset the counter if a new model was loaded
        self.ResetCounter()

        return ErrorCodes.NO_ERROR

    def SaveModel(self, OutputPath, ModelFileName = "Model", LabelFileName = "Label"):
        ModelPath = OutputPath + "\\" + ModelFileName + ".hdf5"
        LabelPath = OutputPath + "\\" + LabelFileName

        # Check if the model and the path exist
        if((self.LeNet != 0) and (len(self.Binarizer.classes_) != 0)):
            if(os.path.exists(OutputPath)):
                print("[INFO] Save model as " + ModelPath + " to disk...")

                # Save the model at the given path
                self.LeNet.save(ModelPath)

                print("[INFO] Save label as " + LabelPath + " to disk...")

                # Open the file at the given path
                LabelFile = open(LabelPath, "w")

                # Save the classes (label) into the file
                LabelFile.write(str(self.Binarizer.classes_))

                # Close the file
                LabelFile.close()
            else:
                print("[ERROR] Unknown output path for model!")
                return ErrorCodes.UNKNOWN_PATH
        else:
            print("[ERROR] No model available!")
            return ErrorCodes.NO_MODEL

        return ErrorCodes.NO_ERROR

    def LoadTrainingData(self, InputPath, OutputPath, SplitRatio = 0.25, RandomState = 0):
        # Preprocess the input images
        self.ImageProcessor.PreprocessAndSaveImages(InputPath, OutputPath, self.Bordersize)

        # Check if path exist
        if(not(os.path.exists(InputPath))):
            # Check if path is a directory
            if(not(os.path.isdir(InputPath))):
                print("[ERROR] Unknown path to trainingdata!")
                return ErrorCodes.UNKNOWN_PATH

        # Load the files from path
        Folder = os.listdir(OutputPath)
        
        print("[INFO] Found {} folder".format(len(Folder)))

        for [FolderIndex, FolderName] in enumerate(Folder):
            # Some status message
            print(" [STATUS] Open folder {}/{}".format(FolderIndex + 1, len(Folder)))

            # Get all the files from the chosen folder
            Files_Folder = os.listdir(OutputPath + "\\" + FolderName)

            for [FileIndex, FileName] in enumerate(Files_Folder):
                # Another status message
                print("     [STATUS] Load file {}/{}".format(FileIndex + 1, len(Files_Folder)))

                # Load a image
                Image = cv2.imread(OutputPath + "\\" + FolderName + "\\" + FileName, 0)

                # Convert it to an array for the neural network
                Image = img_to_array(Image)
                self.TrainingData.append(Image)
            
                # Store the image labels
                self.TrainingLabel.append(FolderName)

        # Count the different labels
        self.Labelcount = len(set(self.TrainingLabel))

        # Skale the data
        print("[STATUS] Scale images...")
        self.TrainingData = numpy.array(self.TrainingData, dtype = "float") / 255.0

        # Split the data into training and test data
        print("[STATUS] Split data...")
        (self.TrainX, self.TestX, self.TrainY, self.TestY) = train_test_split(self.TrainingData, self.TrainingLabel, test_size = SplitRatio, random_state = RandomState)

        # Preprocess the labels
        print("[STATUS] Found {} label".format(self.Labelcount))
        print("[STATUS] Convert label...")
        self.Binarizer = LabelBinarizer().fit(self.TrainingLabel)
        self.TrainY = self.Binarizer.transform(self.TrainY)
        self.TestY = self.Binarizer.transform(self.TestY)

        return ErrorCodes.NO_ERROR

    def TrainModel(self):
        # Check if training data are available
        if((not(len(self.TrainX) == 0)) and (not(len(self.TrainY) == 0))):
            print("[STATUS] Build LeNet...")
            self.LeNet = self.LeNet.build(self.Width, self.Height, self.Depth, self.Labelcount)
            self.LeNet.compile(loss = "categorical_crossentropy", optimizer = SGD(lr = 0.01), metrics = ["accuracy"])

            print("[STATUS] Train network...")
            self.History = self.LeNet.fit(self.TrainX, self.TrainY, validation_data = (self.TestX, self.TestY), batch_size = self.Batchsize, epochs = self.Epochs, verbose = 1)

            print("[STATUS] Evaluate network...")
            self.Predictions = self.LeNet.predict(self.TestX, batch_size = self.Batchsize)
        else:
            print("[ERROR] No data loaded!")
            return ErrorCodes.NO_DATA

        return ErrorCodes.NO_ERROR

    def ResetCounter(self):
        self.LetterCounter = 0
        self.CorrectCounter = 0

    def GetCounter(self):
        return [self.LetterCounter, self.CorrectCounter]

    def Predict(self, Image, DrawingColor = (255, 0, 0), Debug = False):
        PredictetCaptcha = 0
        Predictions = []
        ColorImage = 0

        if(self.LeNet != 0):
            # Error handling
            try:
                # Load image and find all contours
                # Create a border with width around the image to prevent that the captcha touch the border of the image
                Return = self.ImageProcessor.PreprocessImage(Image, self.Bordersize)
                if(type(Return) is int):
                    return Return
                else:
                    [ColorImage, BinaryImage, Contours] = Return

                if(Debug == True):
                    print("[ACTION] Press <y> if predicted letter is correct, or <n> if it´s not.")
                    # Increase the lettercounter by the number of contours
                    self.LetterCounter += len(Contours)

                # Add a 15px white Border under the image for the label
                ColorImage = cv2.copyMakeBorder(ColorImage, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value = (255, 255, 255))

                # Loop over each contour
                for [ContourNr, Contour] in enumerate(Contours):
                    # Extract each contour
                    (x, y, w, h) = cv2.boundingRect(Contour)
                    ROI = BinaryImage[y:y + h, x:x + w]
                    ROI = cv2.resize(ROI, (self.Width, self.Height))

                    # Convert the image to an array and normalize it
                    Data = numpy.expand_dims(img_to_array(ROI), axis = 0) / 255.0

                    # Create a new prediction
                    Prediction = self.LeNet.predict(Data, self.Batchsize)
                    PredictionClasses = self.Binarizer.classes_[Prediction.argmax(axis = 1)]

                    # Draw a rectangle around each letter
                    cv2.rectangle(ColorImage, (x - self.Bordersize - 4, y - self.Bordersize - 4), (x + w - self.Bordersize + 4, y + h - self.Bordersize + 4), DrawingColor, 2)
                    
                    # Add the label as text to the image
                    cv2.putText(ColorImage, str(PredictionClasses[0].upper()), (x, ColorImage.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DrawingColor, 2)

                    # Show preview and activate check for prediction
                    if(Debug == True):
                        # Print prediction results
                        print("[INFO] Prediction: {}".format(Prediction.argmax(axis = 1)))
                        print("[INFO] Label: {}".format(PredictionClasses))

                        cv2.imshow("Preview", ROI)

                        # Key handling
                        Key = cv2.waitKey(0)
                        while((not(chr(Key) == 'y')) and (not(chr(Key) == 'n')) and (not(Key == 27))):
                            print("   [ERROR] Unknown key!")
                            Key = cv2.waitKey(0)

                        if(chr(Key) == 'y'):
                            print("   [INFO] Correct letter. Increase counter")
                            self.CorrectCounter += 1.0

                        elif(chr(Key) == 'n'):
                            print("   [INFO] Wrong letter. Skip.")
                        # Cancel classification on ESC-Key
                        elif(Key == 27):
                            print("   [INFO] Cancel captcha classification...")
                            cv2.destroyWindow("Preview")

                            return ColorImage

                        cv2.destroyWindow("Preview")

            except Exception as e:
                print("[ERROR] {}".format(e))
        else:
            return ColorImage

        # Print prediction results in debug mode
        if(Debug == True):
            print("[INFO] Predict {} of {} letters correct.".format(int(self.CorrectCounter), self.LetterCounter))

            if(self.LetterCounter != 0):
                print("[INFO] Prediction rate {}%".format(round(self.CorrectCounter / float(self.LetterCounter) * 100.0), 3))
            else:
                print("[INFO] No letter found!")

        return ColorImage

    def TakeScreenshot(self):
        # Take a screenshot and convert it into a opencv image
        Screenshot = pyautogui.screenshot()
        Screenshot = cv2.cvtColor(numpy.array(Screenshot), cv2.COLOR_RGB2BGR)

        # Create a new window
        cv2.namedWindow("Preview")
        cv2.setMouseCallback("Preview", self.Click)

        while(self.SelectionAvailable == False):
            cv2.imshow("Preview", Screenshot)
            cv2.waitKey(1)

        cv2.destroyWindow("Preview")

        ROI = Screenshot[self.Selection[1]:self.Selection[3], self.Selection[0]:self.Selection[2]]
        
        # Use ROI for prediction
        self.Predict(ROI)