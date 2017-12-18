import os
import cv2
import numpy
import pydot
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from .Preprocessing import ImagePreprocessing
from .LeNet import LeNet
from .ErrorCodes import ErrorCodes

'''
- In der Predict-Methode muss noch geprüft werden ob ein Model vorhanden ist
'''

class CaptchaSolver(ErrorCodes):     

    def __init__(self, Width, Height, Epochs, Depth = 1, Batchsize = 32):
        # Print some project informations
        print("+-------------------------------------------------------------------------------+")
        print("|            Captcha-Breaker @ Daniel Kampert                                   |")      
        print("| This is a private project for my 'KI & Softcomputing' lecture at HSD germany. |")
        print("| For more informations visit www.github.com/Kampi or write me an E-Mail to     |")
        print("| 'DanielKampert@kampis-elektroecke.de'                                         |")
        print("+-------------------------------------------------------------------------------+")

        self.TrainingData = []
        self.TrainingLabel = []
        self.TrainX = []
        self.TrainY = []
        self.TestX = []
        self.TestY = []
        self.CorrectCounter = 0
        self.LetterCounter = 0
        self.Predictions = []
        self.LeNet = LeNet()
        self.Binarizer = 0
        self.History = 0
        self.Labelcount = 0
        self.Width = Width
        self.Height = Height
        self.Depth = Depth
        self.Epochs = Epochs
        self.Batchsize = Batchsize
        self.ImageProcessor = ImagePreprocessing(Width, Height)

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

    def LoadModel(self, InputPath, ModelFileName = "Model.hdf5", LabelFileName = "Label"):
        ModelPath = InputPath + "\\" + ModelFileName
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

    def SaveModel(self, OutputPath, ModelFileName = "Model.hdf5", LabelFileName = "Label"):
        ModelPath = OutputPath + "\\" + ModelFileName
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
                print("[ERROR] Unknown input path!")
                return ErrorCodes.UNKNOWN_PATH
        else:
            print("[ERROR] No model available!")
            return ErrorCodes.NO_MODEL

        return ErrorCodes.NO_ERROR

    def LoadTrainingData(self, InputPath, OutputPath, SplitRatio = 0.25, RandomState = 0):
        # Preprocess the input images
        self.ImageProcessor.PreprocessAndSaveImages(InputPath, OutputPath)

        # Check if path exist
        if(not(os.path.exists(InputPath))):
            # Check if path is a directory
            if(not(os.path.isdir(InputPath))):
                print("[ERROR] Unknown input path!")
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
                print("     [STATUS] Load file {}/{}".format(FileIndex + 1, len(FileName)))

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

    def Predict(self, InputImagePath):
        PredictetCaptcha = 0
        Predictions = []

        if(self.LeNet != 0):
            try:
                # Load image and find all contours
                Return = self.ImageProcessor.PreprocessImage(InputImagePath)
                if(type(Return) is int):
                    return Return
                else:
                    [BinaryImage, Contours] = Return

                print("[ACTION] Press 'y' if predicted letter is correct, or 'n' if it´s not.")

                # Increase the lettercounter by the number of contours to
                self.LetterCounter += len(Contours)

                # Loop over each contour
                for Contour in Contours:
                    # Extract each contour
                    (x, y, w, h) = cv2.boundingRect(Contour)
                    ROI = BinaryImage[y:y + h, x:x + w]
                    ROI = cv2.resize(ROI, (self.Width, self.Height))
            
                    # Convert the image to an array and normalize it
                    Data = numpy.expand_dims(img_to_array(ROI), axis = 0) / 255.0

                    # Create a new prediction
                    Prediction = self.LeNet.predict(Data, self.Batchsize)

                    # Print prediction results
                    print("[INFO] Prediction: {}".format(Prediction.argmax(axis = 1)))
                    print("[INFO] Label: {}".format(self.Binarizer.classes_[Prediction.argmax(axis = 1)]))
            
                    cv2.imshow("Preview", ROI)

                    # Key handling
                    Key = cv2.waitKey(0)
                    if(chr(Key) == 'y'):
                        print("[INFO] Correct letter. Increase counter")
                        self.CorrectCounter += 1.0
                    elif(chr(Key) == 'n'):
                        print("[INFO] Wrong letter. Skip.")
                    # Cancel classification on ESC-Key
                    elif(Key == 27):
                        print("[INFO] Cancel captcha classification...")
                        cv2.destroyWindow("Preview")

                        return ErrorCodes.NO_ERROR

                    cv2.destroyWindow("Preview")
            except Exception as e:
                print("[ERROR] {}".format(e))
        else:
            return ErrorCodes.NO_MODEL

        # Print prediction results
        print("[INFO] Predict {} of {} letters correct.".format(self.CorrectCounter, self.LetterCounter))
        print("[INFO] Prediction rate {}%".format(round(self.CorrectCounter / float(self.LetterCounter) * 100.0), 3))

        return ErrorCodes.NO_ERROR
            