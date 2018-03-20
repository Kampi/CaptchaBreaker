import os
import cv2
import numpy
from string import ascii_lowercase

from CaptchaSolver.ErrorCodes import ErrorCodes

class CaptchaImagePreprocessor:
    def __init__(self, ImageWidth, ImageHeight):
        self.ImageWidth = ImageWidth
        self.ImageHeight = ImageHeight
        self.Labels = {}

        # Create the kernel for the erode-function
        self.Kernel =  cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def LoadImages(self, Path):
        # Check if path exist
        if(os.path.exists(Path)):
            # Check if path is a directory
            if(os.path.isdir(Path)):
                # list all files
                return os.listdir(Path)
            else:
                return ErrorCodes.UNKNOWN_PATH
        else:
            return ErrorCodes.UNKNOWN_PATH

    def GetContours(self, Image, Border, ErodeIterations = 1, Debug = False):
        # Return error if image is empty
        if(Image is None):
            return ErrorCodes.NO_IMAGE

        # Show the original image
        if(Debug):
            cv2.imshow("Original image", Image)
            cv2.waitKey(100)

        # Add border to the image
        if(Border > 0):
            Image = cv2.copyMakeBorder(Image, Border, Border, Border, Border, cv2.BORDER_REPLICATE)

        # Create a binary image
        [ret, BinaryImage] = cv2.threshold(Image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # If the image is a white image, invert the image
        if(cv2.mean(BinaryImage)[0] > 128.0):
            cv2.bitwise_not(BinaryImage.copy(), BinaryImage)

        # Display the image after thesholding
        if(Debug):
            cv2.imshow("Threshold", BinaryImage)
            cv2.waitKey(100)

        # Erode the image
        #BinaryImage = cv2.erode(BinaryImage, self.Kernel, iterations = ErodeIterations)
        Morph = cv2.morphologyEx(BinaryImage, cv2.MORPH_OPEN, self.Kernel)
        Morph = cv2.erode(Morph, self.Kernel, iterations = 1)

        # Display the image when debug is active
        if(Debug):
            cv2.imshow("Morphological Transformation", Morph)
            cv2.waitKey(100)
        
        # Search the image for contours
        [im2, Contours, Hierarchy] = cv2.findContours(Morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return [BinaryImage, Contours]

    def PreprocessImage(self, InputImage, Border = 8, Debug = False):
        # Check the image type
        if(type(InputImage) is str):
            # Check if path exist
            if(not(os.path.exists(InputImage))):
                return ErrorCodes.UNKNOWN_PATH

            # Load the image
            Image = cv2.imread(InputImage)
        elif(type(InputImage) is numpy.ndarray):
            Image = InputImage
        else:
            return ErrorCodes.NO_DATA

        # Convert the image to grayscale, add a border and search contours
        return [Image] + self.GetContours(cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY), Border, Debug)

    def PreprocessAndSaveImages(self, InputPath, OutputPath, Border, Debug = False):
        LetterList = []

         # Check if path exist
        if(not(os.path.exists(InputPath))):
            # Check if path is a directory
            if(not(os.path.isdir(InputPath))):
                print("[ERROR] Unknown input path!")
                return ErrorCodes.UNKNOWN_PATH

        # Load the images
        Images = self.LoadImages(InputPath)

        # Check if list is empty
        if(len(Images) == 0):
            print("[ERROR] No images found. Abort preprocessing!")
            return ErrorCodes.UNKNOWN_PATH
        else:
            print("[STATUS] {} Images found...".format(len(Images)))

        # Create subdirectories for labeled images
        if(not(len(OutputPath) == 0)):
            if(not(os.path.exists(OutputPath))):
                os.mkdir(OutputPath)

            for Char in ascii_lowercase:
                if(not(os.path.exists(OutputPath + "\\" + Char))):
                    os.mkdir(OutputPath + "\\" + Char)

            for Number in range(0, 10):
                if(not(os.path.exists(OutputPath + "\\" + str(Number)))):
                    os.mkdir(OutputPath + "\\" + str(Number))
            # Preload label count when directories exist
            else:
                DirectoryList = os.listdir(OutputPath)

                # Loop over each directory
                for Dir in DirectoryList:
                    FileList = []
                    # Get the files and remove file ending
                    for File in os.listdir(OutputPath + "\\" + Dir):
                        Index = File.rfind('.')
                        File = File[:Index]
                        FileList.append(File)

                    # Get the max index which can be used
                    if(len(FileList) > 0):
                        self.Labels.update({Dir:int(max(FileList)) + 1})

        print("[INFO] Starting image classification...")
        print("[INFO] Press the key corrosponding to the shown letter.")
        print("[ACTION] Press <ESC> to chancel the classification.")
        print("[ACTION] Press any other key to skip the letter.")

        # Preprocess each image
        for(ImageCounter, ImageFromList) in enumerate(Images):
            print("[STATUS] Process image {} - {}/{}".format(ImageFromList, ImageCounter + 1, len(Images)))

            # Try to start the preprocessing
            try:
                ImagePath = InputPath + "\\" + ImageFromList
                Image = cv2.imread(ImagePath, 0)
                Return = self.GetContours(Image, Border, Debug)

                # Check if image is not empty
                if(type(Return) is int):
                    print("[ERROR] Can´t read image '{}'...".format(ImagePath))
                    return Return
                else:
                    [BinaryImage, Contours] = Return

                # Loop and process each contour
                for Contour in Contours:
                    (x, y, w, h) = cv2.boundingRect(Contour)
                    ROI = BinaryImage[y:y + h, x:x + w]
                    ROI = cv2.resize(ROI, (self.ImageWidth, self.ImageHeight))

                    # Preview of the current letter
                    cv2.imshow("Preview", ROI)

                    # Wait for user input
                    RawKey = cv2.waitKey(0)

                    # Convert user input
                    Key = chr(RawKey)
                    Key = Key.lower()

                    if(Key.isalpha() or Key.isnumeric()):
                        # Get the image directory
                        DirPath = OutputPath + "\\" + Key

                        # Get the current image number
                        Label = self.Labels.get(Key, 1)
                        FilePath = os.path.sep.join([OutputPath, Key])
                        FilePath = os.path.sep.join([FilePath, "{}.png".format(str(Label).zfill(6))])
                        cv2.imwrite(FilePath, ROI)
                        self.Labels[Key] = Label + 1
                    # Handle ESC-Key
                    elif(RawKey == 27):
                        print("[INFO] Cancel image classification...")
                        cv2.destroyWindow("Preview")
                        return ErrorCodes.CANCEL
                    else:
                        print("[INFO] Skip image...")

                    LetterList.append(ROI)

                    # Close the preview window
                    cv2.destroyWindow("Preview")

            except Exception as e:
                print(e)

        return LetterList