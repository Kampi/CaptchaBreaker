import os
import cv2
import numpy
from string import ascii_lowercase

from CaptchaSolver.ErrorCodes import ErrorCodes

class ImagePreprocessing:

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

    def GetContours(self, Image, Border):             
        
        if(Border > 0):
            Image = cv2.copyMakeBorder(Image, Border, Border, Border, Border, cv2.BORDER_REPLICATE)

        # Create a binary image
        [ret, BinaryImage] = cv2.threshold(Image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Erode the image
        BinaryImage = cv2.erode(BinaryImage, self.Kernel, iterations = 1)

        # Search the image for contours
        [im2, Contours, Hierarchy] = cv2.findContours(BinaryImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return [BinaryImage, Contours]

    def PreprocessImage(self, InputPath, Border = 8):
        # Check if path exist
        if(not(os.path.exists(InputPath))):
            return ErrorCodes.UNKNOWN_PATH

        # Load image
        Image = cv2.imread(InputPath)

        # Convert the image to grayscale, add a border and search contours
        return [Image] + self.GetContours(cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY), Border)

    def PreprocessAndSaveImages(self, InputPath, OutputPath, Border):
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

        print("[INFO] Starting image classification...")
        print("[INFO] Press the key corrosponding to the shown letter.")
        print("[ACTION] Press <Esc> to chancel the classification.")
        print("[ACTION] Press any other key to skip the letter.")

        # Preprocess each image
        for(ImageCounter, ImageFromList) in enumerate(Images):
            print("[STATUS] Process image {} - {}/{}".format(ImageFromList, ImageCounter + 1, len(Images)))

            # Try to start the preprocessing
            try:
                Image = cv2.imread(InputPath + "\\" + ImageFromList, 0)
                [BinaryImage, Contours] = self.GetContours(Image, Border)

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