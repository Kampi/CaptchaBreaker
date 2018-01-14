import os
import cv2
import numpy
from string import ascii_lowercase

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
                return -1
        else:
            return -1

    def ProcessImages(self, InputPath, OutputPath = ""):
        LetterList = []

         # Check if path exist
        if(not(os.path.exists(InputPath))):
            # Check if path is a directory
            if(not(os.path.isdir(InputPath))):
                print("[ERROR] Unknown input path!")
                return -1

        # Load the images
        Images = self.LoadImages(InputPath)

        # Check if list is empty
        if(len(Images) == 0):
            print("[ERROR] No images found. Abort preprocessing!")
            return -1
        else:
            print("[INFO] {} Images found...".format(len(Images)))

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

        # Preprocess each image
        for(ImageCounter, ImageFromList) in enumerate(Images):
            print("[STATUS] Process image {} - {}/{}".format(ImageFromList, ImageCounter + 1, len(Images)))

            # Try to start the preprocessing
            try:
                # Load each image as grayimage
                GrayImage = cv2.imread(InputPath + "\\" + ImageFromList, 0)
                
                # Create a border with width of 8px around the image to prevent that the captcha touch the border of the image
                GrayImage = cv2.copyMakeBorder(GrayImage, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

                # Create a binary image
                [ret, BinaryImage] = cv2.threshold(GrayImage, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

                # Erode the image
                BinaryImage = cv2.erode(BinaryImage, self.Kernel, iterations = 1)

                # Search the image for contours
                [im2, Contours, Hierarchy] = cv2.findContours(BinaryImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for Contour in Contours:
                    (x, y, w, h) = cv2.boundingRect(Contour)
                    ROI = BinaryImage[y:y + h, x:x + w]
                    ROI = cv2.resize(ROI, (self.ImageWidth, self.ImageHeight))

                    # Store the images on the system, if the length of the path is not zero.
                    # Otherwise store them into a list
                    if(not(len(OutputPath) == 0)):
                        # Preview of the current letter
                        cv2.imshow("Preview", ROI)

                        # Wait for user input
                        Key = cv2.waitKey(0)

                        # Convert user input
                        Key = chr(Key)
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
                        else:
                            print("[INFO] Skip image...")
                    else:
                        LetterList.append(ROI)

                    # Close the preview window
                    cv2.destroyWindow("Preview")

                return LetterList

            except Exception as e:
                print(e)