import os
import sys
import cv2
import argparse

from CaptchaSolver.CaptchaSolver import CaptchaSolver

# Check the version of python
if(sys.version_info <= (3, 6, 2)):
    print("Python Version: {}".format(sys.version))
    print("Version not supported!")
    exit()

Solver = CaptchaSolver(28, 28, 30)

# Set commandline arguments
ArgParser = argparse.ArgumentParser()
ArgParser.add_argument("-m", "--mode", choices = ["demo", "live"], help = "Select the application mode", default = "demo", required = True)
ArgParser.add_argument("-w", "--model", help = "Select if the CNN should be learned or loaded from file", nargs = 2, required = True)
ArgParser.add_argument("-i", "--input", help = "Input files for classification", nargs = '+', required = True)
ArgParser.add_argument("-d", "--doc", help = "Create a documentation about the training")
ArgParser.add_argument("-s", "--save", help = "Save the model to the given path")
args = vars(ArgParser.parse_args())

if(args["mode"] == "demo"):
    print("Use demo mode")

    if(args["model"][0] == "train"):
        # Train a new model
        Solver.LoadTrainingData(args["model"][1] + "\\train", args["model"][1] + "\\preprocessing")
        Solver.TrainModel()

        if(args["doc"] is not None):
            # Save training documentation to disk
            Solver.Report(args["doc"][0])
            Solver.PrintModel(args["doc"][0])
    elif(args["model"][0] == "load"):
        # Load the model from disk
        Solver.LoadModel(args["model"][1])

    # Start captcha classification
    print("[INFO] Use {} captchas...".format(len(args["input"])))
    print("[INFO] Start prediction...")

    # Check if input path is a directory and read all files
    if(os.path.exists(args["input"][0])):
        if(os.path.isdir(args["input"][0])):
            print("[INFO] Read directory {}...".format(args["input"][0]))
            InputFiles = [os.path.join(args["input"][0], f) for f in os.listdir(args["input"][0])]
        else:
            InputFiles = args["input"]
    else:
        print("[ERROR] Unknown path or image!")

    print("[INFO] Read {} files...".format(len(InputFiles)))

    for [CaptchaNr, Captcha] in enumerate(InputFiles):
        print("[INFO] Process captcha {}/{}...".format(CaptchaNr + 1, len(InputFiles)))
        NewImage = Solver.Predict(Captcha)

        cv2.imshow("Captcha", NewImage)
        cv2.waitKey(0)

    # Save the model
    if(args["save"] is not None):
        Solver.SaveModel(args["model"][1])

else:
    print("Use live mode")
