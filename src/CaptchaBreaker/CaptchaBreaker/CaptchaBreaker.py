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
ArgParser.add_argument("-i", "--input", help = "Input files for classification", nargs = '+')
ArgParser.add_argument("-d", "--doc", help = "Create a documentation about the training", nargs = 5)
ArgParser.add_argument("-s", "--save", help = "Save the model to the given path")
args = vars(ArgParser.parse_args())

if(args["mode"] == "demo"):
    print("[INFO] Use demo mode")

    if(args["model"][0] == "train"):  
        # Load trainingdata
        if(Solver.LoadTrainingData(args["model"][1] + "\\train", args["model"][1] + "\\preprocessing")  < 0):
            exit()

        # Train a new model
        if(Solver.TrainModel() < 0):
            exit()

        if(args["doc"] is not None):
            # Save training documentation to disk
            Solver.Report(args["doc"][0])
            Solver.PrintModel(args["doc"][0])
    elif(args["model"][0] == "load"):
        # Load the model from disk
        Solver.LoadModel(args["model"][1])

    # Check if input set, if it is a path to a directory and read all files
    if(args["input"] is not None):
        if(os.path.exists(args["input"][0])):
            if(os.path.isdir(args["input"][0])):
                # Start captcha classification
                print("[INFO] Read directory. Found {} captchas...".format(args["input"][0]))
                InputFiles = [os.path.join(args["input"][0], f) for f in os.listdir(args["input"][0])]
            else:                
                print("[INFO] Use {} captchas...".format(len(args["input"])))
                InputFiles = args["input"]
        else:
            print("[ERROR] Unknown path or image!")

        print("[INFO] Read {} files...".format(len(InputFiles)))
        print("[INFO] Start prediction...")

        for [CaptchaNr, Captcha] in enumerate(InputFiles):
            print("[INFO] Process captcha {}/{}...".format(CaptchaNr + 1, len(InputFiles)))
            NewImage = Solver.Predict(Captcha, Debug = True)

            # Show the captcha
            cv2.imshow("Captcha", NewImage)
            Key = cv2.waitKey(0)
    else:
        print("[INFO] No input files!")

    # Save the model
    if(args["save"] is not None):
        Solver.SaveModel(args["save"])

else:
    print("[INFO] Use live mode")
    print("[ACTION] Press <ESC> to cancel")

    if(args["model"][0] == "load"):
        # Load the model from disk
        Solver.LoadModel(args["model"][1])

    Solver.EnableLiveMode()

    while True:
        pass