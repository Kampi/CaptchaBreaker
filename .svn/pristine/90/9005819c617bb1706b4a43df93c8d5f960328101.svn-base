import sys
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
ArgParser.add_argument("-i", "--input files", help = "Input captcha for classification", nargs = '+', required = True)
ArgParser.add_argument("-d", "--docu", help = "Create a documentation about the training", nargs = 4, required = True)
ArgParser.add_argument("-s", "--save", help = "Save the model to the given path", nargs = 2)
args = vars(ArgParser.parse_args())

if(args["mode"] == "demo"):
    print("Use demo mode")

    if(args["model"][0] == "train"):
        # Train a new model
        Solver.LoadTrainingData(args["model"][1] + "\\train", args["model"][1] + "\\preprocessing")
        Solver.TrainModel()

        if(args["docu"] is not None):
            # Save training documentation to disk
            Solver.Report(args["docu"][0], args["docu"][1], args["docu"][2])
            Solver.PrintModel(args["docu"][0], args["docu"][3])
    elif(args["model"][0] == "load"):
        # Load the model from disk
        Solver.LoadModel(args["model"][1])

    # Start captcha classification
    print("[INFO] Use {} captchas...".format(len(args["input files"])))
    print("[INFO] Start prediction...")

    for [CaptchaNr, Captcha] in enumerate(args["input files"]):
        print("[INFO] Process captcha {}/{}...".format(CaptchaNr + 1, len(args["input files"])))
        Solver.Predict(Captcha)

    # Save the model
    if(args["save"] is not None):
        Solver.SaveModel(args["model"][1])

else:
    print("Use live mode")
