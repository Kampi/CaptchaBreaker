import os
import time
import shutil
from random import shuffle
from string import ascii_lowercase

Dir1 = "D:\\Dropbox\\GitHub\\Machine-Learning\\CaptchaBreaker\\data\download\\preprocessing"
Dir2 = "D:\\Dropbox\\GitHub\\Machine-Learning\\CaptchaBreaker\\data\\preprocessing"

OutputDir = "D:\\Dropbox\\GitHub\\Machine-Learning\\CaptchaBreaker\\data\\mixed\\preprocessing"

DirList = [Dir1, Dir2]

LetterList = []

# Loop over each directory
for Dir in DirList:
    if(os.path.exists(Dir)):
        # Check if path is a directory
        if(os.path.isdir(Dir)):
            # Store all files
            FolderList = os.listdir(Dir)

    for Folder in FolderList:
        Path = Dir + "\\" + Folder
        for File in os.listdir(Path):
            LetterList.append(Path + "\\" + File)

# Shuffle the list
shuffle(LetterList)

# Create directories
for Char in ascii_lowercase:
    if(not(os.path.exists(OutputDir + "\\" + Char))):
        os.mkdir(OutputDir + "\\" + Char)

for Number in range(0, 10):
    if(not(os.path.exists(OutputDir + "\\" + str(Number)))):
        os.mkdir(OutputDir + "\\" + str(Number))

for [FileNr, File] in enumerate(LetterList):
    # Get all indexes
    Index = File.rfind('\\')
    LetterFolderIndex = File.rfind('\\', 0, File.rfind("\\"))

    # Create the new filepath
    FileName = File[Index:].replace("\\", "")
    LetterFolder = File[LetterFolderIndex:Index].replace("\\", "")
    OutputPath = OutputDir + "\\" + LetterFolder + "\\" + str(FileNr) + ".jpg"

    # Copy the file to the new path
    shutil.copy(File, OutputPath)

# Rename each file in each folder
DirList = os.listdir(OutputDir)
for Dir in DirList:
    Directory = OutputDir + "\\" + Dir
    Files = os.listdir(Directory)

    # Rename each file with a new number
    for [FileNr, File] in enumerate(Files):
        os.rename(Directory + "\\" + File, Directory + "\\" + str(FileNr + 1) + ".png")