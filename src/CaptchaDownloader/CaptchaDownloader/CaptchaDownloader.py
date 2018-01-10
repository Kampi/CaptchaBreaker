import io
import time
import imghdr
import argparse
import urllib.request
from PIL import Image

CaptchaURL = "https://www.telekom.de/is-bin/INTERSHOP.enfinity/WFS/EKI-PK-Site/de_DE/-/EUR/ViewGeneratedContent-CreateCaptcha?CaptchaProtectionFor=quickcheck"
ProjectDir = "D:\\Dropbox\\GitHub\\Machine-Learning\\CaptchaBreaker\\"

# Set commandline arguments
ArgParser = argparse.ArgumentParser()
ArgParser.add_argument("-c", "--count", help = "Number of captchas for download", required = True)
args = vars(ArgParser.parse_args())

for Captcha in range(0, int(args["count"])):
    Path = ProjectDir + "data\\download\\" + str(Captcha)

    # Open the url
    Response = urllib.request.urlopen(CaptchaURL)
    
    # Read the data from the url
    Request = Response.read()

    # Check if the file and convert it to jpg
    Type = imghdr.what(io.BytesIO(Request))
    if(Type == "gif"):
        print("[INFO] Found gif image. Start conversion...")

        # Convert the image to jpg
        Img = Image.open(io.BytesIO(Request))
        RGB = Img.convert("RGB")
        RGB.save(Path + ".jpg")

    # Some output
    print("[INFO] Save captcha {} / {}".format(Captcha + 1, int(args["count"])))

    time.sleep(0.5)