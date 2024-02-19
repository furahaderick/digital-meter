import cv2
import pytesseract
import matplotlib.pyplot as plt
# import numpy as np

img_path = "./data/reading.png"
img = cv2.imread(img_path)
### Expected reading - "069871"

### Image preprocessing for optimization
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
median = cv2.medianBlur(img, 9)

ret1, thresh1 = cv2.threshold(median, 100, 240, cv2.THRESH_BINARY_INV)

### Extracting text from the picture and printing it
config = r"--oem 3 --psm 9 -c tessedit_char_whitelist=0123456789"
text = pytesseract.image_to_string(thresh1, config=config)
print(text)