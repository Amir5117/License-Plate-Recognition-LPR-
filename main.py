import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import imutils
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

# Ask the user to select the desired image and display it
Tk().withdraw()
file = askopenfilename()
image = cv2.imread(file)
image = imutils.resize(image, width=500)

# Display the original image using Matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Perform image processing
gray_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_scaled = cv2.bilateralFilter(gray_scaled, 11, 17, 17)
edged = cv2.Canny(gray_scaled, 170, 200)

# Find contours based on edges
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw all contours
image_copy = image.copy()
cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 3)

 # Reverse sort all the contours in terms of area and take the top 30 contours
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
number_plate_contour = None

for current_contour in contours:
    perimeter = cv2.arcLength(current_contour, True)
    approx = cv2.approxPolyDP(current_contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        number_plate_contour = approx
        break

mask = np.zeros(gray_scaled.shape, np.uint8)
new_image = cv2.drawContours(mask, [number_plate_contour], 0, 255, -1)
new_image = cv2.bitwise_and(image, image, mask=mask)

# Display the number plate using Matplotlib
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title('Number Plate')
plt.show()

gray_scaled1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
ret, processed_img = cv2.threshold(gray_scaled1, 125, 255, cv2.THRESH_BINARY)

# Display the processed number plate using Matplotlib
plt.imshow(processed_img, cmap='gray')
plt.title('Number Plate Processed')
plt.show()

# Use tesseract to convert image into string
text = pytesseract.image_to_string(processed_img)
print("Number is:", text)
