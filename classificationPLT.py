import os.path
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from keras.models import load_model 

def find_contours(dilationimage):
	
	# Get Contours in Image
	_, contours, hierarchy = cv2.findContours(dilationimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Find Contours
	for cnt in contours:
		area = cv2.contourArea(cnt)
		#print(area)
		# If area is within these paramters then it will skip over the current contour
		if area < 500 or area > 5500:
			continue
		
		# An ellipse needs at least 5 points to draw
		if len(cnt) < 5 :	
			continue
			
		# Draw ellipse around contours
		ellipse = cv2.fitEllipse(cnt)
		cv2.ellipse(img.copy(), ellipse, (0,255,0), 2)
		
		# Get dimension of bounding box for each contour / coin
		(x, y, w, h) = cv2.boundingRect(cnt)
		
		# If height (h) is less than 34 then it will skip over the current contour
		if h < 34:	
			continue
			
		# Crop coin image from the original input image so that we can get the best resolution to compare against the model
		import matplotlib.pyplot as plt
		input_image2 = plt.imread(file_path) #Read in the image (3, 14, 20)
		coin = input_image2[int(y*width/FRAME_WIDTH) : int((y + h)*width/FRAME_WIDTH), int(x*width/FRAME_WIDTH) : int((x + w)*width/FRAME_WIDTH)]

		
		classify(coin)

	# Draw Contours on Image
	cv2.drawContours(img, contours, -1, (0,0,255),2)
	cv2.imshow('All Contours', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def classify(coinimage):
	global i
	# Convert image to RGB
	coin_rgb = cv2.cvtColor(coinimage, cv2.COLOR_BGR2RGB)
	#cv2.imshow("coin_rgb", coin_rgb)
	# Resize image to the same size that the model was trained in
	from skimage.transform import resize
	my_image_resized = resize(coinimage, (img_size,img_size,3)) 
	#my_image_resized = cv2.resize(coinimage, (img_size, img_size))
	#cv2.imshow("my_image_resized", my_image_resized)
	
	probabilities = model.predict(np.array( [my_image_resized,] ))
	# Label names
	labels = ['2e', '1e', '50c', '20c', '10c', '5c', '2c', '1c']
	# Sort the probabilities from lowest to highest
	index = np.argsort(probabilities[0,:])

	print("Coin #{}".format(i + 1))
	# Print probabilities
	print("1st:", labels[index[7]], "- Probability:", probabilities[0,index[7]])
	print("2nd:", labels[index[6]], "- Probability:", probabilities[0,index[6]])
	print("3rd:", labels[index[5]], "- Probability:", probabilities[0,index[5]])
	
	# Show coin detected with labelled denomination and probability
	cv2.putText(coinimage, "{} - {:.2f}%".format(labels[index[7]], probabilities[0,index[7]]*100) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
	cv2.imshow("Coin{}".format(i + 1), coinimage)
	filename = "Coin{}.jpg".format(i + 1)
	cv2.imwrite(filename, coinimage)
	i = i + 1



# Choose input image
root= tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
print(file_path)

FRAME_WIDTH = 600
img_size = 100
input_image = cv2.imread(file_path)

# Resize Image
height, width, depth = input_image.shape
img = cv2.resize(input_image, (int(FRAME_WIDTH), int(FRAME_WIDTH * height / width)))

# Greyscale
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Greyscale', grey)
#cv2.waitKey(0)

# Blurring
blur = cv2.GaussianBlur(grey, (7, 7), 0)
#cv2.imshow('Blur', blur)
#cv2.waitKey(0)

# Adaptive Thresholding
adThresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 5)
#cv2.imshow('Adaptive4',adThresh)
#cv2.waitKey(0)

# Erosion
kernel = np.ones((2,2), np.uint8)
erosion = cv2.erode(adThresh, kernel, iterations = 3)
#cv2.imshow('Eroded', erosion)
#cv2.waitKey(0)

# Dilate the image
kernel = np.ones((2,2),np.uint8)
dilation = cv2.dilate(erosion, kernel, iterations=1)
#cv2.imshow('Dilate',dilation)
#cv2.waitKey(0)
	
i = 0	# Variable to iterate over the contours / coins

# Load the model
if os.path.isfile('EuroCNN100_16.h5'):
	model = load_model('EuroCNN100_16.h5')
	print ("Model loaded")
else:
	print ("Model doesn't exist not exist")
	
find_contours(dilation)