from detection import detect_people
from scipy.spatial import distance as dist
from math import ceil, floor
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import time


def rect_distance(rect1, rect2):
	(x1, y1, x1b, y1b) = rect1
	(x2, y2, x2b, y2b) = rect2
	left = x2b < x1
	right = x1b < x2
	bottom = y2b < y1
	top = y1b < y2
	if top and left:
			return dist.euclidean((x1, y1b), (x2b, y2))
	elif left and bottom:
			return dist.euclidean((x1, y1), (x2b, y2b))
	elif bottom and right:
			return dist.euclidean((x1b, y1), (x2, y2b))
	elif right and top:
			return dist.euclidean((x1b, y1b), (x2, y2))
	elif left:
			return x1 - x2b
	elif right:
			return x2 - x1b
	elif bottom:
			return y1 - y2b
	elif top:
			return y2 - y1b
	else:
			return  -1

# Threshold for human detection minumun confindence
MIN_CONF = 0.3
# Threshold for Non-maxima surpression
NMS_THRESH = 0.2
# Threshold for distance violation
SOCIAL_DISTANCE = 50

weightsPath = "YOLO-LITE/YOLOv3-tiny/yolov3-tiny.weights"
configPath = "YOLO-LITE/YOLOv3-tiny/yolov3-tiny.cfg"

# Load the YOLOv3-tiny pre-trained COCO dataset 
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# Set the preferable backend to CPU since we are not using GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the names of all the layers in the network
ln = net.getLayerNames()
# Filter out the layer names we dont need for YOLO
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Read from video
cap = cv2.VideoCapture("video/5.mp4")

# Start counting time for processing speed calculation
t0 = time.time()
frameCount = 0
violateCountFrame = []
violatePeriodTotal = 0
humanCountFrame = []
humanPeriodTotal = 0
warningTimeout = 0

while True:
	(ret, frame) = cap.read()

	if not ret:
		break

	# Resize Frame to 720p
	frame = imutils.resize(frame, width=720)
	frameCount += 1
	# Get the dimension of the frame
	(frameHeight, frameWidth) = frame.shape[:2]
	# Initialize lists needed for detection
	boxes = []
	centroids = []
	confidences = []

	# Construct a blob from the input frame 
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)

	# Perform forward pass of YOLOv3, output are the boxes and probabilities
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# For each output
	for output in layerOutputs:
		# For each detection in output 
		for detection in output:
			# Extract the class ID and confidence 
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# Class ID for person is 0, check if the confidence meet threshold
			if classID == 0 and confidence > MIN_CONF:
				# Scale the bounding box coordinates back to the size of the image
				box = detection[0:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
				(centerX, centerY, width, height) = box.astype("int")
				# Derive the coordinates for the top left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# Add processed results to respective list
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	# Perform Non-maxima suppression to suppress weak and overlapping boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	humansDetected = []
	violate = set()
	# Process frames only when there are detection
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# Add probability, coordinates and centroid to detection list
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			humansDetected.append(r)

		# Initialize set for violate so only individual will be recorded only once even violate more than once 
		violateCount = np.zeros(len(humansDetected))
		if len(humansDetected) >= 2:
			centroidsDetected = np.array([d[2] for d in humansDetected])
			# Compute euclidean (straight line) distances between all pairs of the detection centroids
			distances = dist.cdist(centroidsDetected, centroidsDetected, metric="euclidean")

			# Loop over the upper triangular of the distance matrix
			for i in range(0, distances.shape[0]):
				for j in range(i + 1, distances.shape[1]):
					# Check if the distance between two points violate the minimum distance
					if distances[i, j] < SOCIAL_DISTANCE:
						# Add violation to set
						violate.add(i)
						violateCount[i] += 1
						violate.add(j)
						violateCount[j] += 1

		for (i, (prob, bbox, centroid)) in enumerate(humansDetected):
			(startX, startY, endX, endY) = bbox
			color = (0, 255, 0)
			if i in violate:
				color = (0, 0, 255)
				cv2.putText((frame), str(int(violateCount[i])), (startX	, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	if (len(violate) > 0):
		warningTimeout = 10
	else: 
		warningTimeout -=1

	if warningTimeout > 0:
		text = "Violation count: {}".format(len(violate))
		cv2.putText(frame, text, (200, frame.shape[0] - 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

	text = "Crowd count: {}".format(len(humansDetected))
	cv2.putText(frame, text, (10, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	

	if frameCount % 5 == 0:
		violateCountFrame.append(ceil(violatePeriodTotal / 5))
		humanCountFrame.append(ceil(humanPeriodTotal / 5))
		violatePeriodTotal = 0
		humanPeriodTotal = 0
	else:
		violatePeriodTotal += len(violate)
		humanPeriodTotal += len(humansDetected)
	
	cv2.imshow("Processed Output", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
t1 = time.time() - t0
print("Frame Count: ", frameCount)
print("Time elapsed: ", t1)
print("Processed FPS: ", frameCount/t1)
timeAxis = [f * 5 for f in range(floor(frameCount/5))]
plt.plot(timeAxis, violateCountFrame, label="Violation Count")
plt.plot(timeAxis, humanCountFrame, label="Crowd Count")
plt.title("Violation & Crowd Count versus Time")
plt.xlabel("Frames")
plt.ylabel("Count")
plt.legend()
plt.show()