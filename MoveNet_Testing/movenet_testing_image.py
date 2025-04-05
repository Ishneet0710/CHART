import cv2
from PIL import Image
from MoveNet_Processing_Utils import movenet_processing

original = cv2.imread('./test_images/test_img.png') 	# Change this to the path of the image to be tested
processed = Image.fromarray(movenet_processing(original[:, :, ::-1], max_people=1, \
                                               mn_conf=0.05, pred_conf=1, blur_faces=False), 'RGB') # Converting BGR to RGB
processed.save("./test_images/processed_img.png") 		# Change this to the path where the processed image will be saved
