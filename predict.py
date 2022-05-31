import cv2
from sklearn import svm
import numpy as np
import glob
import functools
from sklearn.preprocessing import StandardScaler
import timeit
import pickle
import argparse

parser = argparse.ArgumentParser(description='A male/female handwriting classifier.')
parser.add_argument('-i', '--inputdir',
                    help='The path to the test images directory.')
parser.add_argument('-o', '--outputdir',
                    help='The path to the (times.txt & results.txt) directory.')
parser.add_argument('-c', '--classifier',
                    help='The path to the pickled classifier to use.')
parser.add_argument('-s', '--standerd_scaler',
                    help='the path to the pickled standerd scaler',
                    )
args = parser.parse_args()


try:
   with open(args.classifier, 'rb') as clf_file:
        svm_clf = pickle.load(clf_file)
except Exception as e:
    print(f"Couldn't unpickle the classifier {args.classifier}\nError: {e}")
try:
   with open(args.standerd_scaler, 'rb') as std_file:
        std_scaler = pickle.load(std_file)
except Exception as e:
    print(f"Couldn't unpickle the std scaler {args.standerd_scaler}\nError: {e}")

'''
takes a function and it's arguments and then call it and calculate the time it takes to run
and return it with the returned value
'''
def function_timer(func, arg1, arg2, arg3):
    s = timeit.default_timer()
    val = func(arg1, arg2, arg3)
    f = timeit.default_timer()
    return (round((f-s)*100)/100.0) , val
'''
ceates a file with the given name and returnthe file object
if the file already exist it opens it and wipes it out
'''
def open_file(fileName):
    fileName = args.outputdir + '/' + fileName
    return open(fileName, "w")  #a --> for append       w --> for overwrite
'''
write a specific number in a given file object
'''
def write_in_file(file,number):
    file.write(str(number)+"\n")
    return
'''
closes a given file object
'''
def close_file(file):
    file.close()
    return

def Preprocessing (image) : 
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)  # convert color from RGB to GRAY
    height, width = image.shape # get image dimensions
    img = cv2.GaussianBlur(image, (9, 9), 0) #decrease noise for dialation
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 101, 30) # apply threshold on blured image
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 101, 30)  # apply threshold on original image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 20)) 
    img = cv2.dilate(img, kernel, iterations=8)
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] 
    biggest_contour = functools.reduce(lambda c1, c2: c1 if cv2.contourArea(c1) > cv2.contourArea(c2) else c2,contours) #find the biggest contour for text area
    x, y, w, h = cv2.boundingRect(biggest_contour) # find smallest rect that can contain the text area after dialation
    image = image[y:y + h, x:x + w]
    return image

#define some constants for hinge 
N_ANGLE_BINS = 40
BIN_SIZE = 360 // N_ANGLE_BINS
LEG_LENGTH = 25

def get_contour_pixels(bw_image):
        contours, _= cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
        # contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
        
        img2 = bw_image.copy()[:,:,np.newaxis]
        img2 = np.concatenate([img2, img2, img2], axis = 2)
        return contours

def get_hinge_features(bw_image):
        
        contours = get_contour_pixels(bw_image)
        
        hist = np.zeros((N_ANGLE_BINS, N_ANGLE_BINS))
            
        # print([len(cnt) for cnt in contours])
        for cnt in contours:
            n_pixels = len(cnt)
            if n_pixels <= LEG_LENGTH:
                continue
            
            points = np.array([point[0] for point in cnt])
            xs, ys = points[:, 0], points[:, 1]
            point_1s = np.array([cnt[(i + LEG_LENGTH) % n_pixels][0] for i in range(n_pixels)])
            point_2s = np.array([cnt[(i - LEG_LENGTH) % n_pixels][0] for i in range(n_pixels)])
            x1s, y1s = point_1s[:, 0], point_1s[:, 1]
            x2s, y2s = point_2s[:, 0], point_2s[:, 1]
            
            phi_1s = np.degrees(np.arctan2(y1s - ys, x1s - xs) + np.pi)
            phi_2s = np.degrees(np.arctan2(y2s - ys, x2s - xs) + np.pi)
            
            indices = np.where(phi_2s > phi_1s)[0]
            
            for i in indices:
                phi1 = int(phi_1s[i] // BIN_SIZE) % N_ANGLE_BINS
                phi2 = int(phi_2s[i] // BIN_SIZE) % N_ANGLE_BINS
                hist[phi1, phi2] += 1
                
        normalised_hist = hist / np.sum(hist)
        feature_vector = normalised_hist[np.triu_indices_from(normalised_hist, k = 1)]
        
        return feature_vector

def get_features(img):
    hinge_features = get_hinge_features(img)
    return hinge_features

'''
function that test a given image with a given svm model and returns it's prediction 
valu:
1 --> male
0 --> females
'''
def svm_test(svm_clf,std_scaler, img):
    img = Preprocessing(img)
    features = get_features(img)
    features = std_scaler.transform([features])
    return  svm_clf.predict(features)[0]





#open 2-files to write the output in them
results_file = open_file("results.txt")
time_file = open_file("times.txt")


for file in sorted(glob.glob((args.inputdir)+'/*.jpg')):   
    try: 
        #print(file,"\n")
        s_error = timeit.default_timer()
        img = cv2.imread(file)  #read male images
        time,value = function_timer(svm_test, svm_clf, std_scaler, img)
        if time == 0 :
            time += 0.01
        write_in_file(time_file,time)
        write_in_file(results_file,int(value))
    except Exception as e:
        print(e)
        write_in_file(results_file,-1)
        time = round((timeit.default_timer()-s_error)*100)/100.0
        if time == 0 :
            time += 0.01
        write_in_file(time_file,time)


close_file(results_file)
close_file(time_file)

print("finished successfully \n")