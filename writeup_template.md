**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train common classifiers such as Linear SVM, Logistic Regression and MLP classifier
* A color transform and append binned color features, as well as histograms of color, to the HOG feature vector is applied. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

get_hog_features which will be used to compute a Histogram of Oriented Gradients for a given image.  

```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features
    ```

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I choose 

* color space = 'YCrCb'
* orientations = 9
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = "ALL"

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
```python
car_features = []
notcar_features = []
X_scaler = []
scaled_X =[]
car_features , notcar_features = train_hog(cars, notcars)
if len(car_features) > 0:
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                 
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, len(cars))
    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(cars[car_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
    ```
First, I trained a linear SVM using
```python
svc = LinearSVC()
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()

print('SVC results')
print(t2-t, 'Seconds to train SVC...')
print('accuracy on training data: ', svc.score(X_train, y_train))
print('accuracy on test data: ', svc.score(X_test, y_test))
t=time.time()
prediction = svc.predict(X_test[0].reshape(1, -1))
t2 = time.time()
print(t2-t, 'Seconds to predict with SVC')
    ```
SVC results
24.180999994277954 Seconds to train SVC...
accuracy on training data:  1.0
accuracy on test data:  0.989864864865
0.0019998550415039062 Seconds to predict with SVC

Second, I trained Logistic Regression  
```python
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(max_iter=10)
t=time.time()
lrc.fit(X_train, y_train)
t2 = time.time()
print(t2-t, 'Seconds to train LRC...')
print('Train Accuracy of LRC = ', lrc.score(X_train, y_train))
print('Test Accuracy of LRC = ', lrc.score(X_test, y_test))
t=time.time()
prediction = lrc.predict(X_test[0].reshape(1, -1))
t2 = time.time()
print(t2-t, 'Seconds to predict with LRC')
    ```
With following results

43.76300001144409 Seconds to train LRC...
Train Accuracy of LRC =  1.0
Test Accuracy of LRC =  0.990709459459
0.0010001659393310547 Seconds to predict with LRC

Finally I trained MLP

```python
mlp = MLPClassifier(random_state=999)
t=time.time()
mlp.fit(X_train, y_train)
t2 = time.time()
print('MLP results')
print(t2-t, 'Seconds to train MLP...')

print('accuracy on training data: ', mlp.score(X_train, y_train))
print('accuracy on test data: ', mlp.score(X_test, y_test))
t=time.time()
prediction = mlp.predict(X_test[0].reshape(1, -1))
t2 = time.time()
print(t2-t, 'Seconds to predict with MLP')
    ```
And I obtained the best results
MLP results
37.37000012397766 Seconds to train MLP...
accuracy on training data:  1.0
accuracy on test data:  0.996621621622
0.0009999275207519531 Seconds to predict with MLP
...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions by using the function :
```python
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.75, 0.75)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) 
    ny_windows = np.int(yspan/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = (xs+1)*nx_pix_per_step + x_start_stop[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = (ys+1)*ny_pix_per_step + y_start_stop[0]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list
    ```
    
![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I use the OpenCV function cv2.findContours to find all of the objects in the mask image and once the contours are found I used the OpenCV function cv2.boundingRect on each contour to get the coordinates of the bounding rect for each vehicle.

A class named as boxes is to store the detected bounding rectangles history from each of the previous 12 frames in a list in order to track vehicles. The OpenCV function cv2.groupRectangles is to combine overlapping rectangles in to consolidated bounding boxes if there are greater than 10 overlapping boxes in the list. In other words, a detection is filtered out if it is classified as a vehicle less than 10 out of 12 consecutive frames


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This pipleline may fail if there are non-trained objects on the road such as motorbikes big trucks etc.  To fix this, we would have to append our trainining and test sets with images of classified images of bikes, etc and adjust our feature extraction algorithm. Another issue is that the pipeline can not work in real time therefore the search and feature extraction parts shoul be modified. 
