## Writeup Project 5

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[vehicleExample]: ./examples/1.png
[notvehicleExample]: ./examples/extra202.png
[hogImage]: ./output_images/hogImage.png
[firstWindowVisualize]: ./output_images/firstWindowVisualize.png
[secondWindowVisualize]: ./output_images/secondWindowVisualize.png
[boxesAndHeatMap]: ./output_images/boxesAndHeatMap.png
[labeledCars]: ./output_images/labeledCars.png

---

### Data Set

The training data from the KITTI and DTI databases. They are divided into "car" and "not car" samples. Here is one example of each:

![vehicle][vehicleExample]
![not vehicle][notvehicleExample]

### Histogram of Oriented Gradients (HOG)

#### Extracing HOG features

HOG is useful as it not only takes the gradient (which we know from lane finding is an effective way to find lines) it also preserves information about the direction of the gradient. This makes it easier to identify shapes. Here is an example of what the HOG looks like for a vehicle image:

![HOG example][hogImage]


#### HOG parameters.

I left the `skimage.hog()` orientations parameter at 9, pixels_per_cell at 9, and cells_per_block at 2. It's possible to get more granular features and better prediction accuracy by changing the parameters, but it came at the cost of more time to compute the HOG. These parameter values give sufficient accuracy and take less time (though it still isn't fast).

After some testing, I found that the best color space was the YCrCb color space. I transformed the images to YCrCb and used the 1st channel to compute the gradient. This produced the best results out of any other colors I tried.

### Color features

I also extracted features from the color information in the images. The first is the spatial positioning of the colors. This means just taking the image and lining up the pixels from left to right.

The second is the distribution of colors throughout the image. This involves breaking each image into it's three color channels then calculating a histogram of color strength on each channel. I broke the histogram into 32 bins, so essentially I'm counting the number of pixels with a color strength between 0-7, 8-15, 16-23, etc.

### Creating the classifier

#### Collecting data

I calculated features for all the images in the training datasets. I created a corresponding array of "truth" values by marking 1 for vehicle images and 0 for non-vehicles. Then I shuffled the data and split it into a training set and a test set. Lastly, I calculated a scaler based on the training set, then applied that scaler to all the data.

#### Training

I used a linear SVC without modifying any of the default parameters. I trained on the training set and my test set had an accuracy of 99%.

### Sliding Window Search

#### First approach

I started with the basic sliding window search where I defined the window size and the step size. I would extract each piece of the image and resize it then calculate all the features on that piece. This worked, but was incredibly slow (calculating HOG features is time expensive). In order to decide the window size and where on the image to search I displayed the boxes on a test image and inspected the relative sizes of the cars. Here is a visualization of the windows I searched:

![Visualization of windows][firstWindowVisualize]

#### Second approach

To speed up the searching I used HOG subsampling so I only had to calculate HOG once per window size, instead of once per window. I scaled the whole image to the desired window size, calculated the HOG for the whole image, then stepped over the desired windows and combined the existing HOG features. Color features were calculated the same as before.

Here are some images that show the windows that I used:

![Visualization of windows][secondWindowVisualize]

### Heatmap

The window search identifies all the windows in the image that the classifier predicts have a car in them. The search identifies several false positives. It also identifies more than one box around the cars.

I create a heat map to filter out the false positives and identify where the car is. Each window increases the values of the pixels it contains by one. In order to be counted as an actual car, a pixel has to have a value of at least 3. Here is an example of an image with all the windows that the classifier identified and then the heat map after filtering out low pixel values:

![car image with boxes][boxesAndHeatMap]

I then use the label function from the scipy library to draw bounding boxes around the identified pixels:

![car image with just car boxes][labeledCars]

This still has some false positives in it. To even further reduce the noise in the video I aggregate the last ten frames of the video and increase the threshold from 3 to 30.


### Image Pipeline

The final pipeline works by combining all the previous pieces. It takes in an image and then does a sliding window search across four scale sizes. For each window it extracts the HOG and color features and passes them to the classifier. If the classifier thinks the window contains a car it saves the window in a list. After all the windows are searched that list is passed to the heat map functionality which reduces the noise and identifies the cars.

In order to further reduce the noise in the video, I track the location of the cars identified in previous frames and then focus an extra round of searching in that area. That extra search helped verify that the object really was a car and when coupled with higher thresholding it reduced the amount of false positives. The extra thresholding was accomplished with the decision_function in the LinearSVC. I only accepted the window as actually a car if the decision_function returned 1 or higher.

---

### Video Results

Here's a [link to my video result](./output_images/project_video.mp4)

The video isn't perfect. There are still some false positives and it loses track of the white car for a few seconds. I definitely wouldn't trust this pipeline to control my car. But it does do a decent job of identifying the cars during most of the video.

---

### Discussion / Improvements

#### Time

The biggest problem with my pipeline is how long it takes. It took about 40 minutes to process the minute long video. That would be unacceptable in a real life scenario where decisions have to be made in an instant. It also greatly hindered my ability to tune parameters since I would have to wait for 40 minutes to see if it made an improvement or not.

The speed of my code could be improved with parallelization. The most time consuming part is computing the HOG features. Each image scale is independent of the others and could be parallelized. This would likely greatly improve the speed issue.

#### Masking

The railing on the bridges produce a lot of false positives. Some sort of intelligent masking could erase those false positives. Obviously we can't always ignore everything to our left, but if we see that we are in the far left lane we could ignore things over there. (at least we would know there aren't cars, we should still be checking for humans or things like that)

#### Window Finding

I just kinda guessed at good window sizes and eyeballed that they seemed to match the actual size of cars in the image. That could use some more formal analysis. It should be possible to use trigonometry to figure out the expected apparent size of a car at certain y positions in the image.