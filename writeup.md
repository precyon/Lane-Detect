## Writeup Template

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms and gradients and thresholds to create a thresholded binary image.
* Apply a perspective transform to rectify binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibresult.png "Undistorted"
[image2]: ./output_images/perspresult.png "Perspective corrected"
[image3]: ./output_images/threshresult.png "Thresholded binary example"
[image4]: ./output_images/laneresult.png "Lane line detection"
[image5]: ./output_images/outresult.png "Final output"
[image6]: ./output_images/diagresult.png "Diagnostic output"

## Rubric Points

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

All code for the project is in the ```FindLanes.py``` file. In particular, the camera calibration is performed by the ```cameraCalib()``` function in the same file.   

I start by preparing *world points*, which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane such that ```z = 0```. I also assume that world points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

Camera calibration depends only on the camera configuration. Therefore this calibration is performed just once and the calibration parameters are pickled and stored on the disk. If the code finds this pickled file, it does not perform the calibration and just reads the parameters off the file. 

### Pipeline (single frame)

This section will describe the recurrent processing steps that are done on every frame in the video to detect and annotate the lane lines.

#### 1. Distortion correction

Distortion correction is done using the camera calibration stored in a pickled file. The ```undistort``` function in ```OpenCV``` is used for this purpose. 

#### 2. Perspective transform

Perspective transformation and thresholding can be done in any order. We chose to perform perspective correction first because the resulting image only has the features we're looking for - the lane lines. This helps us design and tune our thresholding algorithms better. 

Perspective transformation depends on the pose of the camera with respect to the surface of the road. Though the camera is fixed, its position changes because of slope of the road or movement of the vehicle because of the suspension. We, however, chose to ignore these effects and assume a constant transformation. Following are the points we used.
 
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |


To calculate this transformation, the source points were chosen manually. These source points were mapped to manually chosen destination points. ```computePerpectiveTransforms()``` function in our code uses ```getPerspectiveTransform()``` function from ```OpenCV``` to compute both the forward and inverse perspective transforms. The following set of images shows the performance of perspective correction. The lanes look reasonably parallel in the corrected image.

![alt text][image2]

#### 3. Thresholding to select features

Like mentioned in the previous point, thresholding of the image was performed after the perspective transformation because it allows us tune the design only for the lanes and not consider its affect on other irrelevant parts of the scene. Various thresholds were attempted and tuned. 

1. Gray level
2. RGB level
3. HSV and HSL levels
4. Gradients along x, y, their total magnitude and direction
5. Laplacian (to capture the dark-light-dark features of the lane lines)

However, the following thresholding sequence was finalized. The ``` thresholdFrame(img) ``` functions contains this logic and the threshold values
1. Yellow and white colours were segmented using thresholds in the HSV space. The following piece of code was used.
```cBin = cv2.inRange(hsv, hsvLowYellow, hsvHighYellow) | cv2.inRange(hsv, hsvLowWhite, hsvHighWhite)```. The values of the thresholds are in the function and are not duplicated here.
2. Since the lanes mostly go vertical, the Sobel gradient in the x-direction was used. The gradient was applied on the S channel of the HLS colourspace since gray channel did not give adequate performance. The following code snippet does this
```
sobelx = np.absolute(cv2.Sobel(hls[:,:,1], cv2.CV_64F, 1, 0, ksize=sobelSize))
sobelx = sobelx/np.max(sobelx)
gmxBin = (sobelx > gradxThresh[0]) & (sobelx <= gradxThresh[1])
```
3. A gray threshold is used to select the right parts from the gradient image. This ensures that some large gradients like construction or asphalt artefacts that are darker in colour (hence lower in gray level) do not get selected.
```
gray = 0.5*img[:,:,0] + 0.4*img[:,:,1] + 0.1*img[:,:,2]
grayBin = (gray > grayThres[0]) & (gray <= grayThres[1])
```
4. The thresholds were then combined as follows to create he final image
```return ((cBin) | (grayBin & gmxBin))```

The following image shows the thresholding performance. The lane lines and markers are reasonably well selected.
  
![alt text][image3]

#### 4. Lane line detection

Two methods have been used to detect lane lines from the pixels
1. **Sliding window method**: This method is straight up from the class notes and will not be described in detail. Our implementation is in the ```slidingLanePixelSearch()``` function in the code. Following modifications were done to increase the robustness of the method
	1. Outlier detection was performed on the detected pixels for a noise-robust fitting. The code: 
	```
	def rejectOutliers(x, y, m = 2.):
    # Since lines are mostly vertical, we reject outliers in the x-direction
    x, y = np.array(x), np.array(y)
    d = np.abs(x - np.median(x))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    select = s<m
    return list(x[select]), list(y[select])

	```
	2. To compute a robust base starting value, various peak detection methods were tried. However, a simple ```argmax``` followed by thresholding was good enough for the case at hand. Code snippet is as follows:
	```
	def calcBaseIdx(signal, threshold=10):
    if np.any(signal):
        idxMax = np.argmax(signal)
        if signal[idxMax] >= threshold:
            return idxMax

    return signal.shape[0]//2
	```
	3. The first (or the base) window was made twice as large to increase the chances of catching the lane pixels. Outlier detection would take care of rejecting the extra pixels not part of the lane.
	4
2. ***Look-ahead filter***: It is assumed that lane lines don't significantly change from frame-to-frame. If a lines have been successfully found in the previous frame, search is performed only in a small margin around the last lane line. We implemented is as follows:
	```
	# Create a mask from the lines of the previous frame
	pts = np.transpose(np.vstack([x_fit, self.yvals])).reshape((-1,1,2)).astype(np.int32)
	cv2.drawContours(self.mask, linePts, -1, (255,255,255), thickness=settings['maskSearchThickness'])

	# Search only in the mask
	def searchInMask(self, img):
	    img = img.astype(np.uint8)
	    masked = cv2.bitwise_and(img, self.mask)
	    pts = cv2.findNonZero(masked)
	    if pts is not None:
	        pts = pts.reshape((-1,2))
	        self.update(pts[:,0], pts[:,1], 0)
	    else:
	        self.detected = False
	
	```
Example of the detected lane lines along with the search mask is shown below

![alt text][image4]

#### 5. Radius of curvature and vehicle position

Radius of curvature in distance units can be found out by fitting the a curve again after multiplying the pixels with pixels/meter scaling. Performing the fitting twice is an expensive operation. Therefore, we instead chose to compute an analytical expression for the radius in distance units in terms of the polynomial coefficients in the pixels units. The following function does that:

```	
def curvature(self, x, y, yEval):
    yEval = yEval*settings['yScale']
    fit = self.currFit
    sx, sy = settings['xScale'], settings['yScale']
    A = fit[0]*sx/sy/sy
    B = fit[1]*sx/sy
    return ((1 + (2*A*yEval + B)**2)**1.5) / np.absolute(2*A)
```	

#### 6. Final output

The lanes lines and the filled polygon that they enclose is project back to the original camera image using the inverse perspective transform. The result is as shown below

![alt text][image5]z


### Pipeline (video)

The following videos are provided:
1. Annotated video with detection lane drawn [here](./project_video_out.mp4)
2. A diagnostic version of the above video that also shows intermediate processing [here](./project_video_out_diag.mp4). The image below shows an example frame from this video.

![alt text][image6]

### Discussion


**Filtering**
Raw lane lines vary quite a bit between frames. We therefore smooth the fitted lines. A lot of smoothing options are possible including low-pass filtering the final polynomial coefficients. We have implemented a filtering scheme where we keep a track of line points detected in the last ```7``` valid frames. Points detected in the current frame are added to this list and a polynomial fit is found for all ```n=8``` frames together. This gives our filter some desirable properties:
1. Fairly easy to compute in the same pipeline. No extra filtering functions are needed
2. Points from each frame automatically get weighted according to their number. A frame with small number of detected points are likely to be invalid or noisy. With our scheme they are automatically outweighed by frames where the number of points are large.        

The implementation is as follows:
```
# pastPixx and pastPixy are deques that automatically only keep the 
# data from last 8 frames
self.pastPixx.append(x)
self.pastPixy.append(y)

# Find the best fit line
fit = self.fitLine(list(chain.from_iterable(self.pastPixx)),
               list(chain.from_iterable(self.pastPixy)))
self.currFit = fit

```

**Sanity check**
We have also implemented very basic sanity check on the pixels that contribute to our fits. It was observed that bad line fits were caused by data that is either very small in number or that is lumped very closely together. The later does not define a curve very well and cause largely undesirable fits. Therefore a check on the pixel count and their standard deviation was added to avoid this issue. 

```
if len(y) > settings['minPixelsForFit'] and np.std(y) > settings['minStdForFit']:
    
	<code to go ahead with fitting>
    self.detected = True
    self.dropCount = 0
else:
	<drop this frame>
	self.detected = True
    self.dropCount += 1
```


**Search logic**
As discussed above, we have two methods that to detect potential lane line pixels from the thresholding image - a sliding window search that searches from scratch and a mask-based one that uses fits from the previous frames. The program always starts with the sliding window search. As lane lines are discovered which pass the sanity check, we switch to the mask-based search. If that does not yield good fits for continuously ``` settings['dropThresh'] = 10``` frames, we switch back to the sliding window search. Snippet:

```
if (leftLine.xLine is None) or (rightLine.xLine is None) or \
   (leftLine.dropCount > settings['dropThresh']) or (rightLine.dropCount > settings['dropThresh']):
    yLeft, xLeft, yRight, xRight, winOver = slidingLanePixelSearch(threshImg, visualize=True)
	<other code>    
else:
    leftLine.searchInMask(threshImg)
    rightLine.searchInMask(threshImg)
    <other code>
```
  
**Analytical computation of radius of curvature**
As already described earlier, radius of curvature is computed analytically to avoid running polynomial fit twice for every frame.


#### Improvements
The cases where our pipeline fails are fairly easy to find - just by running the challenge videos. We did not considered those during its design and consequently the pipeline fails. There could be significant improvement in thresholding logic
1. Gradient and direction thresholds to improve performance under different lighting conditions
2. We have used just the x-gradient. This somewhat fails in presence of sharp curves when x-gradient weakens and y-gradient strengthens. Gradient magnitude could be of help here.
3. Contrast enhancement to improve performance under windshield glare

In addition,
1. Perspective transform is not constant in general. By using a less accurate but faster method for lane detection,  source points could be found automatically.
2. We compute fits for left and right lanes independently. The lane lines, however, are related to each other by parallelism and other constraints. It is therefore possible to parametrize the lines differently and compute fits that are more likely to pass our sanity checks.