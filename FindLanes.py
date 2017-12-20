import os
import sys
import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from collections import deque
from itertools import chain

# All the tunable settings in the program

settings =  {
            'diagnostic': False,
            'calibDims': (9, 6),
            'persrc': np.array([
                   [703, 461], # top right
                   [1000,650], # bottom right
                   [308, 650], # bottom left
                   [580, 461]  # top left
                            ], dtype=np.float32),
            'perdst': np.array([
                   [1000, 0  ], # top right
                   [1000, 700], # bottom right
                   [300 , 700], # bottom left
                   [300 , 0  ]  # top left
                             ], dtype=np.float32),
            'minPixelsForFit': 150,
            'minStdForFit': 100,
            'maskSearchThickness':150,
            'lpf': 0.85,
            'nFilt': 8,
            'yScale': 30/800,  # m per pixel
            'xScale': 3.7/700, # m per pixel
            'dropThresh': 10
            }


# Helper functions for plotting and display

def readFolderToStack(path='./test_images/'):
    """
    Loads all images in the path into a list. Also stores
    the filnames of those images
    """
    imgList = []
    imgNames = []
    for file in os.listdir(path):
        imgList.append(mpimg.imread(path + file))
        filename = os.path.splitext(file)[0]
        imgNames.append(file)

    return imgList, imgNames


def displayImagelist(imgList, cmap=None, cols=2):
    """
    Display all images in a the list imgList
    """
    rows = np.ceil(len(imgList)/cols)

    plt.figure()
    for i, img in enumerate(imgList):
        plt.subplot(rows, cols, i+1)
        if len(img.shape) == 2:
            cmap = 'gray'
        plt.imshow(img, cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.show()


def compareImageList(imgListLeft, imgListRight, cmap=None):
    """
    Here
    """
    cols = 2
    rows = np.ceil(len(imgListLeft))

    plt.figure()
    for row in range(len(imgListLeft)):
        imgLeft  = imgListLeft[row]
        imgRight = imgListRight[row]

        plt.subplot(rows, cols, 2*row + 1)
        plt.imshow(imgLeft, 'gray' if len(imgLeft.shape) == 2 else cmap)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(rows, cols, 2*row + 2)
        plt.imshow(imgRight, 'gray' if len(imgRight.shape) == 2 else cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


def setupAndCalib(calibPath):
    # Chessboard settings
    nx, ny = settings['calibDims']
    imgStack = glob.glob(calibPath)
    mtx, dist = cameraCalib(imgStack, nx, ny)
    return mtx, dist


def cameraCalib(imgStack, nx, ny):
    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    for fname in imgStack:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)


    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    return mtx, dist

class Line():
    def __init__(self, imgSize):
        # was the line detected in the last iteration?
        self.detected = False
        #polynomial coefficients for the most recent fit
        self.currFit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius = None
        self.vehPos = None
        #distance in meters of vehicle center from the line
        #self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float')

        # x values for detected line pixels in current frame
        self.pixx = None
        # y values for detected line pixels in current frame
        self.pixy = None

        self.pastPixx = deque(maxlen=settings['nFilt'])
        self.pastPixy = deque(maxlen=settings['nFilt'])
        self.yLine = np.arange(imgSize[0])
        self.xLine = None
        #self.xFilt = None
        self.mask = np.ones(imgSize, dtype=np.uint8)*255

        self.mode = 0
        self.dropCount = 0


    def fitLine(self, x, y):
        return np.polyfit(y, x, 2)

    def curvature(self, x, y, yEval):
        yEval = yEval*settings['yScale']

        ## Method 1: Fit again
        #x = np.float64(x)*settings['xScale']
        #y = np.float64(y)*settings['yScale']
        #fitCoeffs = self.fitLine(x,y)
        #return ((1 + (2*fitCoeffs[0]*yEval + fitCoeffs[1])**2)**1.5) / np.absolute(2*fitCoeffs[0])


        ## Method 2: Derive analytical expression
        fit = self.currFit
        sx, sy = settings['xScale'], settings['yScale']
        A = fit[0]*sx/sy/sy
        B = fit[1]*sx/sy
        return ((1 + (2*A*yEval + B)**2)**1.5) / np.absolute(2*A)


    def eval(self, y):
        return np.polyval(self.currFit, y)

    def updateMask(self):
        self.mask.fill(0)

    def update(self,  x, y, mode):
        self.mode = mode

        if len(y) > settings['minPixelsForFit'] and np.std(y) > settings['minStdForFit']:
            self.pixx, self.pixy = x, y
            self.pastPixx.append(x)
            self.pastPixy.append(y)


            # Find the best fit line
            fit = self.fitLine(list(chain.from_iterable(self.pastPixx)),
                               list(chain.from_iterable(self.pastPixy)))
            self.currFit = fit

            # Remember the current line
            self.xLine = self.eval(self.yLine)

            self.radius = self.curvature(self.xLine, self.yLine, np.max(self.yLine))
            self.vehPos = self.xLine[-1]

            # Compute a search mask for the next frame
            self.mask.fill(0)
            linePts = np.transpose(np.vstack([self.xLine, self.yLine])).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(self.mask, linePts, -1, (255,255,255), thickness=settings['maskSearchThickness'])

            self.detected = True
            self.dropCount = 0
        else:
            self.currFit = [np.array([False])]
            self.detected = False
            self.dropCount += 1

    def searchInMask(self, img):
        img = img.astype(np.uint8)
        masked = cv2.bitwise_and(img, self.mask)
        pts = cv2.findNonZero(masked)
        if pts is not None:
            pts = pts.reshape((-1,2))
            self.update(pts[:,0], pts[:,1], 0)
        else:
            self.detected = False



def thresholdFrame(img):

    # Gray thresholds
    grayThres = (20, 255)
    gray = 0.5*img[:,:,0] + 0.4*img[:,:,1] + 0.1*img[:,:,2]
    grayBin = (gray > grayThres[0]) & (gray <= grayThres[1])

    # RGB thresholds


    # HSV thresholds
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsvLowYellow  = np.array([ 0, 100, 100])
    hsvHighYellow = np.array([ 50, 255, 255])

    hsvLowWhite  = np.array([0, 0, 200])
    hsvHighWhite = np.array([15, 20,255])

    cBin = cv2.inRange(hsv, hsvLowYellow, hsvHighYellow) | cv2.inRange(hsv, hsvLowWhite, hsvHighWhite)

    # HLS thresholds
    sThres = (170, 255)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    sBin = (s > sThres[0]) & (s <= sThres[1])

    # Gradient thresholds
    sobelSize = 15
    gradxThresh = (0.05, 0.75)
    sobelin = s # choose the input image for gradient calculation
    sobelx = np.absolute(cv2.Sobel(sobelin, cv2.CV_64F, 1, 0, ksize=sobelSize))
    sobelx = sobelx/np.max(sobelx)
    gmxBin = (sobelx > gradxThresh[0]) & (sobelx <= gradxThresh[1])

    #sobely = np.absolute(cv2.Sobel(sobelin, cv2.CV_64F, 0, 1, ksize=sobelSize))
    #sobely = sobely/np.max(sobely)
    ##gmyBin


    ## Calculate the gradient magnitude
    #gradMagThres = (0.25, 1)
    #gradmag = np.sqrt(sobelx**2 + sobely**2)
    #gradmag = gradmag/np.max(gradmag)
    #gmBin = (gradmag > gradMagThres[0]) & (gradmag <= gradMagThres[1])

    ## Calculate the gradient direction
    #gradDirThres = (0.8, 1.2)
    #graddir = np.arctan2(sobely, sobelx)
    #gdBin = (graddir > gradDirThres[0]) & (graddir <= gradDirThres[1])

    ## Calculate the laplacian of gaussian
    #gaussianKernelShape = (5,5)
    #laplacianKernelSize = 21
    #laplacianThresh = 0.1
    #logImg = cv2.GaussianBlur(s, gaussianKernelShape, 0)
    #logImg = cv2.Laplacian(logImg, cv2.CV_64F, ksize=laplacianKernelSize)
    #logBin = (logImg < laplacianThresh*np.min(logImg))


    #return ((grayBin & (gmxBin | sBin))*255).astype(np.uint8)
    return ((cBin==255) | (grayBin & gmxBin))*255

def computePerpectiveTransforms():

    M    = cv2.getPerspectiveTransform(settings['persrc'], settings['perdst'])
    Minv = cv2.getPerspectiveTransform(settings['perdst'], settings['persrc'])

    return M, Minv

def changePerpective(img, M, visualize=False):

    changed = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    if(visualize):
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img)
        colors = ['ro', 'go', 'bo', 'wo']
        pts = settings['persrc']
        for i in range(4):
            plt.plot(pts[i,0], pts[i,1], colors[i])


        plt.subplot(1,2,2)
        plt.imshow(changed)
        pts = state['perdst']
        for i in range(4):
            plt.plot(pts[i,0], pts[i,1], colors[i])

        plt.show()

    return changed

def rejectOutliers(x, y, m = 2.):
    # Since lines are mostly verticals, we reject outliers mostly in the xdirection
    x, y = np.array(x), np.array(y)
    d = np.abs(x - np.median(x))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    select = s<m
    return list(x[select]), list(y[select])

def calcBaseIdx(signal, threshold=10):

    if np.any(signal):
        idxMax = np.argmax(signal)
        if signal[idxMax] >= threshold:
            return idxMax

    return signal.shape[0]//2

def slidingLanePixelSearch(img, nWin=10, margin=200, pixThres = 50, visualize=False):

    height, width = img.shape[0], img.shape[1]
    hist = np.sum(img[:height//2, :], axis=0)

    idxMid   = hist.shape[0]//2

    ## Starting indices
    idxStartLeft  = calcBaseIdx(hist[:idxMid])
    idxStartRight = calcBaseIdx(hist[idxMid:]) + idxMid

    #plt.plot(hist)
    #plt.plot(idxStartLeft, hist[idxStartLeft], 'ro')
    #plt.plot(idxStartRight, hist[idxStartRight], 'bo')
    #plt.show()

    winHeight = np.int(img.shape[0]/nWin)

    pltImg = np.zeros(img.shape, dtype=np.uint8) if visualize else None

    yLeft, xLeft, yRight, xRight = [], [], [], []
    for i in range(nWin)[::-1]:
        winWdith  = 2*margin if i==nWin-1 else margin

        # Read out pixels in the windows
        lWin = img[i*winHeight:(i+1)*winHeight, idxStartLeft-winWdith//2:idxStartLeft+winWdith//2]
        rWin = img[i*winHeight:(i+1)*winHeight, idxStartRight-winWdith//2:idxStartRight+winWdith//2]

        if visualize:
            cv2.rectangle(pltImg, (idxStartLeft-winWdith//2, i*winHeight),
                                  (idxStartLeft+winWdith//2, (i+1)*winHeight),
                                  65, 2)
            cv2.rectangle(pltImg, (idxStartRight-winWdith//2, i*winHeight),
                                  (idxStartRight+winWdith//2, (i+1)*winHeight),
                                  65, 2)

        # Get the non-zero pixels in the windows
        yLeftCur , xLeftCur  = lWin.nonzero()
        yRightCur, xRightCur = rWin.nonzero()


        # Append to lane pixels
        yLeft.extend( yLeftCur  + winHeight*i)
        xLeft.extend( xLeftCur  + idxStartLeft  - winWdith//2)
        yRight.extend(yRightCur + winHeight*i)
        xRight.extend(xRightCur + idxStartRight - winWdith//2)

        #xLeft, yLeft = rejectOutliers(xLeft, yLeft)
        #xRight, yRight = rejectOutliers(xRight, yRight)

        if len(xLeftCur) > pixThres:
            idxStartLeft = np.int(np.median(xLeftCur)) + idxStartLeft - winWdith//2
        if len(xRightCur) > pixThres:
            idxStartRight = np.int(np.median(xRightCur)) + idxStartRight - winWdith//2


    return yLeft, xLeft, yRight, xRight, pltImg


def processFrame(img):

    rightLine, leftLine = state['rightLine'], state['leftLine']
    if rightLine is None:
        rightLine = Line((*img.shape[:2],))
    if leftLine is None:
        leftLine = Line((*img.shape[:2],))


    undistImg = cv2.undistort(img, state['mtx'], state['dist'], None, state['mtx'])
    perspImg = changePerpective(undistImg, state['per_m'])
    threshImg = thresholdFrame(perspImg).astype(np.uint8)

    if (leftLine.xLine is None) or (rightLine.xLine is None) or \
       (leftLine.dropCount > settings['dropThresh']) or (rightLine.dropCount > settings['dropThresh']):
        yLeft, xLeft, yRight, xRight, winOver = slidingLanePixelSearch(threshImg, visualize=True)
        leftLine.update(xLeft, yLeft, 0)
        rightLine.update(xRight, yRight, 0)
    else:
        leftLine.searchInMask(threshImg)
        rightLine.searchInMask(threshImg)
        winOver = rightLine.mask | leftLine.mask


    detected = leftLine.detected and rightLine.detected

    ## Create image for draw output on
    diagImg = np.zeros((*threshImg.shape[:2], 3), dtype=np.uint8)
    #if detected:
    yLine = leftLine.yLine
    xLineLeft  = leftLine.xLine  #np.polyval(leftLineCoeffs, yLine)
    xLineRight = rightLine.xLine #np.polyval(rightLineCoeffs, yLine)

    markImg = np.zeros((*threshImg.shape, 3), dtype=np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    ptsLeft  = np.array([np.transpose(np.vstack([xLineLeft, yLine]))])
    ptsRight = np.array([np.flipud(np.transpose(np.vstack([xLineRight, yLine])))])
    pts = np.hstack((ptsLeft, ptsRight))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(markImg, np.int_([pts]), (0,255,0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    markImg = cv2.warpPerspective(markImg, state['per_minv'], (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistImg, 1, markImg, 0.3, 0)
    # Add text annotations
    error = (img.shape[1] - (rightLine.vehPos + leftLine.vehPos))*settings['xScale']/2
    cv2.putText(result,
            'Mean radius of curvature: %d m'%( np.int(np.mean([leftLine.radius, rightLine.radius]))), (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(result,
            'Cross-track error: %.3f m'%(error), (50,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    ## Create a lane detection diagnostic image
    diagImg[leftLine.pixy, leftLine.pixx] = [255,0,0]
    diagImg[rightLine.pixy, rightLine.pixx]=[0,0,255]

    # Change the annotation based on mode
    diagImg[:,:,1] = winOver
    cv2.polylines(diagImg, np.int32([ptsLeft]), isClosed=False, color=[255, 255, 0], thickness=8)
    cv2.polylines(diagImg, np.int32([ptsRight]), isClosed=False, color=[255, 255, 0], thickness=8)


    if settings['diagnostic']:
        imgTopLeft  = undistImg
        imgTopRight = np.dstack((threshImg,threshImg,threshImg))
        imgBotLeft  = diagImg
        imgBotRight = result
        result = np.vstack((
                np.hstack( (imgTopLeft, imgTopRight) ),
                np.hstack( (imgBotLeft, imgBotRight) )
                )).astype(np.uint8)

    state['rightLine'], state['leftLine'] = rightLine, leftLine

    return result


if __name__=='__main__':

    inFile = sys.argv[1]
    outFile = sys.argv[2]

    calDataFile = 'calibData.p'
    calImageFiles = 'camera_cal/calibration*.jpg'

    if os.path.isfile(calDataFile):
        print('Loading camera calibration ...')
        loaded = pickle.load(open(calDataFile, 'rb'))
        mtx, dist = loaded['mtx'], loaded['dist']
    else:
        print('Computing camera calibration ...')
        mtx, dist = setupAndCalib(calImageFiles)
        toPickle = {}
        toPickle['mtx'] = mtx
        toPickle['dist'] = dist
        pickle.dump(toPickle, open(calDataFile, 'wb'))


    state = {'mtx': mtx,
             'dist': dist,
             'rightLine': None,
             'leftLine': None
             }


    state['per_m'], state['per_minv'] = computePerpectiveTransforms()

    #videoIn = VideoFileClip(inFile).subclip(20, 26)
    #videoIn = VideoFileClip(inFile).subclip(37, 42)
    videoIn = VideoFileClip(inFile)
    videoOut = videoIn.fl_image(processFrame)
    videoOut.write_videofile(outFile, audio=False)

    ## Test thresholding
    #inFile = 'test_images\straight_lines1.jpg'
    #inImg = mpimg.imread(inFile)
    #outImg = processFrame(inImg)
    #f, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    #ax2.set_xticks([])
    #ax2.set_yticks([])
    #ax1.imshow(inImg)
    #ax1.set_title('Input colour')
    #ax2.imshow(outImg, cmap='gray')
    #ax2.set_title('Thresholded binary')
    #plt.savefig('output_images/threshresult.png', bbox_inches='tight')
    #plt.show()

    ## Show perspective correction
    #inFile = 'test_images\straight_lines1.jpg'
    #inImg = mpimg.imread(inFile)
    #outImg = processFrame(inImg)
    #f, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    #ax2.set_xticks([])
    #ax2.set_yticks([])
    #ax1.imshow(inImg)
    #ax1.set_title('Camera')
    #x, y = np.hsplit(settings['persrc'], 2)
    #ax1.plot(x, y, 'r', linewidth=3)
    #ax2.imshow(outImg)
    #ax2.set_title('Corrected')
    #x, y = np.hsplit(settings['perdst'], 2)
    #ax2.plot(x, y, 'r', linewidth=3)
    #plt.savefig('output_images/perspresult.png', bbox_inches='tight')
    #plt.show()

    ## Test camera calibration
    #inFile = 'camera_cal\calibration1.jpg'
    #inImg = mpimg.imread(inFile)
    #outImg = cv2.undistort(inImg, state['mtx'], state['dist'], None, state['mtx'])
    #f, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    #ax2.set_xticks([])
    #ax2.set_yticks([])
    #ax1.imshow(inImg)
    #ax1.set_title('Uncalibrated')
    #ax2.imshow(outImg)
    #ax2.set_title('Calibrated')
    #plt.savefig('output_images/calibresult.png', bbox_inches='tight')
    #plt.show()

    #imgStack, _ = readFolderToStack()
    #outStack = [processFrame(img) for img in imgStack]
    ##compareImageList(imgStack, outStack)
    #displayImagelist(outStack)

    #inFile = 'test_images/test1.jpg'
    #inImg = mpimg.imread(inFile)


    #outImg = processFrame(inImg)

    #fig = plt.figure()
    #plt.subplot(1, 2, 1)
    #plt.imshow(inImg)

    ##plt.subplot(1, 2, 2)
    ##plt.imshow(outImg, cmap='gray' )
    #plt.show()


