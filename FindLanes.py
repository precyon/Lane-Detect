import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from scipy.signal import find_peaks_cwt as findPeaks


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
    nx, ny = 9, 6
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


def thresholdFrame(img):

    # Gray thresholds
    grayThres = (30, 255)
    gray = 0.5*img[:,:,0] + 0.4*img[:,:,1] + 0.1*img[:,:,2]
    grayBin = (gray > grayThres[0]) & (gray <= grayThres[1])

    # RGB thresholds


    # HSV thresholds

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

    sobely = np.absolute(cv2.Sobel(sobelin, cv2.CV_64F, 0, 1, ksize=sobelSize))
    sobely = sobely/np.max(sobely)
    #gmyBin


    # Calculate the gradient magnitude
    gradMagThres = (0.25, 1)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    gradmag = gradmag/np.max(gradmag)
    gmBin = (gradmag > gradMagThres[0]) & (gradmag <= gradMagThres[1])

    # Calculate the gradient direction
    gradDirThres = (0.8, 1.2)
    graddir = np.arctan2(sobely, sobelx)
    gdBin = (graddir > gradDirThres[0]) & (graddir <= gradDirThres[1])

    # Calculate the laplacian of gaussian
    gaussianKernelShape = (5,5)
    laplacianKernelSize = 21
    laplacianThresh = 0.1
    logImg = cv2.GaussianBlur(s, gaussianKernelShape, 0)
    logImg = cv2.Laplacian(logImg, cv2.CV_64F, ksize=laplacianKernelSize)
    logBin = (logImg < laplacianThresh*np.min(logImg))


    return ((grayBin & (gmxBin | sBin))*255).astype(np.uint8)

def computePerpectiveTransforms():

    #srcPts = np.array([
    #    [703, 461],  # top right
    #    [1000,650], # bottom right
    #    [308, 650],  # bottom left
    #    [580, 461]  # top left
    #], dtype=np.float32)

    #dstPts = np.array([
    #    [1000, 0  ], # top right
    #    [1000, 700], # bottom right
    #    [300 , 700],  # bottom left
    #    [300 , 0  ]   # top left
    #    ], dtype=np.float32)


    M    = cv2.getPerspectiveTransform(cache['persrc'], cache['perdst'])
    Minv = cv2.getPerspectiveTransform(cache['perdst'], cache['persrc'])

    return M, Minv

def changePerpective(img, M, visualize=False):

    changed = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    if(visualize):
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img)
        colors = ['ro', 'go', 'bo', 'wo']
        pts = cache['persrc']
        for i in range(4):
            plt.plot(pts[i,0], pts[i,1], colors[i])


        plt.subplot(1,2,2)
        plt.imshow(changed)
        pts = cache['perdst']
        for i in range(4):
            plt.plot(pts[i,0], pts[i,1], colors[i])

        plt.show()

    return changed

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
    winWdith  = margin

    pltImg = np.zeros(img.shape) if visualize else None

    yLeft, xLeft, yRight, xRight = [], [], [], []
    for i in range(nWin)[::-1]:

        # Read out pixels in the windows
        lWin = img[i*winHeight:(i+1)*winHeight, idxStartLeft-winWdith//2:idxStartLeft+winWdith//2]
        rWin = img[i*winHeight:(i+1)*winHeight, idxStartRight-winWdith//2:idxStartRight+winWdith//2]

        if visualize:
            cv2.rectangle(pltImg, (idxStartLeft-winWdith//2, i*winHeight),
                                  (idxStartLeft+winWdith//2, (i+1)*winHeight),
                                  255, 2)
            cv2.rectangle(pltImg, (idxStartRight-winWdith//2, i*winHeight),
                                  (idxStartRight+winWdith//2, (i+1)*winHeight),
                                  255, 2)

        # Get the non-zero pixels in the windows
        yLeftCur , xLeftCur  = lWin.nonzero()
        yRightCur, xRightCur = rWin.nonzero()

        # TODO: Reject outliers

        # Append to lane pixels
        yLeft.extend( yLeftCur  + winHeight*i)
        xLeft.extend( xLeftCur  + idxStartLeft  - winWdith//2)
        yRight.extend(yRightCur + winHeight*i)
        xRight.extend(xRightCur + idxStartRight - winWdith//2)

        if len(xLeftCur) > pixThres:
            idxStartLeft = np.int(np.median(xLeftCur)) + idxStartLeft - winWdith//2
        if len(xRightCur) > pixThres:
            idxStartRight = np.int(np.median(xRightCur)) + idxStartRight - winWdith//2

    #cv2.putText(pltImg,'%.2f'%(np.std(yRight)),(50,50),  cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

    return yLeft, xLeft, yRight, xRight, pltImg

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def currentCurvature(x, y):

    x = np.float64(x)*xm_per_pix
    y = np.float64(y)*ym_per_pix

    fitCoeffs = np.polyfit(y, x, 2)
    yEval = np.max(y)
    return ((1 + (2*fitCoeffs[0]*yEval + fitCoeffs[1])**2)**1.5) / np.absolute(2*fitCoeffs[0])


def linesFromPixels(yLeft, xLeft, yRight, xRight):

    leftLineCoeffs, rightLineCoeffs = None, None

    if len(yLeft) > 150 and np.std(yLeft) > 100:
        leftLineCoeffs  = np.polyfit(yLeft, xLeft, 2)

    if len(yRight) > 150 and np.std(yRight) > 100:
        rightLineCoeffs = np.polyfit(yRight, xRight, 2)

    return leftLineCoeffs, rightLineCoeffs


def processFrame(img):

    undistImg = cv2.undistort(img, cache['mtx'], cache['dist'], None, cache['mtx'])
    threshImg = thresholdFrame(undistImg)
    perspImg = changePerpective(threshImg, cache['per_m'])

    yLeft, xLeft, yRight, xRight, winOver = slidingLanePixelSearch(perspImg, visualize=True)
    leftLineCoeffs, rightLineCoeffs = linesFromPixels(yLeft, xLeft, yRight, xRight)

    if leftLineCoeffs is not None and rightLineCoeffs is not None:
        # Compute radii of curvature
        rLeft, rRight = currentCurvature(xLeft, yLeft), currentCurvature(xRight, yRight)
        detected = True
    else:
        detected = False


    # Draw overlays
    #pltImg = np.zeros((*perspImg.shape, 3), dtype=np.uint8)
    #pltImg[yLeft, xLeft] = [255,0,0]
    #pltImg[yRight, xRight]=[0,0,255]
    #pltImg[:,:,1] = winOver
    #plt.imshow(pltImg)
    ## Generate lane lines
    diagImg = np.zeros((*perspImg.shape, 3), dtype=np.uint8)
    if detected:
        yLine = np.array(range(perspImg.shape[0]))
        xLineLeft = np.polyval(leftLineCoeffs, yLine)
        xLineRight = np.polyval(rightLineCoeffs, yLine)
        #plt.plot(xLineLeft, yLine, color='yellow')
        #plt.plot(xLineRight, yLine, color='yellow')#plt.show()

        ## Create the output annotated image
        markImg = np.zeros((*perspImg.shape, 3), dtype=np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        ptsLeft  = np.array([np.transpose(np.vstack([xLineLeft, yLine]))])
        ptsRight = np.array([np.flipud(np.transpose(np.vstack([xLineRight, yLine])))])
        pts = np.hstack((ptsLeft, ptsRight))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(markImg, np.int_([pts]), (0,255,0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        markImg = cv2.warpPerspective(markImg, cache['per_minv'], (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undistImg, 1, markImg, 0.3, 0)

        ## Create a lane detection diagnostic image
        diagImg[yLeft, xLeft] = [255,0,0]
        diagImg[yRight, xRight]=[0,0,255]
        diagImg[:,:,1] = winOver
        cv2.polylines(diagImg, np.int32([ptsLeft]), isClosed=False, color=[255, 255, 0], thickness=8)
        cv2.polylines(diagImg, np.int32([ptsRight]), isClosed=False, color=[255, 255, 0], thickness=8)
    else:
        result = undistImg

    diagnostic = True
    if diagnostic:

        imgTopLeft = undistImg
        imgTopRight = np.dstack((threshImg,threshImg,threshImg))
        imgBotLeft = diagImg
        imgBotRight = result
        result = np.vstack((
                np.hstack( (imgTopLeft, imgTopRight) ),
                np.hstack( (imgBotLeft, imgBotRight) )
                ))

    return result


if __name__=='__main__':
    calDataFile = 'calib_data.npz'
    calImageFiles = 'camera_cal/calibration*.jpg'

    #dist_pickle = {}
    #dist_pickle['mtx'] = mtx
    #dist_pickle['dist'] = dist
    #pickle.dump(dist_pickle, open('./camera_cal/calibration_pickle.p', 'wb'))

    #dist_pickle = pickle.load(open('./camera_cal/calibration_pickle.p', 'rb'))
    #mtx = dist_pickle['mtx']
    #dist = dist_pickle['dist']

    if os.path.isfile(calDataFile):
        print('Loading camera calibration ...')
        loaded = np.load(calDataFile)
        mtx, dist = loaded['mtx'], loaded['dist']
    else:
        print('Computing camera calibration ...')
        mtx, dist = setupAndCalib(calImageFiles)
        np.savez(calDataFile, mtx=mtx, dist=dist)


    cache = {'mtx': mtx,
             'dist': dist,
             'persrc': np.array([
                    [703, 461],  # top right
                    [1000,650], # bottom right
                    [308, 650],  # bottom left
                    [580, 461]  # top left
                             ], dtype=np.float32),
             'perdst': np.array([
                    [1000, 0  ], # top right
                    [1000, 700], # bottom right
                    [300 , 700],  # bottom left
                    [300 , 0  ]   # top left
                              ], dtype=np.float32)
             }


    cache['per_m'], cache['per_minv'] = computePerpectiveTransforms()

    #imgStack, _ = readFolderToStack()
    #outStack = [processFrame(img) for img in imgStack]
    #compareImageList(imgStack, outStack)

    #inFile = 'test_images/test1.jpg'
    #inImg = mpimg.imread(inFile)


    #outImg = processFrame(inImg)

    #fig = plt.figure()
    #plt.subplot(1, 2, 1)
    #plt.imshow(inImg)

    #plt.subplot(1, 2, 2)
    #plt.imshow(outImg, cmap='gray' )
    #plt.show()

    inFile = 'project_video.mp4'
    outFile = 'project_video_out.mp4'
    #videoIn = VideoFileClip(inFile).subclip(20, 26)
    videoIn = VideoFileClip(inFile)
    videoOut = videoIn.fl_image(processFrame)
    videoOut.write_videofile(outFile, audio=False)





