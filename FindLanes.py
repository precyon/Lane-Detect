import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

    # RGB thresholds
    grayThres = (30, 255)
    gray = 0.5*img[:,:,0] + 0.4*img[:,:,1] + 0.1*img[:,:,2]
    grayBin = (gray > grayThres[0]) & (gray <= grayThres[1])

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


    return grayBin & (gmxBin | sBin)
    #return gray

def computePerpectiveTransforms():

    srcPts = np.array([
        [703, 461],  # top right
        [1000,650], # bottom right
        [308, 650],  # bottom left
        [580, 461]  # top left
    ], dtype=np.float32)

    dstPts = np.array([
        [1000, 0  ], # top right
        [1000, 700], # bottom right
        [300 , 700],  # bottom left
        [300 , 0  ]   # top left
        ], dtype=np.float32)


    #plt.imshow(inImg)
    #plt.plot(srcPts[:,0], srcPts[:,1], 'r', linewidth='4')
    #plt.plot(dstPts[:,0], dstPts[:,1], 'b', linewidth='4')
    #plt.show()
    M    = cv2.getPerspectiveTransform(srcPts, dstPts)
    Minv = cv2.getPerspectiveTransform(dstPts, srcPts)

    return M, Minv

def changePerpective(img, M):

    changed = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR)

    return changed

def processFrame(img):

    return thresholdFrame(img)


if __name__=='__main__':
    calDataFile = 'calib_data.npz'
    calImageFiles = 'camera_cal/calibration*.jpg'

    #if os.path.isfile(calDataFile):
    #    print('Loading camera calibration ...')
    #    loaded = np.load(calDataFile)
    #    mtx, dist = loaded['mtx'], loaded['dist']
    #else:
    #    print('Computing camera calibration ...')
    #    mtx, dist = setupAndCalib(calImageFiles)
    #    np.savez(calDataFile, mtx=mtx, dist=dist)


    #imgStack, _ = readFolderToStack()
    #outStack = [processFrame(img) for img in imgStack]

    #displayImagelist(outStack, cols = 4)

    #compareImageList(imgStack, outStack)


    inFile = 'test_images/straight_lines1.jpg'
    inImg = mpimg.imread(inFile)
    M, Minv = computePerpectiveTransforms()

    plt.imshow(changePerpective(inImg, M))
    plt.show()
    #outImg = processFrame(inImg)

    #fig = plt.figure()
    #plt.subplot(1, 2, 1)
    #plt.imshow(inImg)

    #plt.subplot(1, 2, 2)
    #plt.imshow(outImg, cmap='gray' )
    #plt.show()

    #inFile = 'project_video.mp4'
    #outFile = 'project_video_out.mp4'
    #videoIn = VideoFileClip(inFile)
    #videoOut = videoIn.fl_image(process_frame)
    #videoOut.write_videofile(outFile, audio=False)





