"""
Develop a Graphical User Interface, using any graphics library, with the following functions (use buttons):
    - Load and display a stereo pair of images (any two stereo images),
    - Use mouse clicks to select and match at least 10 pixels in the two images (manual matching)
    - Calculate the fundamental matrix F using the matched pixels.
    - Then, whenever the user clicks on any pixel, either left or right image, the corresponding epipolar line is drawn.

- Calculate your fundamental matrix either using your own project 02 function or by using an external one 
   from  OpenCV or any other source.

- Create a "Pixel-Matching" mode where:
     * First, the user can click on any pixel on either image, like in project 02. Then, a cross should appear at the clicked pixel's
        coordinates (x, y).

     * Then, the corresponding epipolar line is drawn on the other image, like in project 02,

     * Your program should perform the matching using two methods. (1) search in a bounding box. defined by the two vertices 
       (x-50, y-50) and (x+50, y+50), completely ignoring the epipolar constraint. (2) search along the epipolar line, within -50 pixels 
       +50 pixels, for the best matching pixel. In both methods, the pixel with the highest ZNCC score (you can use another correlation 
       function if you want) should be selected as the best match.  The 2 matching pixels should be drawn as crosses, i.e., the matched
       pixel without epipolar constraint and the matching pixel with the constraint. Use different colors.
"""

import tkinter as tk
from tkinter import messagebox
import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import random
from skimage.draw import line
from skimage.feature import match_template

class Helper(object):
    """
    The fundamental matrix describes the relationship between two images which have similar feature points.
    It can be calculated using at least 8 points, therefore the 8 point algorithm
    """
    @staticmethod
    def calc_FundamentalMatrixFromPoints(leftPoints,rightPoints):
        A = []
        for i in range(0,min(len(leftPoints), len(rightPoints))):
            x,y = leftPoints[i][0], leftPoints[i][1]
            x_, y_ = rightPoints[i][0], rightPoints[i][1]
            A.append((x*x_, x*y_, x, y*x_, y*y_, y, x_, y_, 1))
        A = np.matrix(A)

        _, S, V = np.linalg.svd(A)
        minS = min(S)
        F = np.matrix(np.reshape(V[np.where(S==minS)],(3,3)))

        return F
    
    """
    Calculate epipolar line for a point on the other image, using the already calculated fundamental matrix.
    We calculate this by multiplying the Point (x,y) with the Fundamental Matrix. 
    Then solve for a line using the obtained solution.
    """
    @staticmethod
    def calc_EpipolarLineCoords(x, y, isLeftImage, F, imageW):
        if(not isLeftImage): F = F.T
        point = np.matrix(np.reshape(np.array([x,y,1]),(3,1))) 
        solution = F * point
        a, b, c = float(solution[0]), float(solution[1]), float(solution[2])
        x1, x2 = 0, imageW
        if (b!=0):
            y1 = int(-((c/b) + ((a*x1)/b)))
            y2 = int(-((c/b) + ((a*x2)/b)))
        else:
            y1 = y2 = 0
        if isLeftImage:
            x1 += imageW
            x2 += imageW
        return x1, y1, x2, y2

    """
    Simple correlation between two patches of an image indicate how similar they are. This is not ZNCC.
    """
    @staticmethod
    def correlation(patch1, patch2):
        product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
        stds = patch1.std() * patch2.std()
        if stds == 0:
            return 0
        else:
            product /= stds
            return product

class MainApplication(tk.Frame):
    PRECALCULATE_FUNDAMENTAL_MATRIX = True
    WINDOW_SIZE = 30

    leftImageName = "im1.jpeg"
    rightImageName = "im2.jpeg"
    imageHeight = 480
    imageWidth = 640

    #3.19e-05 -5.66e-05 -0.024736 
    #6.59e-05 -3.6e-06 -0.0329196 
    #0.0097647 0.0362219 1.0 

    #We define a fixed fundamental matrix so that we don't have click points each time
    fixedFundamentalMatrix = np.array([[4.021368e-7, 5.841629e-5,  -1.727590e-2], [ -6.705804e-5, -7.098293e-7, 4.999593e-2 ], [1.981079e-2,  -5.404245e-2,  9.969398e-1]])
    colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "white", "black", "cyan2", "gray", "thistle3"]
    fundamentalMatrix = []

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        root.title("Fundamental Matrix Calculation Mode")

        #Use PIL library to put image fon canvas with the appropriate measurements
        self.canvas = tk.Canvas(root, cursor="cross", width = self.imageWidth*2, height = self.imageHeight)

        self.leftImage = ImageTk.PhotoImage(image = Image.open(self.leftImageName).resize((self.imageWidth, self.imageHeight), Image.ANTIALIAS))
        self.rightImage = ImageTk.PhotoImage(image = Image.open(self.rightImageName).resize((self.imageWidth, self.imageHeight), Image.ANTIALIAS))
        self.canvas.create_image(0, 0, image=self.leftImage, anchor=tk.NW)
        self.canvas.create_image(self.imageWidth, 0, image=self.rightImage, anchor=tk.NW)

        #Also save an OpenCV instance so that we can perform calculations on this
        self.leftImageOCV = cv2.resize(cv2.cvtColor(cv2.imread(self.leftImageName), cv2.COLOR_BGR2RGB), (self.imageWidth, self.imageHeight))
        self.rightImageOCV = cv2.resize(cv2.cvtColor(cv2.imread(self.rightImageName), cv2.COLOR_BGR2RGB), (self.imageWidth, self.imageHeight))

        #onClickCanvas is called after clicking on the canvas
        self.canvas.bind("<ButtonPress-1>", self.onClickCanvas)
        self.canvas.pack()

        self.button_calcFunMatrix = tk.Button(text="Calculate Fundamental Matrix", width=20, height=5, fg="blue", command=self.onClickFunMatrixButton)
        self.button_calcFunMatrix.pack(side=tk.BOTTOM)

        #Save clicks on the left image and right image
        self.imageClickCoords = {"left":[], "right":[]}
        #Exit Fundamental Matrix Mode and go directly to Epipolar Line Mode
        if(self.PRECALCULATE_FUNDAMENTAL_MATRIX == True):
            #w, h = self.imageWidth, self.imageHeight
            self.fundamentalMatrix = self.fixedFundamentalMatrix
            self.canvas.delete("oval")
            self.button_calcFunMatrix["state"] = "disabled"
            root.title("Epipolar Line Mode")


    def onClickCanvas(self, e):

        #Fundamental Matrix Mode, saves coordinates of clicked points on left and right images
        if self.fundamentalMatrix == []: 
            x, y = e.x, e.y
            self.canvas.create_oval(x-8,y-8,x+8,y+8,fill='blue', tags = "oval")
            isLeftImage = (x <= self.imageWidth) 
            x = x if isLeftImage else x-self.imageWidth

            print(x,y)

            if isLeftImage: self.imageClickCoords["left"].append([x,y])
            else: self.imageClickCoords["right"].append([x,y])
        
        #Epipolar Line Mode
        else:
            #Register click and select point (x,y) on the image, color it uniquely
            x, y = e.x, e.y
            fillColor = random.choice(self.colors)
            if (len(self.colors)>2): self.colors.remove(fillColor)
            self.canvas.create_oval(x-8,y-8,x+8,y+8,fill=fillColor)
            
            #Create epipolar line from selected point (x,y) using Helper function described above. Then draw the line on canvas.
            isLeftImage = (x <= self.imageWidth) 
            x = x if isLeftImage else x-self.imageWidth
            x1, y1, x2, y2 = Helper.calc_EpipolarLineCoords(x,y,isLeftImage,self.fundamentalMatrix, self.imageWidth)
            self.canvas.create_line(x1, y1, x2, y2, fill = fillColor, width = 3)
            
            #Using scipy library to extract all points on the line
            pointsOnLine = list(zip(*line(*(x1,y1), *(x2,y2))))
            
            """
            Capture window around the selected query point according to the pre-defined window size. 
            This window is filterWindowQuery, it will be used to make comparisons on the other image
            """
            filterWindowQuery = None
            if(isLeftImage): filterWindowQuery = self.leftImageOCV[y-self.WINDOW_SIZE:y+self.WINDOW_SIZE, x-self.WINDOW_SIZE:x+self.WINDOW_SIZE]
            else: filterWindowQuery = self.rightImageOCV[y-self.WINDOW_SIZE:y+self.WINDOW_SIZE, x-self.WINDOW_SIZE:x+self.WINDOW_SIZE]

            """
            The epipolar line formed on the other image is over-extended. So we limit it according to our image height.
            We then iterate through each point, and create each filterWindow in a similar fashion as above.
            These windows is filterWindowLine
            We immediately compare the current filterWindowLine with filterWindowQuery and establish a correlation score
            This correlation score is stored in an array allFilterPointCoords along with the correlation score
            """
            allFilterPointCoords = []
            for x_,y_ in pointsOnLine:
                if(self.imageHeight > y_ > 0): 
                    if(isLeftImage):
                        x_-=self.imageWidth #locally in the image so start from 0
                        filterWindowLine = self.rightImageOCV[y_-self.WINDOW_SIZE:y_+self.WINDOW_SIZE, x_-self.WINDOW_SIZE:x_+self.WINDOW_SIZE]
                    else: #isRightImage
                        filterWindowLine = self.leftImageOCV[y_-self.WINDOW_SIZE:y_+self.WINDOW_SIZE, x_-self.WINDOW_SIZE:x_+self.WINDOW_SIZE]

                    if(filterWindowLine.shape[0] == self.WINDOW_SIZE*2 and filterWindowLine.shape[1] == self.WINDOW_SIZE*2):
                        #print(x_,y_,self.leftImageOCV.shape,filterWindowLine.shape, filterWindowQuery.shape)
                        allFilterPointCoords.append(((x_, y_), Helper.correlation(filterWindowQuery, filterWindowLine)))

            """
            We then sort the allFilterPointCoords array with respect to the correlation score
            After arriving at the highest scored point (filter, actually), we plot that point on the other image
            Therefore, we have the most likely similar point on the epipolar line using correlation score
            """
            allFilterPointCoords = sorted(allFilterPointCoords, key = lambda x: -x[1])
            max_x, max_y = allFilterPointCoords[0][0]
            if (isLeftImage):
                max_x += self.imageWidth
            self.canvas.create_oval(max_x-8,max_y-8,max_x+8,max_y+8,fill=fillColor)
            #print(allFilterPointCoords)

            """
            For global search, we use an in-built OpenCV method which helps match a filter to a main image. 
            It iterates through the image in a sliding fashion.
            The correlation score chosen here is ZNCC.
            We then plot the box of the sliding window with the highest ZNCC score
            """
            if(isLeftImage): res = cv2.matchTemplate(self.rightImageOCV,filterWindowQuery,cv2.TM_CCORR_NORMED)
            else: res = cv2.matchTemplate(self.leftImageOCV,filterWindowQuery,cv2.TM_CCORR_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] + self.WINDOW_SIZE*2, top_left[1] + self.WINDOW_SIZE*2)
            if (not isLeftImage):
                self.canvas.create_rectangle(top_left[0],top_left[1],bottom_right[0],bottom_right[1],fill="", outline=fillColor, width = 2)
            else:
                self.canvas.create_rectangle(top_left[0]+self.imageWidth,top_left[1],bottom_right[0]+self.imageWidth,bottom_right[1],fill="", outline=fillColor, width = 2)

            
            if isLeftImage:
                p1x, p1y = x,y
                p2x, p2y = allFilterPointCoords[0][0]
                print(p1x, p1y, p2x, p2y)
                M1, M2, ThreeDResult = self.ThreeDReconstruction(p1x, p1y, p2x, p2y)

                print(M1)
                print(M2)
                print(ThreeDResult[0][0])

                messagebox.showinfo("FM","3D Reconstruction\n"
                +'\n'+"M1:"+'\n'
                +'\n'.join([''.join(['{:3} '.format(round(item,2)) for item in row]) for row in M1.tolist()])
                +'\n'+"M2:"+'\n'
                +'\n'.join([''.join(['{:3} '.format(round(item,2)) for item in row]) for row in M2.tolist()])
                +'\n'+"3D Points:"'\n'
                +'\n'.join([''.join(['{:3} '.format(round(item,7)) for item in ThreeDResult[0][0]])]))

    def onClickFunMatrixButton(self):
        #print(self.fundamentalMatrix , self.fixedFundamentalMatrix)
        if (self.fundamentalMatrix != self.fixedFundamentalMatrix):
            lenLeftPoints = len(self.imageClickCoords["left"])
            lenRightPoints = len(self.imageClickCoords["right"])
            if (lenLeftPoints == 0 or lenLeftPoints != lenRightPoints or lenLeftPoints < 8):
                messagebox.showwarning("Points error","No. of points in both images must be equal and greater than 8")
                return
            
            F = Helper.calc_FundamentalMatrixFromPoints(self.imageClickCoords["left"], self.imageClickCoords["right"])

            #OpenCV function for Fundamental Matrix gives better results
            F = cv2.findFundamentalMat(np.array(self.imageClickCoords["left"]).astype(float), np.array(self.imageClickCoords["right"]).astype(float), cv2.FM_8POINT)[0]

            messagebox.showinfo("FM","Fundamental Matrix\n"
                +"No. of points: "+str(lenLeftPoints)+"\n"
                +'\n'.join([''.join(['{:3} '.format(round(item,7)) for item in row]) for row in F.tolist()]))

        self.fundamentalMatrix = F

        self.canvas.delete("oval")
        self.button_calcFunMatrix["state"] = "disabled"
        root.title("Epipolar Line Mode")

    def ThreeDReconstruction(self, p1x, p1y, p2x, p2y):
        ThreeDpoints = np.loadtxt("3Dpoints.txt", usecols=range(0,3), dtype=np.float32).tolist()
        TwoDpointsIm1 = np.loadtxt("2Dpoints im1.txt", usecols=range(0,2), dtype=np.float32).tolist()
        TwoDpointsIm2 = np.loadtxt("2Dpoints im2.txt", usecols=range(0,2), dtype=np.float32).tolist()

        #Projection Matrix for projection of 3D points on 2D points
        A = []
        for ((X,Y,Z),(u,v)) in list(zip(ThreeDpoints, TwoDpointsIm1)): #Formulate A as required in equation
            A.append([-X,-Y,-Z,-1,0,0,0,0,u*X,u*Y,u*Z,u])
            A.append([0,0,0,0,-X,-Y,-Z,-1,v*X,v*Y,v*Z,v])
        A = np.matrix(A)
        _,_,V = np.linalg.svd(A) #Singular value decomposition to solve linear equations
        M1 = V[-1].reshape(3,4) #Smallest eigen value is the last one, which results in our Matrix M

        A = []
        for ((X,Y,Z),(u,v)) in list(zip(ThreeDpoints, TwoDpointsIm2)): #Formulate A as required in equation
            A.append([-X,-Y,-Z,-1,0,0,0,0,u*X,u*Y,u*Z,u])
            A.append([0,0,0,0,-X,-Y,-Z,-1,v*X,v*Y,v*Z,v])
        A = np.matrix(A)
        _,_,V = np.linalg.svd(A) #Singular value decomposition to solve linear equations
        M2 = V[-1].reshape(3,4) #Smallest eigen value is the last one, which results in our Matrix M

        t = cv2.triangulatePoints(M1,M2,np.array([p1x, p1y],dtype=np.float),np.array([p2x, p2y],dtype=np.float)).transpose()
        ans = cv2.convertPointsFromHomogeneous(t).tolist()
        return M1,M2,ans

if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()