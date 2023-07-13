import cv2
import numpy as np

class DetectArea():
    def __init__(self, img):
        self.img = img

    def max_con(self, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        return sorted_contours[0]
    
    def co_sort(self, box):
        result = []
        for b in box:
            result.append([b[0], b[1]])
        result.sort()

        if result[0][1] > result[1][1]:
            a = result[0]
            b = result[1]
            result[0] = b
            result[1] = a
        if result[2][1] > result[3][1]:
            a = result[2]
            b = result[3]
            result[2] = b
            result[3] = a
            
        result = np.array(result)
        return result    
    
    def resizing(self):
        img1 = self.img.copy()
        img = cv2.bilateralFilter(self.img, -1, 50, 10)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, sharpening_mask2)

        _, thresh1 = cv2.threshold(img,np.mean(img)+np.std(img), 255, 0, cv2.THRESH_BINARY)

        an1 = self.max_con(thresh1)
    
        if cv2.contourArea(an1) < 150000:
            _, thresh1 = cv2.threshold(img,np.mean(img)-np.std(img), 255, 0, cv2.THRESH_BINARY)
            thresh1 = 255 - thresh1 
            an1 = self.max_con(thresh1)

        peri = cv2.arcLength(an1, True)
        approx = cv2.approxPolyDP(an1, 0.02*peri, True)

        if len(approx)==4:
            box = approx
            box = np.squeeze(box)
            box = self.co_sort(box)

        else:
            rect = cv2.minAreaRect(approx)        
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = self.co_sort(box)
        
        src_pts = box.astype("float32")

        if src_pts[0][0] - src_pts[2][0] > src_pts[0][1] - src_pts[1][1]:

            dst_pts = np.array([[0, 0],
                                [0, 950], 
                                [500, 0], 
                                [500, 950]
                                ], dtype="float32") 
        
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img1, M, (500, 950))
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        else:
            dst_pts = np.array([[0, 0],
                                [0, 500], 
                                [950, 0], 
                                [950, 500]
                                ], dtype="float32") 

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img1, M, (950, 500))
        
        warped = cv2.resize(warped, (950, 500))
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

        return warped