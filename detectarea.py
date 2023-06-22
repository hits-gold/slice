import cv2
import numpy as np

class DetectArea():
    def __init__(self, img):
        self.img = img

    def max_con(self, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mx = max([cv2.contourArea(con) for con in contours])
        # for con in contours:
        #     if cv2.contourArea(con) == mx:
        #         an = con
        #         return an
        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        return sorted_contours[0]
    
    def resizing(self):
        img1 = self.img.copy()
        # img = cv2.bilateralFilter(self.img, -1, 50, 10)
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, sharpening_mask2)

        _, thresh1 = cv2.threshold(img,np.mean(img)+np.std(img), 255, 0, cv2.THRESH_BINARY)

        an1 = self.max_con(thresh1)
    
        if cv2.contourArea(an1) < 150000:
            img = cv2.bitwise_not(img)
            _, thresh1 = cv2.threshold(img,np.mean(img)+np.std(img), 255, 0, cv2.THRESH_BINARY)
            an1 = self.max_con(thresh1)

        rect = cv2.minAreaRect(an1)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width = int(rect[1][0])
        height = int(rect[1][1])
        
        src_pts = box.astype("float32")

        dst_pts = np.array([[0, 0],
                            [height-1, 0],
                            [height-1, width-1],
                            [0, width-1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img1, M, (height, width))


        if np.abs(box[0][0] - box[1][0]) < np.abs(box[0][0] - box[3][0]):
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE) 

        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        
        warped = cv2.resize(warped, (950, 500))
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

        return warped
