from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import messagebox
from tkinter import filedialog
import cv2
import imutils
from imutils import perspective
from imutils import contours
import numpy as np
from scipy.spatial import distance as dist

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

class GUI:
    def __init__(self,root):
        self.frame = Frame(root)
        self.frame1 = Frame(root)
        self.frameButton=Frame(root)
        self.frameButton1=Frame(root)
        self.frameButton2=Frame(root)
        self.frameButton3=Frame(root)
        self.frameButton4=Frame(root)

        # self.video_source=video_source

        # self.vid= MyVideoCapture(self.video_source)

        self.image = StringVar()
        self.tkvar=StringVar()
        self.size = DoubleVar()
        self.size.set(1)

        root.iconbitmap(r'12.ico')
        root.title('')
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        width = 750
        height = 500
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        root.geometry('%dx%d+%d+%d' % (width, height, x, y))
        root.resizable(0, 0)
        Label (text ="IMAGE PROCESSING", bg ="grey",fg="black",width="600",height="1",font=("calibri",20)).pack() 
        Label (text ="Original Image",fg="red",font=("tahoma",14)).place(x=120,y=55)
        Label (text ="Result",fg="blue",font=("tahoma",14)).place(x=520,y=55)
        Label (text ="compiled by : Haryanto (M07158031)",bg ="grey",
        	fg="red",width="750",height="2",font=("tahoma",9)).place(x=0,y=470)

        self.canvas = Canvas(self.frame, bg='white')
        self.canvas.grid(row=0, column=1)
        self.canvas1 = Canvas(self.frame1, width=350, height=250, bg='white')
        self.canvas1.grid(row=0, column=1)
        # self.openImage()
        self.makeButton()

        # self.frame.place(x=15,y=90)
        # self.frame1.place(x=380,y=90)
   
    def makeButton(self):
        Button(self.frameButton,relief='raised', text="Load Media", command=self.browseButton).grid(row=0, column=1)
        self.frameButton.place(x=75, y=400)
        Button(self.frameButton1,relief='raised', text="New Image", command=self.openCAM).grid(row=0, column=1)
        self.frameButton1.place(x=225, y=400)
        Button(self.frameButton2,relief='raised', text="Quit ?", command=self.exit).grid(row=0, column=1)
        self.frameButton2.place(x=685, y=440)
        Button(self.frameButton3,relief='raised', text="SizeMeasurment", command=self.sizeDetect).grid(row=0, column=1)
        self.frameButton3.place(x=400, y=400)
        Button(self.frameButton4,relief='raised', text="ObjectDetection", command=self.objectDetect).grid(row=0, column=1)
        self.frameButton4.place(x=550, y=400)

    def browseButton(self):
        try:
            tipeFile = (('image files', '*.jpg'), ('png files', '*.png'), ('all files', '*'))
            self.path = filedialog.askopenfilename(filetypes=tipeFile)
            global image
            image = cv2.imread(self.path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # height, width, no_channels = cv_img.shape
            image_pil = Image.fromarray(image_rgb)
            # image2 = cv2.resize(oriimg,(360,480))
            self.img = ImageTk.PhotoImage(image_pil)
            self.canvas.create_image(0,0,image=self.img, anchor='nw')
            self.frame.place(x=15,y=90)

        except:
            messagebox.showwarning("error","wrong format media, please check again")

    def openCAM(self):
    	print("in development")

        
    def sizeDetect(self):
        # global image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray1 = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(gray, 50, 200)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None
        # cv2.imshow('original',edged)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # self.img1 = ImageTk.PhotoImage(edged)
        # self.canvas1.create_image(0,0,image=self.img1, anchor='nw')
        # loop over the contours individually
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 100:
                continue

            # compute the rotated bounding box of the contour
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, inches)
            if pixelsPerMetric is None:
                pixelsPerMetric = dB / 3.5

            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}cm".format(dimB),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)
            cv2.putText(orig, "{:.1f}cm".format(dimA),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)

            # show the output image
            cv2.imshow("Image", orig)
            cv2.waitKey()

    def objectDetect(self):
        def get_output_layers(net):

            layer_names = net.getLayerNames()
            
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            return output_layers


        def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

            label = str(classes[class_id])

            color = COLORS[class_id]

            cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

            cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)          
        image1 = image

        Width = image1.shape[1]
        Height = image1.shape[0]
        scale = 0.00392

        classes = None


        with open("yolov3.txt", 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])


        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        cv2.imshow("object detection", image)
        cv2.waitKey()
        cv2.destroyAllWindows()

 
    def exit(self):
    	answer = messagebox.askquestion('', "Are you sure want to Quit ?" )
    	if answer == 'yes':
    		global page
    		root.destroy()
    	elif answer == 'no':  # 'no'
    		pass

root = Tk() 
GUI(root)
root.mainloop()