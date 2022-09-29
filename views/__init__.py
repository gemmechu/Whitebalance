from PIL import Image
from flask import Blueprint, render_template, request, jsonify
from imutils.perspective import four_point_transform
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from sklearn.metrics import mean_squared_error
import itertools
import matplotlib
matplotlib.use('Agg')
# from torch_mtcnn import detect_faces

# from util import is_same, ModelLoaded

base = Blueprint('base', __name__)
THRESHOLD = 1.2

'''
1. Detect the colorchecker passport
2. Get the color value for each square - do Bilinear interpolation
3. Apply whitebalancing
4. Evaluate the whitebalanced result with the other square color value
'''

def detect_colorchecker(src_img):
    '''
    finds reference image in the source image and draw bbx
    '''
    reference_img = temp = cv2.imread("card_corrected.jpg")
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY) 
    H, W = temp.shape
    src2 = src_img.copy()
    src2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    method = cv2.TM_CCOEFF
    result = cv2.matchTemplate(src2, temp, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    location = max_loc
    bottom_right = (location[0] + W, location[1] + H)
    return location,bottom_right
def get_all_color(color_card_img, ref_img):
    color = [["white", "gray_0","gray_1","gray_2","gray_3","black"],["red", "green","yellow","light_blue","purple","pink"],
    ["dark_brown", "light_brown","dark_blue","dark_pink","light_cream","dark_cream"]]
    reference_color_location = {}
    card_parameter = {"VSpace": 20,"HSpace": 10,"Width": 140,"Height1": 300,"Height2": 140,"Rows": 3,"Columns": 6,"HStart": 17,"VStart": 17}


    h_ = color_card_img.shape[0]/ref_img.shape[0]
    w_ = color_card_img.shape[1]/ref_img.shape[1]

    VSpace, Height1, Height2, VStart = h_ * np.array([card_parameter["VSpace"], card_parameter["Height1"], card_parameter["Height2"], card_parameter["VStart"]])
    HSpace, Width, HStart = w_ * np.array([card_parameter["HSpace"], card_parameter["Width"], card_parameter["HStart"]])

    for row in range(len(color)):
        for col in range(len(color[row])):
            h = Height1
            if row >0:
                h = Height2
            x0 = HStart + col * Width 
            y0 = VStart + row * h
            # center = ((x0 +Width//2), ((y0+h//2)))
            if col>0:
                x0 = HStart + col * Width + col * HSpace
            if row >0:
                y0 = VStart + Height1 + row * VSpace + ((row-1) * h)
            
            center = (int(x0 + Width//2), int((y0+h//2)))
            # print(center)
            color_name = color[row][col]
            reference_color_location[color_name] = center
           
    return reference_color_location
def crop(src_img, location, bottom_right):
    x,y = location
    w,h = bottom_right[0]-location[0], bottom_right[1]- location[1]
    crop_img = src_img[y:y+h, x:x+w]

    return crop_img
def warp_perspective(image, template):
    maxFeatures=500
    keepPercent=0.2
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
    matches = sorted(matches, key=lambda x:x.distance)
    matches = sorted(matches, key=lambda x:x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt
        # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    return aligned
def get_whitebalance(src_img,src_gray):
    ref_gray = [150,150,150]
    r_, g_, b_ = ref_gray/src_gray

    R_ = cv2.multiply(src_img[...,0],r_)
    G_ = cv2.multiply(src_img[...,1],g_)
    B_ = cv2.multiply(src_img[...,2],b_)
    scaled_img = np.dstack((R_,G_,B_))
 
    return scaled_img
def evaluate(color_card_whitebalanced, color_locations): 
    
    GT_color = {'white': (255, 255, 255), 'gray_0': (179, 179, 179), 'gray_1': (150, 150, 150), 'gray_2': (119, 119, 119), 'gray_3': (89, 89, 89), 'black': (0, 0, 0), 
    'red': (238, 31, 36), 'green': (103, 188, 69), 'yellow': (255, 242, 0), 'light_blue': (80, 169, 50), 'purple': (51, 23, 0), 'pink': (237, 0, 140), 
    'dark_brown': (173, 50, 58), 'light_brown': (210, 126, 41), 'dark_blue': (9, 137, 164), 'dark_pink': (189, 88, 150), 'light_cream': (201, 154, 138), 'dark_cream': (184, 147, 125)}
    loss = {}
    # color_card_whitebalanced = cv2.cvtColor(color_card_whitebalanced, cv2.COLOR_BGR2RGB)
    for color in GT_color:
        GT= np.array(GT_color[color])
        loc = color_locations[color]
        pred = np.array(color_card_whitebalanced[loc[1], loc[0]])
        loss[color] = int(abs(np.mean(GT-pred)))
    return loss
def draw_circle(card, centers):
    for center in centers:
        cv2.circle(card,centers[center],20,(0,0,200),5)
    return card
def visualize(color_loss):
    try:
        plt.style.use('_mpl-gallery')
        row1 = dict(itertools.islice(color_loss.items(), 6))
        row2 = dict(itertools.islice(color_loss.items(), 6, 12))
        row3 = dict(itertools.islice(color_loss.items(), 12,18))
        datas = [row1,row2,row3]
        # make data
        
        for i in range(3):
            data = datas[i]
            np.random.seed(3)
            x = list(data.keys())
            y = list(data.values())
            fig, ax = plt.subplots()
            fig.set_size_inches(3, 6, forward=True)
            ax.stem(x, y)
            out = "static/row"+str(i+1)+".jpg"
            plt.savefig(out,dpi=fig.dpi,bbox_inches='tight')
            fig.clear()
    except:
        print("error in visualization")

def main(img_path):
    try:
        pil_img = np.array(img_path)
        src_img = cv2.cvtColor(pil_img, cv2.COLOR_RGB2BGR)
        # src_img = cv2.imread(img_path) 
        # cv2.imread(img_path)
        location, bottom_right = detect_colorchecker(src_img)
        #draw
        bbx = cv2.rectangle(src_img.copy(), location,bottom_right, 255, 15)
        cv2.imwrite("static/bbx.jpg", bbx)
        ##crop and warp it to a reference color checker
        color_card = crop(src_img, location, bottom_right)
        reference_color_checker = cv2.imread("card_corrected.jpg")
        warped_color_card = warp_perspective(image=color_card, template=reference_color_checker )

        #Get all the colorchart locations
        
        color_locations = get_all_color(warped_color_card, reference_color_checker)
        card_circled = draw_circle(warped_color_card, color_locations)
        cv2.imwrite("static/card_circled.jpg", card_circled)
        #Whitebalance my src image
        gray_loc = color_locations["gray_1"]
        
        gray_1 = warped_color_card[gray_loc[1],gray_loc[0]]
        img_whitebalanced = get_whitebalance(src_img, gray_1)

        #Evaluate my other colors
        color_card_whitebalanced = get_whitebalance(warped_color_card, gray_1)
        cv2.imwrite("static/card_corrected.jpg", color_card_whitebalanced)

        color_loss = evaluate(color_card_whitebalanced, color_locations)
        # print(color_locations)
        # print(color_loss)
        # img = draw_circle(color_card_whitebalanced, color_locations)

        ###SAVE IMG ###
        visualize(color_loss)
        cv2.imwrite("static/output.jpg", img_whitebalanced)
        return img_whitebalanced
    except:
        print("error")
# result = main()
def is_same(imgPerson, imgA, THRESHOLD):
    return 12
@base.route('/')
def index():
    return render_template('index.html')


@base.route('/predict', methods=['post'])
def predict():
    files = request.files
    inputImage = Image.open(files.get('inputImage')).convert('RGB')
    result = main(inputImage)
    # bbx.src =  "../static/bbx.jpg";
    #             card_perspective.src =  "../static/card_perspective.jpg";
    #             card_circled.src =  "../static/card_circled.jpg";
    #             row1.src =  "../static/row1.jpg";
    #             row2.src =  "../static/row2.jpg";
    #             row3.src =  "../static/row3.jpg";
    return jsonify(output="../static/output.jpg")
