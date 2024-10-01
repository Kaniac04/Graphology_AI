# importing necessary libraries
import cv2
import json
import cv2
import numpy as np
import math
from scipy.signal import find_peaks
from skimage.measure import regionprops

# Initializing threshold globals
ANCHOR_POINT = 6000
MIDZONE_THRESHOLD = 15000
MIN_HANDWRITING_HEIGHT_PIXEL = 20

# Features 
BASELINE_ANGLE = 0.0
TOP_MARGIN = 0.0
LETTER_SIZE = 0.0
LINE_SPACING = 0.0
WORD_SPACING = 0.0
PEN_PRESSURE = 0.0
SLANT_ANGLE = 0.0
GARLANDS = {}

def bilateralFilter(
        image : cv2.typing.MatLike, 
        diameter : int) -> cv2.typing.MatLike:
    '''Applies Bilateral Filter on the image to remove Gaussian Noise.'''
    image = cv2.bilateralFilter(image, diameter, 50, 50)
    return image

def medianFilter(
        image : cv2.typing.MatLike, 
        diameter : int) -> cv2.typing.MatLike:
    '''Applies Median Filter on the image to remove Salt-Pepper Noise.'''
    image = cv2.medianBlur(image, diameter)
    return image

def threshold(
        image : cv2.typing.MatLike, 
        threshold : int) -> cv2.typing.MatLike:
    '''Applies the Binary Inverted Thresholding to clear the black pixel boundaries.'''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return image

def dilate(
        image : cv2.typing.MatLike, 
        kernal_size : int) -> cv2.typing.MatLike:
    '''Dilates the image and enlarges the white spaces in the foreground.'''
    kernel = np.ones(kernal_size, np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return image

def erode(
        image : cv2.typing.MatLike, 
        kernal_size : int) -> cv2.typing.MatLike:
    '''Erodes the image to reduce the black pixels. This highlights only the dark and major pixels in an image'''
    kernel = np.ones(kernal_size, np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return image

def contour_straightener(image : cv2.typing.MatLike) -> cv2.typing.MatLike:
    angle = 0.0
    angle_sum = 0.0
    contour_count = 0

    global BASELINE_ANGLE

    filtered = bilateralFilter(image, 3)
    thresh = threshold(filtered, 120)
    dilated = dilate(thresh, (5, 100))

    ctrs, hier = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, ctr in enumerate(ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        # Countour of a line will be a horizontal rectangle,
        # otherwise it wont be a countour.
        if h > w or h < MIN_HANDWRITING_HEIGHT_PIXEL:
            continue
        
        # Region of Interest
        roi = image[y:y+h, x:x+w]

        # Ignore any line which is very short, (less than one-third of the page width.)
        if w < image.shape[1]/2:
            roi = 255
            image[y:y+h, x:x+w] = roi
            continue

        # minAreaRect is necessary for straightening
        rect = cv2.minAreaRect(ctr)
        center = rect[0]
        angle = rect[2]

        # these two conditions have been taken because the minAreaRect sometimes
        # causes 180 degree rotated rect formation, because of which edge cases must be handled.
        if angle < -45.0:
            angle += 90.0
        if angle > 80.0 :
            angle -=90.0

        rot = cv2.getRotationMatrix2D(((x+w)/2, (y+h)/2), angle, 1)
        extract = cv2.warpAffine(roi, rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
       
        # image is overwritten with the straightened contour
        image[y:y+h, x:x+w] = extract
        
        # counting number of contours and summing up all the angles of baseline
        # for all the lines so that we can calculate the average baseline angle.
        angle_sum += angle
        contour_count += 1

    # mean angle of the contours (not lines) is found
    if contour_count == 0.0:
        mean_angle = angle_sum
    else:
        mean_angle = angle_sum / contour_count

    BASELINE_ANGLE = mean_angle
    return image

def HProjection(image : cv2.typing.MatLike) -> list:
    (h, w) = image.shape[:2]
    sumRows = []
    for j in range(h):
        row = image[j:j+1, 0:w]  
        sumRows.append(np.sum(row))
    return sumRows

def VProjection(image : cv2.typing.MatLike) -> list:
    (h, w) = image.shape[:2]
    sumCols = []
    for j in range(w):
        col = image[0:h, j:j+1]  # y1:y2, x1:x2
        sumCols.append(np.sum(col))
    return sumCols

def lines_extract(image : cv2.typing.MatLike) -> list:
    global LETTER_SIZE
    global LINE_SPACING
    global TOP_MARGIN

    filtered = bilateralFilter(image, 5)
    thresh = threshold(filtered, 160)
    hpList = HProjection(thresh)

    # Extracting Top Margin
    topMarginCount = 0
    for sum in hpList:
        if (sum <= 255):
            topMarginCount += 1
        else:
            break

    # extract the straightened contours from the image by looking at occurance of 0's in the horizontal projection.
    lineTop = 0
    lineBottom = 0
    spaceTop = 0
    spaceBottom = 0
    indexCount = 0
    setLineTop = True
    setSpaceTop = True
    includeNextSpace = True
    space_zero = []  # stores the amount of space between lines
    lines = []  # a 2D list storing the vertical start index and end index of each contour

    for i, sum in enumerate(hpList):
        # sum being 0 means blank space
        if (sum == 0):
            if (setSpaceTop):
                spaceTop = indexCount
                setSpaceTop = False  # spaceTop will be set once for each start of a space between lines
            indexCount += 1
            spaceBottom = indexCount
            if (i < len(hpList)-1):
                # if the next horizontal projectin is 0, keep on counting, it's still in blank space
                if (hpList[i+1] == 0):
                    continue
            # we are using this condition if the previous contour is very thin and possibly not a line
            if (includeNextSpace):
                space_zero.append(spaceBottom-spaceTop)
            else:
                if (len(space_zero) == 0):
                    previous = 0
                else:
                    previous = space_zero.pop()
                space_zero.append(previous + spaceBottom-lineTop)
            # next time we encounter 0, it's begining of another space so we set new spaceTop
            setSpaceTop = True

        # sum greater than 0 means contour
        if (sum > 0):
            if (setLineTop):
                lineTop = indexCount
                setLineTop = False  
            indexCount += 1
            lineBottom = indexCount
            if (i < len(hpList)-1):  
                if (hpList[i+1] > 0):
                    continue

                # if the line/contour is too thin <10 pixels (arbitrary) in height, we ignore it.
                # Also, we add the space following this and this contour itself to the previous space to form a bigger space: spaceBottom-lineTop.
                if (lineBottom-lineTop < 20):
                    includeNextSpace = False
                    setLineTop = True  # next time we encounter value > 0, it's begining of another line/contour so we set new lineTop
                    continue
            includeNextSpace = True
            lines.append([lineTop, lineBottom])
            setLineTop = True  

    
    # extract the very individual lines from the lines/contours we extracted above.
    fineLines = []  # a 2D list storing the horizontal start index and end index of each individual line
    for i, line in enumerate(lines):

        # 'anchor' will locate the horizontal indices where horizontal projection is > ANCHOR_POINT for uphill or < ANCHOR_POINT for downhill
        anchor = line[0]
        anchorPoints = []  
        upHill = True
        downHill = False
        # we put the region of interest of the horizontal projection of each contour here
        segment = hpList[line[0]:line[1]]

        for j, sum in enumerate(segment):
            if (upHill):
                if (sum < ANCHOR_POINT):
                    anchor += 1
                    continue
                anchorPoints.append(anchor)
                upHill = False
                downHill = True
            if (downHill):
                if (sum > ANCHOR_POINT):
                    anchor += 1
                    continue
                anchorPoints.append(anchor)
                downHill = False
                upHill = True

        if (len(anchorPoints) < 2):
            continue

        # len(anchorPoints) > 3 meaning contour composed of multiple lines
        lineTop = line[0]
        for x in range(1, len(anchorPoints)-1, 2):
            lineMid = (anchorPoints[x]+anchorPoints[x+1])/2
            lineBottom = lineMid
            # line having height of pixels <20 is considered defects, so we just ignore it
            # this is a weakness of the algorithm to extract lines (anchor value is ANCHOR_POINT, see for different values)
            if (lineBottom-lineTop < 20):
                continue
            fineLines.append([lineTop, lineBottom])
            lineTop = lineBottom
        if (line[1]-lineTop < 20):
            continue
        fineLines.append([lineTop, line[1]])

    # LINE SPACING and LETTER SIZE will be extracted here
    # We will count the total number of pixel rows containing upper and lower zones of the lines and add the space_zero/runs of 0's(excluding first and last of the list ) to it.
    # We will count the total number of pixel rows containing midzones of the lines for letter size.
    # For this, we set an arbitrary (yet suitable!) threshold MIDZONE_THRESHOLD = 15000 in horizontal projection to identify the midzone containing rows.
    # These two total numbers will be divided by number of lines (having at least one row>MIDZONE_THRESHOLD) to find average line spacing and average letter size.
    space_nonzero_row_count = 0
    midzone_row_count = 0
    lines_having_midzone_count = 0
    flag = False
    for i, line in enumerate(fineLines):
        segment = hpList[int(line[0]):int(line[1])]
        for j, sum in enumerate(segment):
            if (sum < MIDZONE_THRESHOLD):
                space_nonzero_row_count += 1
            else:
                midzone_row_count += 1
                flag = True

        # This line has contributed at least one count of pixel row of midzone
        if (flag):
            lines_having_midzone_count += 1
            flag = False

    if (lines_having_midzone_count == 0):
        lines_having_midzone_count = 1

    # excluding first and last entries: Top and Bottom margins
    total_space_row_count = space_nonzero_row_count + np.sum(space_zero[1:-1])
    # the number of spaces is 1 less than number of lines but total_space_row_count contains the top and bottom spaces of the line
    average_line_spacing = float(total_space_row_count) / lines_having_midzone_count
    average_letter_size = float(midzone_row_count) / lines_having_midzone_count
    # letter size is actually height of the letter and we are not considering width
    LETTER_SIZE = average_letter_size
    if (average_letter_size == 0):
        average_letter_size = 1
    # We can't just take the average_line_spacing as a feature directly. We must take the average_line_spacing relative to average_letter_size.
    # Let's take the ratio of average_line_spacing to average_letter_size as the LINE SPACING, which is perspective to average_letter_size.
    relative_line_spacing = average_line_spacing / average_letter_size
    LINE_SPACING = relative_line_spacing

    # Top marging is also taken relative to average letter size of the handwritting
    relative_top_margin = float(topMarginCount) / average_letter_size
    TOP_MARGIN = relative_top_margin

    return fineLines

def extract_words(image : cv2.typing.MatLike, lines : list):

    global LETTER_SIZE
    global WORD_SPACING

    filtered = bilateralFilter(image, 5)
    thresh = threshold(filtered, 180)

    # Width of the whole document is found once.
    width = thresh.shape[1]
    space_zero = []  # stores the amount of space between words
    words = []  # a 2D list storing the coordinates of each word: y1, y2, x1, x2

    # Isolated words or components will be extacted from each line by looking at occurance of 0's in its vertical projection.
    for i, line in enumerate(lines):
        extract = thresh[int(line[0]):int(line[1]), 0:width]  
        vp = VProjection(extract)
        wordStart = 0
        wordEnd = 0
        spaceStart = 0
        spaceEnd = 0
        indexCount = 0
        setWordStart = True
        setSpaceStart = True
        includeNextSpace = True
        spaces = []

        for j, sum in enumerate(vp):
            
            if (sum == 0):
                if (setSpaceStart):
                    spaceStart = indexCount
                    setSpaceStart = False
                indexCount += 1
                spaceEnd = indexCount
                if (j < len(vp)-1):  # this condition is necessary to avoid array index out of bound error
                    if (vp[j+1] == 0):
                        continue

                # we ignore spaces which is smaller than half the average letter size
                if ((spaceEnd-spaceStart) > int(LETTER_SIZE/2)):
                    spaces.append(spaceEnd-spaceStart)

                # next time we encounter 0, it's begining of another space so we set new spaceStart
                setSpaceStart = True

            if (sum > 0):
                if (setWordStart):
                    wordStart = indexCount
                    setWordStart = False  # wordStart will be set once for each start of a new word/component
                indexCount += 1
                wordEnd = indexCount
                if (j < len(vp)-1):  # this condition is necessary to avoid array index out of bound error
                    if (vp[j+1] > 0):
                        continue

                # append the coordinates of each word/component: y1, y2, x1, x2 in 'words'
                # we ignore the ones which has height smaller than half the average letter size
                # this will remove full stops and commas as an individual component
                count = 0
                for k in range(int(line[1]-line[0])):
                    row = thresh[int(line[0])+k:int(line[0])+k+1,
                                 wordStart:wordEnd]  # y1:y2, x1:x2
                    if (np.sum(row)):
                        count += 1
                if (count > int(LETTER_SIZE/2)):
                    words.append([line[0], line[1], wordStart, wordEnd])
                setWordStart = True

        space_zero.extend(spaces[1:-1])

    space_columns = np.sum(space_zero)
    space_count = len(space_zero)
    if (space_count == 0):
        space_count = 1
    average_word_spacing = float(space_columns) / space_count
    if LETTER_SIZE == 0.0:
        relative_word_spacing = average_word_spacing
    else:
        relative_word_spacing = average_word_spacing / LETTER_SIZE
    WORD_SPACING = relative_word_spacing
    return words

def extract_slant(image : cv2.typing.MatLike, words : list) -> None:
    global SLANT_ANGLE
    '''
	0.01 radian = 0.5729578 degree :: there was a bug yeilding inacurate value for 0.00 rads
	5 degree = 0.0872665 radian :: Hardly noticeable or a very little slant
	15 degree = 0.261799 radian :: Easily noticeable or average slant
	30 degree = 0.523599 radian :: Above average slant
	45 degree = 0.785398 radian :: Extreme slant
	'''
    # We are checking for 9 different values of angle
    radians = [-0.785398, -0.523599, -0.261799, -0.0872665,
             0.01, 0.0872665, 0.261799, 0.523599, 0.785398]
    
    # Corresponding index of the biggest value in s_function will be the index of the most likely angle in 'theta'
    s_function = [0.0] * 9
    count_ = [0]*9

    # apply bilateral filter
    filtered = bilateralFilter(image, 5)
    thresh = threshold(filtered, 180)
    for i, angle in enumerate(radians):
        s_temp = 0.0  # overall sum of the functions of all the columns of all the words!
        count = 0  # just counting the number of columns considered to contain a vertical stroke and thus contributing to s_temp
        for j, word in enumerate(words):
            original = thresh[int(word[0]):int(word[1]), int(word[2]):int(word[3])]  # y1:y2, x1:x2

            height = int(math.ceil((word[1]-word[0])))
            width = int(math.ceil((word[3]-word[2])))

            # the distance in pixel we will shift for affine transformation
            # it's divided by 2 because the uppermost point and the lowermost points are being equally shifted in opposite directions
            shift = math.ceil(math.tan(angle) * height) / 2

            # the amount of extra space we need to add to the original image to preserve information
            # this is adding more number of columns but the effect of this will be negligible
            pad_length = abs(int(shift))

            # create a new image that can perfectly hold the transformed and thus widened image
            blank_image = np.zeros((height, width+pad_length*2, 3), np.uint8)
            new_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)

            try:
                new_image[:, pad_length:width+pad_length] = original
            except ValueError:
                blank_image = np.zeros((height-1, width+pad_length*2, 3), np.uint8)
                new_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
                new_image[:, pad_length:width+pad_length] = original


            # points to consider for affine transformation
            (height, width) = new_image.shape[:2]
            x1 = width/2
            y1 = 0
            x2 = width/4
            y2 = height
            x3 = 3*width/4
            y3 = height

            pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3]])
            pts2 = np.float32([[x1+shift, y1], [x2-shift, y2], [x3-shift, y3]])
            M = cv2.getAffineTransform(pts1, pts2)
            deslanted = cv2.warpAffine(new_image, M, (width, height))

            
            vp = VProjection(deslanted)

            for k, sum in enumerate(vp):
                # the columns is empty
                if (sum == 0):
                    continue

                num_fgpixel = sum / 255

                if (num_fgpixel < int(height/3)):
                    continue

                # the column itself is extracted, and flattened for easy operation
                column = deslanted[0:height, k:k+1]
                column = column.flatten()

                for l, pixel in enumerate(column):
                    if (pixel == 0):
                        continue
                    break
                for m, pixel in enumerate(column[::-1]):
                    if (pixel == 0):
                        continue
                    break

                # the distance is found as delta_y
                delta_y = height - (l+m)
                h_sq = (float(num_fgpixel)/delta_y)**2
                h_wted = (h_sq * num_fgpixel) / height
                s_temp += h_wted

                count += 1

        s_function[i] = s_temp
        count_[i] = count

    # finding the largest value and corresponding index
    max_value = 0.0
    max_index = 4
    for index, value in enumerate(s_function):
        if (value > max_value):
            max_value = value
            max_index = index

    if (max_index == 0):
        angle = 45
        result = " : Extremely right slanted"
    elif (max_index == 1):
        angle = 30
        result = " : Above average right slanted"
    elif (max_index == 2):
        angle = 15
        result = " : Average right slanted"
    elif (max_index == 3):
        angle = 5
        result = " : A little right slanted"
    elif (max_index == 5):
        angle = -5
        result = " : A little left slanted"
    elif (max_index == 6):
        angle = -15
        result = " : Average left slanted"
    elif (max_index == 7):
        angle = -30
        result = " : Above average left slanted"
    elif (max_index == 8):
        angle = -45
        result = " : Extremely left slanted"
    elif (max_index == 4):
        if s_function[3] == 0.0:
            p = s_function[4]  
            q = s_function[4] 
        else:
            p = s_function[4] / s_function[3]
            q = s_function[4] / s_function[5]
        if ((p <= 1.2 and q <= 1.2) or (p > 1.4 and q > 1.4)):
            angle = 0
            result = " : No slant"
        elif ((p <= 1.2 and q-p > 0.4) or (q <= 1.2 and p-q > 0.4)):
            angle = 0
            result = " : No slant"
        else:
            max_index = 9
            angle = 180
            result = " : Irregular slant behaviour"

        if angle == 0:
            print("\n************************************************")
            print("Slant determined to be straight.")
        else:
            print("\n************************************************")
            print("Slant determined to be irregular.")
            angle = 0
            result = " : Straight/No Slant"
            print("Set as"+result)
            print("************************************************\n")
        
        

    SLANT_ANGLE = angle
    
    return

def pen_pressure(image : cv2.typing.MatLike) -> None:
    global PEN_PRESSURE

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:]
    inverted = image
    for x in range(h):
        for y in range(w):
            inverted[x][y] = 255 - image[x][y]

    filtered = bilateralFilter(inverted, 3)
    ret, thresh = cv2.threshold(filtered, 100, 255, cv2.THRESH_TOZERO)
    total_intensity = 0
    pixel_count = 0
    for x in range(h):
        for y in range(w):
            if (thresh[x][y] > 0):
                total_intensity += thresh[x][y]
                pixel_count += 1

    average_intensity = float(total_intensity) / pixel_count
    PEN_PRESSURE = average_intensity
    return

def analyze_writing_style(image_path : str) -> dict:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    arcade_count = 0
    garland_count = 0
    angle_count = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        middle_y = y + h // 2
        middle_zone = binary[middle_y-10:middle_y+10, x:x+w]
        upper_profile = []
        for col in range(middle_zone.shape[1]):
            non_zero = np.nonzero(middle_zone[:, col])[0]
            if len(non_zero) > 0:
                upper_profile.append(non_zero[0])
            else:
                upper_profile.append(middle_zone.shape[0])
        
        peaks, _ = find_peaks(upper_profile, distance=5)
        valleys, _ = find_peaks([-p for p in upper_profile], distance=5)
        
        if len(peaks) > 0 and len(valleys) > 0:
            peak_height = np.mean([upper_profile[p] for p in peaks])
            valley_height = np.mean([upper_profile[v] for v in valleys])
            
            if peak_height < valley_height - 3:  # Arcade
                arcade_count += 1
            elif peak_height > valley_height + 3:  # Garland
                garland_count += 1
            else:  # Angle
                angle_count += 1
    
    total = arcade_count + garland_count + angle_count
    if total > 0:
        arcade_ratio = arcade_count / total
        garland_ratio = garland_count / total
        angle_ratio = angle_count / total
        
        dominant_style = max(('Arcade', arcade_ratio), ('Garland', garland_ratio), ('Angle', angle_ratio))
        
        return {'Note' : 'Garland Detected',
            'Dominant Style': dominant_style[0],
            'Arcade Ratio': arcade_ratio,
            'Garland Ratio': garland_ratio,
            'Angle Ratio': angle_ratio
        }
    else:
        return {'Note' : "No clear style detected",
            'Dominant Style': dominant_style[0],
            'Arcade Ratio': arcade_ratio,
            'Garland Ratio': garland_ratio,
            'Angle Ratio': angle_ratio
        }

def analyze_lower_loops(image_path : str) -> list[dict]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for contour in contours:
        # Filter contours based on area and aspect ratio to identify potential lower loops
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(h) / w
        
        if area > 100 and aspect_ratio > 1.5:
            # Create a mask for the loop
            mask = np.zeros(binary.shape, np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            props = regionprops(mask)[0]
            
            # Determine loop type
            if props.eccentricity < 0.5:  # More circular
                loop_type = "Circular - Full loop"
            elif props.eccentricity < 0.8:  # Elongated
                loop_type = "A little elongated like a Cradle"
            else:  # Very elongated
                loop_type = "Elongated like a straight stroke"
            
            mean_intensity = cv2.mean(image, mask=mask)[0]
            pressure = "Heavy" if mean_intensity < 128 else "Light"
            
            results.append({
                'type': loop_type,
                'pressure': pressure,
                'area': area,
                'eccentricity': props.eccentricity
            })
    
    return results




def extract_features(file_name : str) -> dict :
    global BASELINE_ANGLE 
    global TOP_MARGIN 
    global LETTER_SIZE 
    global LINE_SPACING 
    global WORD_SPACING 
    global PEN_PRESSURE 
    global SLANT_ANGLE 
    image = cv2.imread(file_name)

    # Extract pen pressure. 
    pen_pressure(image)

    # straighten and extract contours
    straightened = contour_straightener(image)
    
    # extract lines 
    lineIndices = lines_extract(straightened)

    # extract words 
    wordCoordinates = extract_words(straightened, lineIndices)

    # extract average slant angle of all the words containing a long vertical stroke
    extract_slant(straightened, wordCoordinates)

    GARLANDS = analyze_writing_style(file_name)
    analyzed_loops = analyze_lower_loops(file_name)

    loop_types = [loop['type'] for loop in analyzed_loops]
    pressures = [loop['pressure'] for loop in analyzed_loops]
    LOOP_ANALYSIS = {}
    try : 


        print(f"""Loop Analysis:
        Total lower loops detected: {len(analyzed_loops)}
        Variety of shapes: {'High' if len(set(loop_types)) > 2 else 'Low'}
        Dominant loop type: {max(set(loop_types), key=loop_types.count)}
        Dominant pressure: {max(set(pressures), key=pressures.count)}""")

        LOOP_ANALYSIS = {"Note" : "Loops Detected" , "Total Number of Loops" : len(analyzed_loops), "Variety of Loops" : 'High' if len(set(loop_types)) > 2 else 'Low' , "Dominant Loop Type" : max(set(loop_types), key=loop_types.count)}

        if len(set(loop_types)) > 2 and len(analyzed_loops) > 5:
            print("  Note: Many varied shapes in the lower zone detected. This may indicate unsettled or unfocused emotions.")
    except ValueError as e:
        print("""Loop Analysis report
              Note : The Handwriting is too chaotic to determine a loop.""") 
        LOOP_ANALYSIS = {"Note" : "The Handwriting is too chaotic to determine a loop." , "Total Number of Loops" : 0, "Variety of Loops" : "Cannot be determined" , "Dominant Loop Type" : "Cannot be determined"}


    BASELINE_ANGLE = round(BASELINE_ANGLE, 2)
    TOP_MARGIN = round(TOP_MARGIN, 2)
    LETTER_SIZE = round(LETTER_SIZE, 2)
    LINE_SPACING = round(LINE_SPACING, 2)
    WORD_SPACING = round(WORD_SPACING, 2)
    PEN_PRESSURE = round(PEN_PRESSURE, 2)
    SLANT_ANGLE = round(SLANT_ANGLE, 2)

    return {"Loops" : LOOP_ANALYSIS,"GARLAND" : GARLANDS,"Baseline Angle" : BASELINE_ANGLE,"Top Margin" : TOP_MARGIN,"Letter Size" : LETTER_SIZE,"Line Spacing" : LINE_SPACING,"Word Spacing" : WORD_SPACING,"Pen Pressure": PEN_PRESSURE, "Slant Angle":SLANT_ANGLE}



