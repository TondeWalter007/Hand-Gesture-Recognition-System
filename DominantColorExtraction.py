import colorsys
import numpy as np
import cv2
from sklearn.cluster import KMeans
import webcolors

print("BEGIN COLOR EXTRACTION...")
print("")


def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def get_color_name(requested_color):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_color)
    except ValueError:
        closest_name = closest_color(requested_color)
        actual_name = None
    return actual_name, closest_name


def rgb_to_hsv(r, g, b):
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    # h, s, v = hue, saturation, value
    cmax = max(r, g, b)  # get maximum of r, g, b
    cmin = min(r, g, b)  # get minimum of r, g, b
    diff = cmax - cmin  # get diff of cmax and cmin.
    
    # if cmax and cmax are equal then h = 0
    if cmax == cmin:
        h = 0
    # if cmax equal r then compute h
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360
    # if cmax equal zero
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100
    # compute v
    
    v = cmax * 100
    return h / 2, s, v # half the hue to use in cv2


def get_dominant_color(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.reshape((img.shape[0] * img.shape[1], 3))

    n_clusters = 2
    kmeans = KMeans(n_clusters) # number 
    kmeans.fit(img)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    label_count = [0 for i in range(n_clusters)]
    for elements in labels:
        label_count[elements] += 1
    index_color = label_count.index(max(label_count))

    #actual_name, closest_name = get_color_name(colors[index_color])
    dominant_rgb = colors[index_color]
    hsv = rgb_to_hsv(dominant_rgb[0], dominant_rgb[1], dominant_rgb[2])
    #print("Dominant Color:")
    #print("RGB:", dominant_rgb)
    #print("HSV: ", hsv)
    #print("Actual name -> " + str(actual_name) + ", Closest name -> " + closest_name)

    h, s, v = hsv[0], hsv[1], hsv[2]

    return h / 2, s, v


def get_hsv_threshold(image):
    OG_image = image
    #OG_image = cv2.resize(OG_image, None, fx=0.1, fy=0.1)
    hsv = cv2.cvtColor(OG_image, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (35, 35), 0) # blur the image to smooth the edges

    dominant_hsv_color = get_dominant_color(image)
    # separate the hue, saturation and value
    h = dominant_hsv_color[0] 
    s = dominant_hsv_color[1]
    v = dominant_hsv_color[2]

    h_min = h - 50 # minimum hue in color space for skin color
    s_min = s + 50 # minimum saturation
    v_min = v + 0 # mnimum value 
    h_max = h + 50
    s_max = 255
    v_max = 255

    if h_min < 0:
        h_min = 0
    if s_min > 255:
        s_min = 255
    if v_min > 255:
        v_min = 255
    if h_max > 255:
        h_max = 255

    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    # apply mask to get thresholded image (black and white pixels)
    mask = cv2.inRange(blur, lower_bound, upper_bound)
    res = cv2.bitwise_and(OG_image, OG_image, mask=mask)

    return mask



