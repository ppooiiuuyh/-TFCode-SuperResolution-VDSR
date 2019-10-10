import cv2
import numpy as np

def find_dot_pos(img, threshold = 40, num_hori_dot = 7, visualize=False):
    """ binarization """
    img = np.mean(img,axis=-1)
    img = ((img > threshold)*255).astype(np.uint8)


    """ find contures """
    contours, hierachy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = np.array(contours)

    """ choose active contures & find dot pos"""
    num_dots_contained_in_conture = []
    for c in contours:
        num_dots_contained_in_conture.append(c.shape[0])
    active_num_dots = np.median(num_dots_contained_in_conture)

    conture_list = []
    pos_list = []
    conture_center_sqaure = None
    for c in contours:
        #c = np.squeeze(c) #(None,1,2) -> (None,2)
        c = c.reshape([-1,2])
        if len(c) > active_num_dots*0.6 and len(c) < active_num_dots*1.5:
            conture_list.append(c)
            pos_list.append( (int(np.mean(c[:,0])), int(np.mean(c[:,1]))) )
    conture_list = np.array(conture_list)
    pos_list = np.array(pos_list)
    #print(len(pos_list))

    """ sorting """
    pos_list_temp = np.copy(pos_list); pos_list = []
    conture_list_temp = np.copy(conture_list); conture_list = []
    while len(pos_list_temp) >0 :
        #가로
        idx = np.argsort(pos_list_temp [:, 0])
        pos_list_temp = pos_list_temp[idx]
        temp_horizontal = pos_list_temp[:num_hori_dot]
        pos_list_temp = pos_list_temp[num_hori_dot:]

        conture_list_temp = conture_list_temp[idx]
        temp_horizontal_conture = conture_list_temp[:num_hori_dot]
        conture_list_temp = conture_list_temp[num_hori_dot:]

        #세로
        idx = np.argsort(temp_horizontal[:,1])
        temp_horizontal = temp_horizontal[idx]
        temp_horizontal_conture = temp_horizontal_conture[idx]
        pos_list += list(temp_horizontal)
        conture_list += list(temp_horizontal_conture)
        #print(len(pos_list),len(conture_list),temp_horizontal)

    pos_list = np.array(pos_list)
    conture_list = np.array(conture_list)


    """ find center pos """
    # there may exist one large conture
    center_pos = None
    center_square_pos = None
    for c in contours:
        #c = np.squeeze(c) #(None,1,2) -> (None,2)
        c = c.reshape([-1,2])
        if len(c) > active_num_dots*1.5:
            center_square_pos = (int(np.mean(c[:,0])), int(np.mean(c[:,1])))
    if center_square_pos == None :
        return pos_list,center_pos,conture_list


    # 사각형의 중심점과 가장 가까운 점이 중심점일것
    min_temp = 9999
    for p in pos_list:
        dist = np.sqrt(np.sum((np.array(p) - np.array(center_square_pos)) ** 2))
        if dist < min_temp :
            min_temp = dist
            center_pos = p



    """ visualize for evalution """
    if visualize == True:
        bg = (img * 0.2).astype(np.uint8)
        for ds,p in zip(conture_list,pos_list):
            for d in ds:
                bg[d[1],d[0]] = 255
            bg[p[1],p[0]]=255

            bg[center_pos[1],center_pos[0]]=255
            cv2.imshow("image",bg)
            cv2.waitKey(0)

    return pos_list, center_pos, conture_list



""" ====================================================================
                        module test
==================================================================== """
if __name__ == "__main__":
    img = cv2.imread("./21.jpg", cv2.IMREAD_GRAYSCALE)
    find_dot_pos(img)