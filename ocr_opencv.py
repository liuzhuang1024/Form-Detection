import cv2
import numpy as np
from PIL import Image
from txt_classify.model import predict as keras_densenet

class OCR_main():
    def __init__(self):
        pass

    def text_detect_v2(self, image):
        # stamp check
        # mask = self.check_stamp(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # th1 = th1 - cv2.bitwise_and(th1, th1, mask=mask)
        th1_orig = th1
        # delete table
        th1 = self.delete_table(th1, 10)
        rects, out_img, _ = self.text_detect_v1(th1, image.copy(), (0, 255, 0), 4)
        cv2.imwrite(str(len(rects)) + '.jpg', out_img)
        merge_rects = []
        if (len(rects)):
            for rect in rects:
                if rect[3] - rect[1] >= 35 and rect[2] - rect[0] >= 100:
                    partImg = th1_orig[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]
                    rects_second, _ = self.text_second_detect(partImg, image.copy(), (255, 0, 0), 2)
                    if len(rects_second) == 1:
                        merge_rects.append(rect)
                    else:
                        for rect_second in rects_second:
                            merge_rects.append((rect[0] + rect_second[0], rect[1] + rect_second[1],
                                                rect[0] + rect_second[2], rect[1] + rect_second[3]))
                else:
                    merge_rects.append(rect)
        return merge_rects


    def check_stamp(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 10, 120])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([156, 10, 120])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        return mask


    def delete_table(self, img, scale):
        try:
            th1 = img
            rows, cols = th1.shape
            kernel1 = np.ones((cols // scale, 1), np.uint8)
            kernel2 = np.ones((1, rows // scale), np.uint8)
            kernel5 = np.ones((1, 4), np.uint8)

            th_orig = th1

            img_cols = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel1)
            img_rows = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel2)

            img_line = cv2.add(img_cols, img_rows)
            img_dot = cv2.bitwise_and(img_cols, img_rows)
            img_sub = cv2.subtract(th_orig, img_line)
        except:
            print('delete table error!')
            return img
        else:
            return img_sub


    def line_detect(self, image):
        height, width, _ = image.shape
        img_forshow = image.copy()
        mask = np.zeros((height, width), dtype=np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(th1, 50, 150, apertureSize=5)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # HoughLines P
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=70, maxLineGap=4)
        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_forshow, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 4)
        except:
            pass

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_fordetect = cv2.subtract(th1, mask)
        return mask, img_forshow, img_fordetect


    def further_delete_line(self, image, padding):
        height, width, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        rows_sum = th1.sum(axis=1) // width
        cols_sum = th1.sum(axis=0) // height

        height_padding = int(0.5 * height)
        width_padding = int(0.33 * width)

        left_padding = width_padding
        right_padding = width - width_padding
        top_padding = height_padding
        bottom_padding = height - height_padding

        rows_loc = np.argwhere(rows_sum > 200)
        cols_loc = np.argwhere(cols_sum > 200)

        if rows_loc.sum():
            top_gaps = rows_loc - top_padding
            if np.where(top_gaps < 0)[0].sum():
                min_top_gap = abs(np.max(top_gaps[np.where(top_gaps < 0)]))
                top = top_padding - min_top_gap + padding
            else:
                top = 0
            bottom_gaps = rows_loc - bottom_padding
            if np.where(bottom_gaps > 0)[0].sum():
                min_bottom_gap = abs(np.min(bottom_gaps[np.where(bottom_gaps > 0)]))
                bottom = bottom_padding + min_bottom_gap - padding
            else:
                bottom = height
        else:
            top = 0
            bottom = height
        if cols_loc.sum():
            left_gaps = cols_loc - left_padding
            if np.where(left_gaps < 0)[0].sum():
                min_left_gap = abs(np.max(left_gaps[np.where(left_gaps < 0)]))
                left = left_padding - min_left_gap + padding
            else:
                left = 0
            right_gaps = cols_loc - right_padding
            if np.where(right_gaps > 0)[0].sum():
                min_right_gap = abs(np.min(right_gaps[np.where(right_gaps > 0)]))
                right = right_padding + min_right_gap - padding
            else:
                right = width
        else:
            left = 0
            right = width
        crop_img = gray[top:bottom, left:right]
        rect = (left, top, right, bottom)
        return crop_img, rect


    def text_detect_v1(self, thresh, output, color_scalar, color_width):
        padding = 6
        w_min = 30
        h_min = 20
        rects = []
        kernel = np.ones((2, 4), np.uint8)
        kernel2 = np.ones((2, 4), np.uint8)

        try:
            blur = cv2.medianBlur(thresh, 3)
            temp_img = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=2)
            line_img = cv2.dilate(temp_img, kernel2, iterations=7)
            height, width = line_img.shape

            contours, _ = cv2.findContours(line_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                x1 = x - 5
                y1 = y - 9
                x2 = x + w - 5
                y2 = y + h
                w = x2 - x1
                h = y2 - y1
                if w < w_min or h < h_min:
                    continue
                elif abs(x - 0) < padding or abs(x - width) < padding:
                    continue
                elif abs(y - 0) < padding or abs(y - height) < padding:
                    continue
                elif h // w > 5:
                    continue
                else:
                    rects.append((x1, y1, x2, y2))
                    cv2.rectangle(output, (x1, y1), (x2, y2), color_scalar, color_width)

        except:
            print('text detection error!')
            return rects, output
        else:
            return rects, output, line_img


    def text_second_detect(self, thresh, output, color_scalar, color_width):
        threshold = 0.05
        gap = 15
        rects = []
        height, width = thresh.shape
        rows = thresh.sum(axis=1) // 255
        lines = []
        lines.append(0)
        lines.append(height)
        for i in range(0, len(rows)):
            if i == 0:
                if rows[i] < min(rows[i + 1], rows[i + 2]) and rows[i] < threshold * width:
                    lines.append(i)
            elif i == 1:
                if rows[i] < min(rows[i + 1], rows[i + 2], rows[i - 1]) and rows[i] < threshold * width:
                    lines.append(i)
            elif i == len(rows) - 1:
                if rows[i] < min(rows[i - 1], rows[i - 2]) and rows[i] < threshold * width:
                    lines.append(i)
            elif i == len(rows) - 2:
                if rows[i] < min(rows[i + 1], rows[i - 2], rows[i - 1]) and rows[i] < threshold * width:
                    lines.append(i)
            elif rows[i] < min(rows[i - 1], rows[i - 2], rows[i + 1], rows[i + 2]) and rows[i] < threshold * width:
                lines.append(i)
        if len(lines) > 2:
            lines = self.merge_ele(lines, gap)
            if len(lines):
                if lines[0] >= gap:
                    lines.insert(0, 0)
                if abs(lines[-1] - height) >= gap:
                    lines.append(height)

                for i in range(0, len(lines)):
                    if i == 0:
                        cv2.line(output, (0, lines[i]), (width, lines[i]), color_scalar, color_width)
                    else:
                        if (abs(lines[i] - lines[i - 1]) >= gap):
                            cv2.line(output, (0, lines[i]), (width, lines[i]), color_scalar, color_width)
                            rects.append((0, lines[i - 1], width, lines[i]))
                return rects, output
            else:
                rects.append((0, 0, width, height))
                return rects, output
        else:
            rects.append((0, 0, width, height))
            return rects, output


    def merge_ele(self, list, min_gap):
        list.sort()
        gap_list = []
        output = []
        for i in range(0, len(list)):
            if i == 0:
                continue
            else:
                gap_list.append(abs(list[i] - list[i - 1]))
        # print(gap_list)

        gaps = [i for i in range(0, len(gap_list)) if gap_list[i] > min_gap]
        if len(gaps):
            for i in range(len(gaps)):
                if i == 0:
                    output.append(list[gaps[i]])
                    output.append(list[gaps[i] + 1])
                else:
                    output.append(list[gaps[i] + 1])
            return output
        else:
            return output


    def takeSecond(self, elem):
        return elem[1]


    def takeFirst(self, elem):
        return elem[0]


    def sort_dict(self, dict):
        output = {}
        rects = []
        for key, value in dict.items():
            rects.append(key)
        rects.sort(key=self.takeSecond)
        cols = []
        loc = []
        try:
            for rect in rects:
                if rect == rects[-1]:
                    if len(cols):
                        if abs(rect[1] - cols[-1][1]) < 20:
                            cols.append(rect)
                            cols.sort(key=self.takeFirst)
                            loc.append(cols[0][0])
                            loc.append(cols[0][1])
                            loc.append(cols[-1][2])
                            loc.append(cols[-1][3])
                            for col in cols:
                                str_tmp += dict[col]
                            output[str_tmp] = str(loc)
                            cols.clear()
                            loc.clear()
                        else:
                            cols.sort(key=self.takeFirst)
                            loc.append(cols[0][0])
                            loc.append(cols[0][1])
                            loc.append(cols[-1][2])
                            loc.append(cols[-1][3])
                            for col in cols:
                                str_tmp += dict[col]

                            output[str_tmp] = str(loc)
                            loc.clear()
                            loc.append(rect[0])
                            loc.append(rect[1])
                            loc.append(rect[2])
                            loc.append(rect[3])
                            output[dict[rect]] = str(loc)

                    else:
                        loc.append(rect[0])
                        loc.append(rect[1])
                        loc.append(rect[2])
                        loc.append(rect[3])
                        output[dict[rect]] = str(loc)
                elif len(cols) == 0:
                    cols.append(rect)
                    str_tmp = ''

                elif abs(rect[1] - cols[-1][1]) < 20:
                    cols.append(rect)
                else:
                    cols.sort(key=self.takeFirst)
                    for col in cols:
                        str_tmp += dict[col]
                    loc.append(cols[0][0])
                    loc.append(cols[0][1])
                    loc.append(cols[-1][2])
                    loc.append(cols[-1][3])
                    output[str_tmp] = str(loc)
                    cols.clear()
                    loc.clear()
                    cols.append(rect)
                    str_tmp = ''
            return output
        except:
            print('sort dict error')
            return output

    # ocr部分
    def rec_txt(self, cropped_image):
        # text detection
        mask = self.check_stamp(cropped_image)
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cropped_image = cv2.cvtColor(cv2.add(gray, mask), cv2.COLOR_GRAY2BGR)

        # _, _, img_fordetect = self.line_detect(cropped_image)
        # img_fordetect = cv2.cvtColor(img_fordetect, cv2.COLOR_GRAY2BGR)
        rects = self.text_detect_v2(cropped_image)
        string_dict = {}
        # print(rects)
        # t0 = time.time()
        for rect in rects:
            (x1, y1, x2, y2) = rect
            if y1 < cropped_image.shape[0]:
                if (y2 - y1) > 0 and (x2 - x1) > 0:
                    partImg = cropped_image[int(y1):int(y2), int(x1):int(x2), :]
                    # partImg, rect2 = self.further_delete_line(partImg, 3)
                    # if rect2[2] - rect2[0] < 20:
                    #     continue
                    # elif rect2[3] - rect2[1] < 20:
                    #     continue
                    # elif (rect2[3] - rect2[1]) // (rect2[2] - rect2[0]) > 3:
                    #     continue
                    # else:
                    partImg = Image.fromarray(partImg).convert('L')
                    if partImg.size[0] / partImg.size[1] < 0.3:
                        continue
                    text = keras_densenet(partImg)
                    string_dict[rect] = text

        output_list = self.sort_dict(string_dict) if len(string_dict) else {}
        # print(output_list)
        # print(time.time() - t0)

        return output_list
if __name__ == "__main__":
    pass
    image_file = '9.jpg'
    image=cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), 1)
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    cropped_image = cv2.cvtColor(cv2.add(gray, mask), cv2.COLOR_GRAY2BGR)

    ocr = OCR_main()
    temp_dict = ocr.rec_txt(image)
    for i in temp_dict.keys():
        print(i)