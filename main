'''
使用tk创建界面
功能：
1. 左侧显示摄像头图像
2. 右侧按钮功能测试，点击开始进行检测
3. 点击创建模板，可以通过鼠标框选创建定位模板
4. 点击创建ROI，可以通过鼠标框选创建检测点

'''
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import math


def toggle_detection():
    global is_detecting
    is_detecting = not is_detecting
    start_button.config(text='停止' if is_detecting else '开始')
def new_template():
    global cap
    ret, frame = cap.read()
    cv2.imwrite("D:/AI/vision/FindTemplate/V188/part1/standard_pic.jpg", frame)
    # 选择ROI
    roi = cv2.selectROI(windowName="original", img=frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    with open('D:/AI/vision/FindTemplate/V188/part1/template_pos.txt', 'w') as f:
        f.write('{:},{:}\n'.format(x,y))
    crop = frame[y:y+h, x:x+w]
    cv2.imwrite("D:/AI/vision/FindTemplate/V188/part1/template.jpg", crop)
def new_ROI():
    
    standard_pic = cv2.imread('D:/AI/vision/FindTemplate/V188/part1/standard_pic.jpg')
    roi = cv2.selectROI(windowName="original", img=standard_pic, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    with open('D:/AI/vision/FindTemplate/V188/part1/roi_pos.txt', 'w') as f:
        f.write('{:},{:},{:},{:}\n'.format(x,y,w,h))
    crop = standard_pic[y:y+h, x:x+w]
    cv2.imwrite("D:/AI/vision/FindTemplate/V188/part1/roi.jpg", crop)
def generate_rotated_templates(template, angles):
    h, w = template.shape[:2]
    center = (w // 2, h // 2)
    rotated_templates = []

    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(template, M, (w, h))
        rotated_templates.append((rotated, angle))

    return rotated_templates

def draw_rotated_rectangle(image, center, width, height, angle, color, thickness):
    rect = ((center[0], center[1]), (width, height), angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color, thickness)

def update_frame():
    global cap
    ret, frame = cap.read()
    if not ret:
        return

    # 转换为灰度图像，以便进行模板匹配
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    best_match_value = -1
    best_match_top_left = (0, 0)
    best_template = None
    best_angle = 0

    if is_detecting:
        for rotated_template, angle in rotated_templates:
            # 进行模板匹配
            result = cv2.matchTemplate(gray_frame, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_match_value:
                best_match_value = max_val
                best_match_top_left = max_loc
                best_template = rotated_template
                best_angle = angle

        # 显示最佳匹配结果
        if best_match_value > 0.7:
            top_left = best_match_top_left
            center_x = top_left[0] + template_w // 2
            center_y = top_left[1] + template_h // 2

            draw_rotated_rectangle(frame, (center_x, center_y), template_w, template_h, -best_angle, (0, 255, 0), 2)
            cv2.putText(frame, f'Similarity: {best_match_value:.2f}, Angle: {best_angle}', 
                        (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # 计算检测框的位置（考虑旋转角度）
            if(best_angle<0):

                detection_center_x = center_x+math.cos(abs(math.radians(best_angle)))*offset_x-math.sin(abs(math.radians(best_angle)))*offset_y
                detection_center_y = center_y+math.sin(abs(math.radians(best_angle)))*offset_x+math.cos(abs(math.radians(best_angle)))*offset_y
            else:
                detection_center_x = center_x+math.cos(abs(math.radians(best_angle)))*offset_x+math.sin(abs(math.radians(best_angle)))*offset_y
                detection_center_y = center_y-math.sin(abs(math.radians(best_angle)))*offset_x+math.cos(abs(math.radians(best_angle)))*offset_y

            draw_rotated_rectangle(frame, (detection_center_x, detection_center_y), roi_pos_w, roi_pos_h, -best_angle, (255, 0, 0), 2)

            # 检测框的位置（中心点为基准）
            detection_top_left =  (int(detection_center_x - roi_pos_w // 2), int(detection_center_y - roi_pos_h // 2))
            detection_bottom_right = (int(detection_center_x + roi_pos_w // 2), int(detection_center_y + roi_pos_h // 2))
            
            
            # 在检测框内进行模板匹配
            detection_frame = gray_frame[detection_top_left[1]:detection_bottom_right[1],
                                         detection_top_left[0]:detection_bottom_right[0]]
            avg_gray = np.mean(detection_frame)
            cv2.putText(frame, f'Gray: {avg_gray:.2f}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

          
            

    # 转换图像格式以适应Tkinter
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    # 更新图像标签
    image_label.imgtk = img_tk
    image_label.configure(image=img_tk)

    # 每隔20ms更新一次图像
    window.after(10, update_frame)

# 打开相机
cap = cv2.VideoCapture(0)


#创建主窗口
window=tk.Tk()
window.title('Camera')

# 初始化检测状态
is_detecting = False

# 创建图像显示Label
image_label = tk.Label(window)
image_label.pack(side=tk.LEFT)

 # 加载需要匹配的模板图像
template = cv2.imread('D:/AI/vision/FindTemplate/V188/part1/template.jpg', 0)
cv2.imshow('template',template)
template_w, template_h = template.shape[::-1]
# 读取模板和检测框参数
template_pos=open('D:/AI/vision/FindTemplate/V188/part1/template_pos.txt')
line_template = template_pos.readline().strip() #读取第一行
template_pos_x=int(line_template.split(',')[0])
template_pos_y=int(line_template.split(',')[1])
template_pos=open('D:/AI/vision/FindTemplate/V188/part1/roi_pos.txt')
line_roi = template_pos.readline().strip() #读取第一行
roi_pos_x=int(line_roi.split(',')[0])
roi_pos_y=int(line_roi.split(',')[1])
roi_pos_w=int(line_roi.split(',')[2])
roi_pos_h=int(line_roi.split(',')[3])
offset_x = (roi_pos_x+roi_pos_w/2)-(template_pos_x+template_w/2)
offset_y = (roi_pos_y+roi_pos_h/2)-(template_pos_y+template_h/2)
# 设定灰度值范围
gray_min = 0
gray_max = 250
# 创建按钮
button_frame = tk.Frame(window)
start_button = tk.Button(button_frame, text="开始", command=toggle_detection)
template_button = tk.Button(button_frame, text="创建模板", command=new_template)
roi_button = tk.Button(button_frame, text="创建ROI", command=new_ROI)
start_button.pack(side=tk.TOP, padx=10, pady=5)
template_button.pack(side=tk.TOP, padx=10, pady=5)
roi_button.pack(side=tk.TOP, padx=10, pady=5)
button_frame.pack(side=tk.RIGHT, padx=10)

# 生成多个角度的检测模板
angle_range = range(-15, 15, 1)  # 设定的角度范围，例如 -30° 到 30°，每隔5度旋转一次
rotated_templates = generate_rotated_templates(template, angle_range)

# 启动图像更新
update_frame()
window.mainloop()

# 清理资源
cap.release()
cv2.destroyAllWindows()

