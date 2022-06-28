###########################################
###########################################

#               ตรวจจับการแตกร้าว
#               threshold ของ 
#           save ภาพวันที่เวลากี่โมงหน้า last
#                   put txt

###########################################
###########################################



# เรามี x y w h จากไฟล์ label
# เราต้องการ H W (ขนาดความสูงและความกว้างของรูปภาพ ใช้คำสั่ง .shape ได้เลย)
# สามารถแปลงโดยใช้สมการด้านล่างนี้ครับ
#x1 = (x - w / 2) * W
#y1 = (y - h / 2) * H
# = (x + w / 2) * W
#y2 = (y + h / 2) * H
## และใช้ x1 y1 x2 y2 เป็น bounding box
#return (x1,y1,x2,y2)

# # import win32api
# #
# # win32api.MessageBox(0, 'hello', 'title')

# # import tkinter
# # from tkinter import messagebox
# #
# # # This code is to hide the main tkinter window
# # # root = tkinter.Tk()
# # # root.withdraw()
# #
# # # Message Box
# # messagebox.showinfo("Title", "Message")

# x1 = 15
# x2 = 15
# x3 = 15
# x4 = 15
#
# if x1 < 20:
#     x11 = x1
#
# if x2 < 14:
#      x22 = x2

# if x3 < 20:
#     x33 = x3
#
# if x4 > 20:
#     x44 = x4

# else:
#     x11 = 0
#     x22 = 0
    # x33 = 0
    # x44 = 0

# print("x1 = ",x11)
# print("x2 = ",x22)
# print("x3 = ",x33)
# print("x4 = ",x44)

# var = 200
# var2 = 150
# var3 = 100
#
# if var == 200:
#    print ("1 - Got a true expression value")
#    print (var)
# if var2 == 150:
#    print ("2 - Got a true expression value")
#    print (var2)
# if var3 == 100:
#    print ("3 - Got a true expression value")
#    print (var3)
# else:
#    print ("4 - Got a false expression value")
#    print (var)
#
# print ("Good bye!")
print("Enter your number_length:")
number_length = int(input())
print("Enter your number_alingment:")
number_alingment = int(input())
print("Enter your number_apposition:")
number_apposition = int(input())

outputLength = 20
outputAlingment = 20
outputApposition = 0

if number_length >= outputLength or number_alingment >= outputAlingment or number_apposition >= outputApposition:
    result = "ผ่าตัด"

if number_length <= outputLength and number_alingment <= outputAlingment and number_apposition <= outputApposition:
    result = "เข้าเฝือก"
else:
    result = "ผ่าตัด1"

print("outputLength === ",type(outputLength)," ==  " ,outputLength)
print("outputAlingment === ",type(outputAlingment)," ==  " ,outputAlingment)
# print("outputApposition === ",outputApposition)
print("result === ",type(result)," ==  " ,result)


 first_surname = TextAreaField("กรุณาใส่ชื่อจริง-นามสกุล")
    sex = TextAreaField("กรุณาระบุเพศ")
    age = IntegerField("di6Ik")
    submit = SubmitField("ตกลง")

 first_surname = ""
        age = 0
        form = MyForm()
        if form.validate_on_submit():
            first_surname = form.first_surname.data
            age = form.age.data
            sex = form.sex.data
            
            form.first_surname.data = ""
            form.age.data = "" 
            form.sex.data = ""
        image = cv2.imread("../yolov5-flask/static/image0.jpg")
        text_data = first_surname+" "+age+" "+sex
        image_text_data =  cv2.putText(image, text_data, (0,0), cv2.FONT_HERSHEY_SIMPLEX,0.4, (255, 255, 255), 1, cv2.LINE_AA)
        image_resize_text = cv2.resize(image,(416,416))
        image_text = cv2.putText(image_resize_text, image_text_data, (0,410), cv2.FONT_HERSHEY_SIMPLEX,0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imwrite("../yolov5-flask/static/image2.jpg", image_text)