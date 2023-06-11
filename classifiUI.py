from classification import Ui_Form as Class_Ui
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
import numpy as np
import onnxruntime as ort
import torch
import cv2
from PyQt5.Qt import QApplication



def predict(img_path):
    img = cv2.imread(img_path)  # 读取图片
    img = cv2.resize(img, (224, 224))  # 调整图片尺寸

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 把图片BGR变成RGB

    img = np.transpose(img, (2, 0, 1))  # 调整维度将HWC - CHW
    img = np.expand_dims(img, 0)  # 添加一个维度 就是batch维度
    img = img.astype(np.float32)  # 格式转成float32
    img /= 255
    ort_session = ort.InferenceSession("home.onnx", providers=torch.device("cpu"))
    # 调用onnxruntime run函数进行模型推理
    outputs = ort_session.run(
        None,
        {"image": img},
    )
    # outputs的输出类型为list类型，所以要先将list转换成numpy再转换成torch
    outputs1 = torch.from_numpy(np.array(outputs))
    # 通过softmax进行最后分数的计算
    value = float(torch.max(torch.softmax(outputs1[0], dim=1)))
    #outputs_softmax = torch.softmax(outputs1[0], dim=1).numpy()[:, 0].tolist()[0]
    index = np.argmax(np.array(outputs))+1

    return value,index

class CamShow(QMainWindow, Class_Ui):
    def __init__(self, parent=None):
        super(CamShow, self).__init__(parent)
        self.setupUi(self)
        self.upload.clicked.connect(self.loadImage)
        self.interference.clicked.connect(self.predict_label)
    # 打开文件功能
    def loadImage(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, '请选择图片', '.', '图像文件(*.jpg *.jpeg *.png)')
        if self.fname:
            #print(self.fname)
            #self.Infolabel.setText("文件打开成功\n" + self.fname)
            # self.Imglabel.set
            #self.result.setText(self.fname)
            jpg = QtGui.QPixmap(self.fname).scaled(self.imglabel.width(), self.imglabel.height())

            print(jpg)
            self.imglabel.setPixmap(jpg)
        else:
            print("打开文件失败")
            #self.Infolabel.setText("打开文件失败")

    def predict_label(self):
        # 开启线程
        #self.result.setText(self.fname)
        if not self.fname:
            self.result.setText("为空退出")
        else:
            value,index = predict(self.fname)
            if index==1:
                self.result.setText("新冠")
            else :
                if index==2:
                    self.result.setText("病毒性肺炎")
                else:
                    if index==3:
                        self.result.setText("正常")
            self.score.setText(str(value))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = CamShow()
    ui.show()
    sys.exit(app.exec_())