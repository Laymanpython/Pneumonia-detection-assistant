from mainUI import Ui_Form as A_Ui  # a界面的库
from classification import Ui_Form as B_Ui  # b界面的库
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from FormUI import Ui_Form
from paddleseg.models import BiSeNetV2
import paddleseg.transforms as T
from paddleseg.core import infer
import paddle
import SimpleITK as sitk
import os
import warnings
from classification import Ui_Form as Class_Ui
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import numpy as np
import onnxruntime as ort
import torch
import cv2

warnings.filterwarnings('ignore')
from PyQt5 import QtCore, QtWidgets
import sys


class AUi(QtWidgets.QMainWindow, A_Ui):
    def __init__(self):
        super(AUi, self).__init__()
        self.setupUi(self)


class BUi(QtWidgets.QMainWindow, B_Ui):
    def __init__(self):
        super(BUi, self).__init__()
        self.setupUi(self)


def windowwc(sitkImage, ww=1500, wc=-550):
    """
    主要用于设置窗宽窗位
    @param sitkImage:SimpleITK图像数据
    @param ww:窗宽窗位
    @param wc:窗宽窗位
    @return:sitkImage
    """
    min = int(wc - ww / 2.0)
    max = int(wc + ww / 2.0)
    intensityWindow = sitk.IntensityWindowingImageFilter()
    intensityWindow.SetWindowMaximum(max)
    intensityWindow.SetWindowMinimum(min)
    sitkImage = intensityWindow.Execute(sitkImage)
    return sitkImage


def readNii(path, ww, wc, isflipud=True, ):
    """
    读取和加载数据。如果图像是上下翻转的，就将其翻转过来
    @param path: 文件路径
    @param ww:窗宽窗位
    @param wc:窗宽窗位
    @param isflipud: 是否需要翻转
    @return: data
    """
    if type(path) == str:
        img = windowwc(sitk.ReadImage(path), ww, wc)
    else:
        img = windowwc(path, ww, wc)
    data = sitk.GetArrayFromImage(img)
    if isflipud:
        data = np.flip(data, 1)
    return data


class InferThread(QThread):
    """
    调用PyQt5.QtCore，建立一个任务线程类, 进行推理任务
    """
    # 收集推理失败的信号
    signal_infer_fail = pyqtSignal()
    # 传递推理结果
    signal_infer_result = pyqtSignal(np.ndarray)

    def __init__(self, sitkImage, model):
        super(InferThread, self).__init__()
        self.sitkImage = sitkImage
        self.model = model
        self.transforms = T.Compose([
            T.Resize(target_size=(512, 512)),
            T.Normalize()
        ])

    def run(self):
        """
            在启动线程后任务开始执行
        """
        try:
            data = readNii(self.sitkImage, 1500, -500)
            inferData = np.zeros_like(data)
            d, h, w = data.shape

            for i in range(d):
                img = data[i].copy()
                img = img.astype(np.float32)
                pre = self.nn_infer(self.model, img, self.transforms)
                inferData[i] = pre

            self.signal_infer_result.emit(inferData)
        except Exception as e:
            print(e)
            self.signal_infer_fail.emit()

    def nn_infer(self, model, im, transforms):
        """
        预测结果
        @param model: 模型参数
        @param im: 图像数据
        @param transforms:传入transforms方法
        @return: 预测结果pred
        """
        img, _ = transforms(im)
        img = paddle.to_tensor(img[np.newaxis, :])
        pre = infer.inference(model, img)
        pred = paddle.argmax(pre, axis=1).numpy().reshape((512, 512))
        return pred.astype('uint8')


class MainWindow(QWidget, Ui_Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.initUI()
        self.setWindowTitle('肺炎检测助手')
        # 打开nii文件选择器
        self.bn_open.clicked.connect(self.openFile)
        # 打开模型文件选择器
        self.bn_loadModel.clicked.connect(self.openModleFile)
        # 推理按钮
        self.bn_infer.clicked.connect(self.infer)
        self.bn_output.clicked.connect(self.outputFile)

        self.sitkImage = object()
        self.npImage = object()
        # 记录当前第几层
        self.currIndex = 0
        # 记录数据的最大层
        self.maxCurrIndex = 0
        # 记录数据的最小层，其实就是0
        self.minCurrIndex = 0
        self.baseFileName = ''
        self.isInferSucceed = False

        # 判断是否按下鼠标右键
        self.isRightPressed = bool(False)

        self.model = object()

        # 判断模型是否加载成功
        self.isModelReady = False

        # 宽宽窗位滑动条
        self.slider_ww.valueChanged.connect(self.resetWWWcAndShow)
        self.slider_wc.valueChanged.connect(self.resetWWWcAndShow)

        # 设置窗宽窗位文本框只能输入一定范围的整数
        intValidator = QIntValidator(self)
        intValidator.setRange(-2000, 2000)
        self.line_ww.setValidator(intValidator)
        self.line_ww.editingFinished.connect(self.resetWWWcAndShow)
        self.line_wc.setValidator(intValidator)
        self.line_wc.editingFinished.connect(self.resetWWWcAndShow)

        self.listWidget.itemDoubleClicked.connect(self.changeLayer)

    def initUI(self):
        try:
            # 定义展示的窗体及其初始的参数
            self.wwwcList = {'肺窗': [1700, -700]}

            self.line_ww.setText(str(1700))
            self.line_wc.setText(str(-700))

            self.slider_ww.setValue(1700)
            self.slider_wc.setValue(-700)
            self.ww = 1700
            self.wc = -700

            self.currWw = self.ww
            self.currWc = self.wc
        except Exception as e:
            print(e)

    def openFile(self):
        """
        打开医学影像文件选择器
        """
        try:
            filename, _ = QFileDialog.getOpenFileName(self,
                                                      "选取文件",
                                                      "./",
                                                      "Nii Files (*.nii);;Nii Files (*.nii.gz);;All Files (*)")
            if filename:
                # 清空列表
                self.listWidget.clear()
                self.isInferSucceed = False
                self.text_loadModel.setText("数据加载完毕")
                self.baseFileName = os.path.basename(filename).split('.')[0]
                self.sitkImage = sitk.ReadImage(filename)
                self.npImage = readNii(self.sitkImage, self.ww, self.wc)
                self.maxCurrIndex = self.npImage.shape[0]
                self.currIndex = int(self.maxCurrIndex / 2)
                self.showImg(self.npImage[self.currIndex])
        except Exception as e:
            print(e)

    def openModleFile(self):
        """
        打开模型文件选择器
        """
        filename, _ = QFileDialog.getOpenFileName(self, "选取文件", "./", "model Files (*.pdparams)")

        if filename:
            try:
                self.text_loadModel.setText(" ")
                num_class = int(2)
                self.model = BiSeNetV2(num_classes=num_class)
                para_state_dict = paddle.load(filename)
                self.model.set_dict(para_state_dict)
                self.text_loadModel.setText("模型加载完毕")
                self.isModelReady = True
            except Exception as e:
                self.text_loadModel.setText("模型加载失败")
                print(e)

    def wheelEvent(self, event):
        """
        定义鼠标滑轮事件
        """
        try:
            if self.maxCurrIndex != self.minCurrIndex:
                self.angle = event.angleDelta() / 8
                self.angleY = self.angle.y()
                if self.angleY > 0:
                    if self.currIndex < self.maxCurrIndex - 1:
                        self.currIndex += 1
                        if self.isInferSucceed:
                            self.showImg(self.drawContours(self.npImage, self.inferData, self.currIndex))
                        else:
                            self.showImg(self.npImage[self.currIndex])
                elif self.angleY < 0:
                    if self.currIndex != self.minCurrIndex:
                        self.currIndex -= 1
                        if self.isInferSucceed:
                            # self.npImage = self.drawContours(self.npImage, self.inferData)
                            self.showImg(self.drawContours(self.npImage, self.inferData, self.currIndex))
                        else:
                            self.showImg(self.npImage[self.currIndex])
        except Exception as e:
            print(e)

    def mousePressEvent(self, event):
        """
            重载鼠标单机事件
        """
        # 左键按下
        if event.buttons() == Qt.RightButton:
            # 左键按下(图片被点住),置Ture
            self.isRightPressed = True
            self.preMousePosition = event.pos()
        elif event.buttons() == Qt.MidButton | Qt.RightButton:
            self.isRightPressed = False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.isRightPressed = False

    def mouseMoveEvent(self, event):
        """
        重载一下鼠标移动事件
        """
        try:
            if self.maxCurrIndex != self.minCurrIndex:
                # 右键按下
                if self.isRightPressed:
                    # 鼠标当前位置-先前位置=单次偏移量
                    self.endMousePosition = event.pos() - self.preMousePosition
                    self.preMousePosition = event.pos()
                    ww = self.endMousePosition.x() + self.currWw
                    wc = self.endMousePosition.y() + self.currWc
                    if ww < -2000:
                        ww = -2000
                    elif ww > 2000:
                        ww = 2000
                    if wc < -2000:
                        wc = -2000
                    elif wc > 2000:
                        wc = 2000
                    self.currWw = ww
                    self.currWc = wc
                    self.slider_ww.setValue(int(self.currWw))
                    self.slider_wc.setValue(int(self.currWc))
                    self.line_ww.setText(str(self.currWw))
                    self.line_wc.setText(str(self.currWc))
                    self.resetWWWcAndShow()
        except Exception as e:
            print(e)

    def showImg(self, img):
        """
        显示图片
        @param img: 待显示的图片
        """
        try:
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
                img = np.concatenate((img, img, img), axis=-1).astype(np.uint8)
            elif img.ndim == 3:
                img = img.astype(np.uint8)
            qimage = QImage(img, img.shape[0], img.shape[1], img.shape[1] * 3, QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(qimage)

            self.canvas.setPixmap(pixmap_imgSrc)
        except Exception as e:
            print(e)

    def resetWWWcAndShow(self):
        """
        通过四个方式可以修改医学图像的窗宽窗位，每次修改后都会在界面呈现出来
        """
        if hasattr(self.sender(), "objectName"):
            objectName = self.sender().objectName()
        else:
            objectName = None
        try:
            if objectName == '':
                self.line_ww.setText(str(1700))
                self.line_wc.setText(str(-700))
                self.slider_ww.setValue(1700)
                self.slider_wc.setValue(-700)
                self.ww = 1700
                self.wc = -700
                self.currWw = self.ww
                self.currWc = self.wc
            if objectName == 'slider_ww' or objectName == 'slider_wc':
                self.currWw = self.slider_ww.value()
                self.currWc = self.slider_wc.value()
                self.line_ww.setText(str(self.currWw))
                self.line_wc.setText(str(self.currWc))
            elif objectName == 'line_ww' or objectName == 'line_wc':
                self.currWw = int(self.line_ww.text())
                self.currWc = int(self.line_wc.text())
                self.slider_ww.setValue(self.currWw)
                self.slider_wc.setValue(self.currWc)
            if self.maxCurrIndex != self.minCurrIndex:
                self.npImage = readNii(self.sitkImage, self.currWw, self.currWc)
                if self.isInferSucceed:
                    self.showImg(self.drawContours(self.npImage, self.inferData, self.currIndex))
                else:
                    self.showImg(self.npImage[self.currIndex])
        except Exception as e:
            print(e)

    def infer(self):
        """
            模型分割预测
        """
        if self.maxCurrIndex != self.minCurrIndex and self.isModelReady:
            self.bn_infer.setEnabled(True)
            # 创建推理线程
            self.infer_thread = InferThread(self.sitkImage, self.model)
            # 绑定推理失败的槽函数
            self.infer_thread.signal_infer_fail.connect(self.infer_fail)
            # 绑定推理成功的槽函数
            self.infer_thread.signal_infer_result.connect(self.infer_result)
            self.infer_thread.start()
            self.text_loadModel.setText("正在推理中······")

        else:
            QMessageBox.warning(self, "提示", "推理失败，推理前请先确保:\n1.加载模型\n2.加载数据", QMessageBox.Yes, QMessageBox.Yes)

    def infer_result(self, inferData):
        """
            分割模型预测成功后，结果保存在self.inferData
            @param inferData: 推理数据
        """
        # 推理成功，并显示结果
        try:
            self.inferData = inferData.astype(np.uint8)
            QMessageBox.information(self, "提示", "模型推理成功!", QMessageBox.Yes, QMessageBox.Yes)
            self.text_loadModel.setText("推理完毕")
            self.isInferSucceed = True
            self.infer_thread.quit()
            self.addListInfo(self.inferData)
            self.showImg(self.drawContours(self.npImage, self.inferData, self.currIndex))
        except Exception as e:
            print(e)

    def infer_fail(self):
        """
            如果推理失败，则报错
        """
        QMessageBox.warning(self, "警告", "模型推理失败!", QMessageBox.Yes, QMessageBox.Yes)

    def outputFile(self):
        """
            将保存模型预测结果为nii格式文件
        """
        try:
            if self.isInferSucceed:
                filedir = QFileDialog.getExistingDirectory(None, "文件保存", os.getcwd())
                if filedir:
                    # 读取nii文件时转换np文件时对数据进行上下翻转，再输入模型推理，保存回nii文件时要翻转回来。
                    self.inferData = np.flip(self.inferData, 1)
                    pre_sitkImage = sitk.GetImageFromArray(self.inferData)
                    pre_sitkImage.CopyInformation(self.sitkImage)
                    pre_sitkImage = sitk.Cast(pre_sitkImage, sitk.sitkUInt8)
                    save_path = os.path.join(filedir, self.baseFileName + '_mask.nii')
                    sitk.WriteImage(pre_sitkImage, save_path)
            else:
                QMessageBox.warning(self, "警告", "无进行过推理，无法保存！", QMessageBox.Yes, QMessageBox.Yes)
        except Exception as e:
            print(e)

    def drawContours(self, npImage, inferData, currIndex):
        """
            通过OpenCV将mask转换成轮廓绘制在原图上
            @param npImage: 图像数据
            @param inferData: 模型推理的结果
            @param currIndex: 层数序号
            @return: 绘制轮廓后的图片
        """
        img = npImage[currIndex]
        img = np.expand_dims(img, axis=2)
        img = np.concatenate((img, img, img), axis=-1).astype(np.uint8)
        ret, thresh = cv2.threshold(inferData[currIndex], 0, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, kernel=np.ones((5, 5), np.uint8), iterations=1)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        # 绘制轮廓过程
        img = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

        return img

    def addListInfo(self, inferData):
        """
            增加列表信息
            @param inferData:模型推理的结果
        """
        self.listWidget.clear()
        d, h, w = inferData.shape
        result = {}
        for i in range(d):
            img = inferData[i]
            if np.sum(img > 0) != 0:
                result[str(i)] = np.sum(img > 0)

        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        for key, value in result:
            self.listWidget.addItem("层 " + str(int(key) + 1))

    def changeLayer(self, item):
        """
        点击列表自动展示该层
        @param item: 控制层数的对象
        """
        self.currIndex = int(item.text().split(' ')[1]) - 1
        if self.isInferSucceed:
            self.showImg(self.drawContours(self.npImage, self.inferData, self.currIndex))
        else:
            self.showImg(self.npImage[self.currIndex])


def predict(img_path):
    img = cv2.imread(img_path)  # 读取图片
    img = cv2.resize(img, (224, 224))  # 调整图片尺寸

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 把图片BGR变成RGB

    img = np.transpose(img, (2, 0, 1))  # 调整维度将HWC - CHW
    img = np.expand_dims(img, 0)  # 添加一个维度 就是batch维度
    img = img.astype(np.float32)  # 格式转成float32
    img /= 255
    ort_session = ort.InferenceSession("H:\\tongxinxitong\\home.onnx", providers=torch.device("cpu"))
    # 调用onnxruntime run函数进行模型推理
    outputs = ort_session.run(
        None,
        {"image": img},
    )
    # outputs的输出类型为list类型，所以要先将list转换成numpy再转换成torch
    outputs1 = torch.from_numpy(np.array(outputs))
    # 通过softmax进行最后分数的计算
    value = float(torch.max(torch.softmax(outputs1[0], dim=1)))
    # outputs_softmax = torch.softmax(outputs1[0], dim=1).numpy()[:, 0].tolist()[0]
    index = np.argmax(np.array(outputs)) + 1

    return value, index


class CamShow(QMainWindow, Class_Ui):
    def __init__(self, parent=None):
        super(CamShow, self).__init__(parent)
        self.setupUi(self)
        self.upload.clicked.connect(self.loadImage)
        self.interference.clicked.connect(self.predict_label)

    # 打开文件功能
    def loadImage(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, '请选择图片', '.', '图像文件(*.jpg *.jpeg *.png)')
        self.result.setText(None)
        self.score.setText(None)
        if self.fname:
            # print(self.fname)
            # self.Infolabel.setText("文件打开成功\n" + self.fname)
            # self.Imglabel.set
            # self.result.setText(self.fname)
            jpg = QtGui.QPixmap(self.fname).scaled(self.imglabel.width(), self.imglabel.height())

            print(jpg)
            self.imglabel.setPixmap(jpg)
        else:
            print("打开文件失败")
            # self.Infolabel.setText("打开文件失败")

    def predict_label(self):
        # 开启线程
        # self.result.setText(self.fname)
        if not self.fname:
            self.result.setText("为空退出")
        else:
            value, index = predict(self.fname)
            if index == 1:
                self.result.setText("新冠")
            else:
                if index == 2:
                    self.result.setText("病毒性肺炎")
                else:
                    if index == 3:
                        self.result.setText("正常")
            self.score.setText(str(value))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    a = AUi()
    a.show()
    b = CamShow()
    c = MainWindow()
    # button是你定义的按钮
    a.Xray.clicked.connect(
        lambda: {a.close(), b.show()}
    )
    a.CT.clicked.connect(
        lambda: {a.close(), c.show()}
    )
    b.xrayback.clicked.connect(
        lambda: {b.close(), a.show()}
    )
    c.CTback.clicked.connect(
        lambda: {c.close(), a.show()}
    )
    a.setWindowIcon(QIcon("H:/tongxinxitong/aid90-ujb0o-001.ico"))
    b.setWindowIcon(QIcon("H:/tongxinxitong/aid90-ujb0o-001.ico"))
    c.setWindowIcon(QIcon("H:/tongxinxitong/aid90-ujb0o-001.ico"))
    sys.exit(app.exec_())
