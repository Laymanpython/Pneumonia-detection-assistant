# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FormUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1286, 794)
        Form.setMaximumSize(QtCore.QSize(1286, 794))
        self.gridLayout_4 = QtWidgets.QGridLayout(Form)
        self.gridLayout_4.setObjectName("gridLayout_4")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem, 1, 0, 1, 1)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.canvas = QtWidgets.QLabel(Form)
        self.canvas.setMinimumSize(QtCore.QSize(512, 512))
        self.canvas.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.canvas.setFont(font)
        self.canvas.setMouseTracking(False)
        self.canvas.setAutoFillBackground(False)
        self.canvas.setFrameShadow(QtWidgets.QFrame.Plain)
        self.canvas.setLineWidth(1)
        self.canvas.setMidLineWidth(0)
        self.canvas.setText("")
        self.canvas.setObjectName("canvas")
        self.gridLayout_5.addWidget(self.canvas, 0, 1, 1, 1)
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setMinimumSize(QtCore.QSize(0, 0))
        self.groupBox_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_2.setFlat(True)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.line_ww = QtWidgets.QLineEdit(self.groupBox_2)
        self.line_ww.setMaximumSize(QtCore.QSize(100, 16777215))
        self.line_ww.setObjectName("line_ww")
        self.horizontalLayout.addWidget(self.line_ww)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.slider_ww = QtWidgets.QSlider(self.groupBox_2)
        self.slider_ww.setMinimum(-2000)
        self.slider_ww.setMaximum(2000)
        self.slider_ww.setOrientation(QtCore.Qt.Horizontal)
        self.slider_ww.setObjectName("slider_ww")
        self.verticalLayout_2.addWidget(self.slider_ww)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.line_wc = QtWidgets.QLineEdit(self.groupBox_2)
        self.line_wc.setMaximumSize(QtCore.QSize(100, 16777215))
        self.line_wc.setObjectName("line_wc")
        self.horizontalLayout_2.addWidget(self.line_wc)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.slider_wc = QtWidgets.QSlider(self.groupBox_2)
        self.slider_wc.setMinimum(-2000)
        self.slider_wc.setMaximum(2000)
        self.slider_wc.setOrientation(QtCore.Qt.Horizontal)
        self.slider_wc.setObjectName("slider_wc")
        self.verticalLayout_2.addWidget(self.slider_wc)
        self.gridLayout_9.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        self.gridLayout_6.addWidget(self.groupBox_2, 2, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setEnabled(True)
        self.groupBox.setMinimumSize(QtCore.QSize(250, 0))
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.groupBox.setInputMethodHints(QtCore.Qt.ImhNone)
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setFlat(True)
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.bn_open = QtWidgets.QPushButton(self.groupBox)
        self.bn_open.setObjectName("bn_open")
        self.gridLayout.addWidget(self.bn_open, 0, 0, 1, 1)
        self.bn_output = QtWidgets.QPushButton(self.groupBox)
        self.bn_output.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.bn_output.setObjectName("bn_output")
        self.gridLayout.addWidget(self.bn_output, 4, 0, 1, 1)
        self.bn_infer = QtWidgets.QPushButton(self.groupBox)
        self.bn_infer.setObjectName("bn_infer")
        self.gridLayout.addWidget(self.bn_infer, 2, 0, 1, 1)
        self.bn_loadModel = QtWidgets.QPushButton(self.groupBox)
        self.bn_loadModel.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.bn_loadModel.setObjectName("bn_loadModel")
        self.gridLayout.addWidget(self.bn_loadModel, 1, 0, 1, 1)
        self.text_loadModel = QtWidgets.QLabel(self.groupBox)
        self.text_loadModel.setText("")
        self.text_loadModel.setObjectName("text_loadModel")
        self.gridLayout.addWidget(self.text_loadModel, 5, 0, 1, 1)
        self.gridLayout_6.addWidget(self.groupBox, 0, 0, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setMinimumSize(QtCore.QSize(0, 0))
        self.groupBox_3.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.listWidget = QtWidgets.QListWidget(self.groupBox_3)
        self.listWidget.setObjectName("listWidget")
        self.gridLayout_2.addWidget(self.listWidget, 0, 0, 1, 1)
        self.gridLayout_6.addWidget(self.groupBox_3, 3, 0, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_6, 0, 2, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_8, 0, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(225, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem1, 0, 2, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_5, 1, 1, 1, 1)
        self.CTback = QtWidgets.QPushButton(Form)
        self.CTback.setObjectName("CTback")
        self.gridLayout_4.addWidget(self.CTback, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox_2.setTitle(_translate("Form", "肺窗"))
        self.label.setText(_translate("Form", "窗宽："))
        self.label_5.setText(_translate("Form", "窗位："))
        self.groupBox.setTitle(_translate("Form", "模型:BiSeNetV2"))
        self.bn_open.setText(_translate("Form", "加载数据"))
        self.bn_output.setText(_translate("Form", "导出Mask"))
        self.bn_infer.setText(_translate("Form", "推理"))
        self.bn_loadModel.setText(_translate("Form", "加载模型"))
        self.groupBox_3.setTitle(_translate("Form", "位置"))
        self.CTback.setText(_translate("Form", "返回"))
