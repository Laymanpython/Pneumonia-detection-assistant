# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1314, 816)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(440, 70, 671, 131))
        font = QtGui.QFont()
        font.setPointSize(48)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(410, 210, 681, 51))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(420, 280, 631, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(520, 340, 381, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(580, 390, 391, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.Xray = QtWidgets.QPushButton(Form)
        self.Xray.setGeometry(QtCore.QRect(270, 490, 231, 101))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Xray.setFont(font)
        self.Xray.setObjectName("Xray")
        self.CT = QtWidgets.QPushButton(Form)
        self.CT.setGeometry(QtCore.QRect(770, 490, 211, 101))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.CT.setFont(font)
        self.CT.setObjectName("CT")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "肺炎识别助手                                              "))
        self.label_2.setText(_translate("Form", "山东大学20级通信三班第五小组"))
        self.label_3.setText(_translate("Form", "李鑫、李欣竹、王籽予、李明晓"))
        self.label_4.setText(_translate("Form", "指导老师：郑来波"))
        self.label_5.setText(_translate("Form", "2023.6.8"))
        self.Xray.setText(_translate("Form", "Xray肺炎分类"))
        self.CT.setText(_translate("Form", "CT病灶分割"))