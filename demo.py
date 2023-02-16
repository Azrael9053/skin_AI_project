import re
import sys
import binascii
import time
from PyQt5.QtCore import QTimer, QUrl
from PyQt5.QtWidgets import *
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import *
from UI import Ui_MainWindow
from PyQt5.QtCore import QDate

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        # 设置实例
        self.CreateItems()
        # 设置信号与槽
        self.CreateSignalSlot()

    def CreateItems(self):
        # Qt 串口类
        self.com = QSerialPort()
        # Qt 定时器类
        self.timer = QTimer(self)  # 初始化一个定时器
        # self.timer.timeout.connect(self.ShowTime)  # 计时结束调用operate()方法
        self.timer.start(100)  # 设置计时间隔 100ms 并启动