import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import random
import torch
import net


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.image = QImage(QSize(400, 400), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.brush_size = 20
        self.brush_color = Qt.black
        self.last_point = QPoint()
        self.loaded_model = None
        self.n = random.randrange(0, 10)
        self.txt = ()
        self.fname ='/Users/ijimin/Desktop/지민/pycharm/pyqt test/MNIST_CNNmodel_99%_state.pth'
        self.arr = np.zeros((28,28))
        self.predicted = 0
        self.initUI()

    def initUI(self):
        #메뉴바_ load, save, clear
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('File')

        load_model_action = QAction('Load model', self)
        load_model_action.setShortcut('Ctrl+L')
        load_model_action.triggered.connect(self.load_model)

        save_action = QAction('Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save)

        clear_action = QAction('Clear', self)
        clear_action.setShortcut('Ctrl+C')
        clear_action.triggered.connect(self.clear)

        filemenu.addAction(load_model_action)
        filemenu.addAction(save_action)
        filemenu.addAction(clear_action)

        self.statusbar = self.statusBar()

        #yes, no button 만들기
        self.formatbar = QToolBar(self)
        self.addToolBar(Qt.BottomToolBarArea, self.formatbar)

        btn_1 = QToolButton(self)
        btn_2 = QToolButton(self)

        btn_1.setText('Right')
        btn_1.setCheckable(False)
        btn_2.setText('Wrong')
        btn_2.setCheckable(False)

        btn_1.clicked.connect(self.clear)
        btn_1.clicked.connect(self.empty_or_not)

        btn_2.clicked.connect(self.save_wrong)
        btn_2.clicked.connect(self.clear)


        self.formatbar.addWidget(btn_1)
        self.formatbar.addWidget(btn_2)

        #숫자 지정해주기
        self.num_label = QLabel(f'{self.n} 써주세요', self)
        self.num_label.move(155,0)

        if self.fname:
            self.loaded_model=torch.load(self.fname)
            self.statusbar.showMessage('Model loaded')

        self.setWindowTitle('MNIST_classifier')
        self.setGeometry(300, 300, 400, 400)
        self.show()

    def on_click(self):
        self.n = random.randrange(0,10)
        self.num_label.setText(f'{self.n} 써주세요')

    def paintEvent(self, e):
        canvas = QPainter(self)
        canvas.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = e.pos()

    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, e.pos())
            self.last_point = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = False
            self.arr = np.zeros((28, 28))
            for i in range(28):
                for j in range(28):
                    self.arr[j, i] = 1 - self.image.scaled(28, 28).pixelColor(i, j).getRgb()[0] / 255.0
            self.arr = self.arr.reshape(-1, 28, 28)
            tensor = torch.from_numpy(self.arr).float()
            x = tensor.unsqueeze(dim=0)

            if self.loaded_model:
                net.model.load_state_dict(self.loaded_model)
                net.model.eval()

                with torch.no_grad():
                    pred = net.model(x)
                    ans = torch.argmax(pred).item()
                    self.predicted = net.classes[ans]
                    self.statusbar.showMessage("추정 값은" + self.predicted + "입니다.")

    def load_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Model', '')
        if fname:
            self.loaded_model = torch.load(fname)
            self.statusbar.showMessage('Model loaded.')

    def save(self):
        fpath, _ = QFileDialog.getSaveFileName(self, 'Save Image', '',
                                               "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        if fpath:
            self.image.scaled(28, 28).save(fpath)
    def save_wrong(self):
        if np.any(self.arr.flatten()>0.) == False:
            print('write something')
            self.statusbar.showMessage('숫자를 쓴 후 눌러주세요.')
            # method empty_or_not 안에 있는 거랑 진짜 똑같은데 왜 안될까

        else:
            test_num_string = self.n
            input_arr = self.arr
            output_num = int(self.predicted)
            s = (test_num_string,output_num,input_arr)
            self.txt += s
            self.on_click()
    def empty_or_not(self):
        if np.any(self.arr.flatten()>0.) == False:
            print('write something')
            self.statusbar.showMessage('숫자를 쓴 후 눌러주세요.')
        else:
            self.on_click()
    def clear(self):
        self.image.fill(Qt.white)
        self.update()
        self.statusbar.clearMessage()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
