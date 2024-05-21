import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.image import resize

class PaintWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.paint_size = 280
        self.setGeometry(100, 100, self.paint_size, self.paint_size)

        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        
        self.lastPoint = QPoint()
        self.paintColor = Qt.white
        self.penWidth = 12
        self.bars = []
        self.model = load_model('models/model.keras')

    def update_values(self):
        image = self.qimage_to_numpy()
        image = image[:,:,0]
        image = image.reshape((1, self.paint_size, self.paint_size, 1))
        image = resize(image, (28, 28))
        image = image / 255

        values = self.model.predict(image, verbose=0)[0]
        self.update_bars(values)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        i = 0
        x_offset = 20
        max_bar_height = 220
        for x, y, height in self.bars:
            painter.setPen(QPen(Qt.black))
            color = 255 * height
            bar_height = -int(height * max_bar_height)
            qt_color = QColor(255 - int(color), int(color), 0)
            painter.drawRect(x, y, x_offset, bar_height)
            painter.fillRect(x, y, x_offset, bar_height, qt_color)
            painter.drawRect(x, y ,x_offset, 10)
            painter.fillRect(x, y ,x_offset, 10, Qt.black)

            painter.drawText(x, y + bar_height - 2, str(int(height * 100)) + "%")
            painter.setPen(QPen(qt_color))
            
            painter.drawText(x + 5, y + 10, str(i))
            i += 1
            

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            tempImage = QImage(self.image)  
            painter = QPainter(tempImage)
            painter.setPen(QPen(self.paintColor, self.penWidth, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            painter.end()
            self.image = tempImage  
            self.update()

    def qimage_to_numpy(self):
        size = self.image.size()
        s = self.image.bits().asstring(size.width() * size.height() * self.image.depth() // 8)  
        arr = np.frombuffer(s, dtype=np.uint8).reshape((size.height(), size.width(), self.image.depth() // 8))

        return arr

    def clearCanvas(self):
        self.image.fill(Qt.black)
        self.update()

    def update_bars(self, arr):
        self.bars = []
        bar_start_pos_x = 295
        bar_start_pos_y = 235
        bar_offset = 25
        for i in range(10):
            x = bar_start_pos_x + i * bar_offset
            y = bar_start_pos_y
            height = arr[i] 
            self.bars.append((x, y, height))

        self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    paintWidget = PaintWidget()
    timer = QTimer()
    timer.timeout.connect(paintWidget.update_values)
    timer.start(50)  


    clearButton = QPushButton('Clear', paintWidget)
    clearButton.clicked.connect(paintWidget.clearCanvas)


    layout = QVBoxLayout()
    layout.addWidget(paintWidget)
    layout.addWidget(clearButton)

    window = QWidget()
    window.setLayout(layout)
    window.setFixedSize(290*2, 300)
    window.setWindowTitle('Paint Mnist')
    window.show()

    sys.exit(app.exec_())
