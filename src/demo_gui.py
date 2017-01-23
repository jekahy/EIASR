import sys
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from scipy.misc import imread, imsave

from canny import gradient, thin_nonmaximum, thin_hysteresis
from hough import hough_learn, hough_detect
import math

import cv2

class Image():

	def __init__(self, img_path):
		self.path = img_path
		self.original = imread(img_path, mode='RGB')
		self.gray = imread(img_path, flatten=True, mode='L')
		self.computed = False

	@property
	def name(self):
		if not hasattr(self,"_name"):
			self._name = self.path.split('/')[-1]
		return self._name

	@property
	def gradient(self):
		if hasattr(self, "_gradient"):
			return self._gradient

	@property
	def thinned_nmax(self):
		if hasattr(self,"_thinned_nmax"):
			return self._thinned_nmax

	@property
	def thinned_hyst(self):
		if hasattr(self,"_thinned_hyst"):
			return self._thinned_hyst
	
	def visualImages(self):
		images = [self.original]
		if self.computed:
			images.extend([self.gradient.magnitudes, self.thinned_nmax.magnitudes, self.thinned_hyst.magnitudes])
		if hasattr(self, "ght_res"):
			images.append(self.ght_res)
		if hasattr(self, "ght_overlayed"):
			imsave('accumulator.png',self.ght_res)
			imsave('result.png',self.ght_overlayed)
			images.append(self.ght_overlayed)
		return [imgToQImg(im) for im in images]

	def compute(self):

		if not self.computed:
			self._gradient = gradient(self.gray)
			self._thinned_nmax =  thin_nonmaximum(self.gradient)
			self._thinned_hyst =  thin_hysteresis(self.thinned_nmax)
			self.computed = True


class HoughShapeDetector():

	def __init__(self, template):
		self.template = template
		# self.progress_func = progress_func
		self.train()

	def train(self):
		self.template.compute()
		self.rtable = hough_learn(self.template.thinned_hyst)
		pts = [item for sublist in self.rtable.values() for item in sublist]
		pts = list(set(pts))
		pts = [list(pt) for pt in pts]
		self.shapePts = pts


	def detect(self, img, progress_func):
		img.compute()
		self.result = hough_detect(self.rtable, img.thinned_hyst, progress_func)
		img.ght_res = np.sum(self.result.accumulator, axis=(0, 1))
		img.ght_overlayed = self.createTempateOverlay(img)

	def createTempateOverlay(self, img):
		
		shapes = []
                w, h = img.gray.shape
                
		for cand in self.result[1]:

			scale, angle, cx, cy = cand
			angle *= 180/np.pi
                        angle = (angle + 180) % 360
			tmpPoints = np.asarray(self.shapePts, dtype=np.float64)
			tmpPoints *= scale
			tmpPoints = tmpPoints.astype(np.int64)
			
			tmpPoints[:,0] += cx
			tmpPoints[:,1] += cy

			tmpPoints = [rotatePoint((cx,cy),p,angle) for p in tmpPoints]
                        tmpPoints = [
                            (x, y) for x, y in tmpPoints
                            if 0 <= x < w and 0 <= y < h
                        ]
			shapes.append(tmpPoints)

		overlay = np.zeros(img.gray.shape)
		marked_img = img.original.copy()
		for shape in shapes:
			
			shape = np.asarray(shape, dtype = np.int)
			marked_img[shape[:,0], shape[:,1]] = (0,255,0)
	
		return marked_img



def imgToQImg(img):

	if img is None:
		return

	if len(img.shape) == 3:
		height, width, channel = img.shape
		bytesPerLine = 3 * width
		return QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
	else:
		height, width = img.shape
		bytesPerLine =  width
		mn = np.min(img)
		mx = np.max(img)
		img = np.uint8((img - mn)*255/(mx - mn))
		return QImage(img, width, height, bytesPerLine, QImage.Format_Indexed8)


class Canny(QWidget):
    
	def __init__(self):

		super(Canny, self).__init__()
		self.images = []
		self.initUI()

	def initUI(self):               
        
		self.setGeometry(0, 0, 780, 530)
		self.setWindowTitle('Jagermeister detector 3000')   
		self.setupImgViews()
		self.setupButtons()
		self.setupTable()
		# self.setupModeBox()
		self.setupProgressView()
		self.show()


	def setupImgViews(self):
		
		self.ig = ImageGallery(parent=self, imagesPerRow=7)
		self.ig.setGeometry(0, 410, 700, 125)

		self.main_img_view = QLabel(self)
		self.main_img_view.setStyleSheet('background-color: gray')
		self.main_img_view.setGeometry(10, 10, 400, 400)

		self.templ_img_view = QLabel(self)
		self.templ_img_view.setStyleSheet('background-color: gray')
		self.templ_img_view.setGeometry(420, 220, 190, 190)


	def setupButtons(self):

		load_imgs_but = QPushButton('Load images', self)
		load_imgs_but.clicked.connect(self.loadImages)
		load_imgs_but.setMaximumWidth(110)
		load_imgs_but.move(620, 315)

		load_templ_but = QPushButton('Load template', self)
		load_templ_but.clicked.connect(self.loadTemplate)
		load_templ_but.setMaximumWidth(110)
		load_templ_but.move(620, 350)

		exit_but = QPushButton('Quit', self)
		exit_but.clicked.connect(QCoreApplication.instance().quit)
		exit_but.move(620, 385)

	def setupTable(self):

		self.table = QTableWidget(self)
		self.table.resize(350, 200)
		self.table.setColumnCount(1)
		self.table.move(420, 10)
		self.table.setHorizontalHeaderItem(0, QTableWidgetItem("Image"))
		self.table.setHorizontalHeaderItem(1, QTableWidgetItem("Result"))
		self.table.setColumnWidth(0, 350)
		self.table.cellDoubleClicked.connect(self.double_clicked_cell)
		self.table.cellClicked.connect(self.clicked_cell)
		self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
		self.table.setSelectionBehavior(QAbstractItemView.SelectRows)

	def setupModeBox(self):
		rb = RadioBox('Detection mode:',['Fast', 'Normal', 'Precise'], parent=self)
		rb.setGeometry(620, 220, 150, 100)


	def setupProgressView(self):
		self.progressView = QWidget(self)
		self.progressView.setGeometry(self.geometry())
		self.progressView.setVisible(False)
		self.progressView.setStyleSheet("background-color: rgba(255, 255, 255, 100)")
		pr_bar = QProgressBar(self.progressView)
		
		pr_bar.setGeometry(0, 0, 250, 20)

		g  = pr_bar.geometry()
		g.moveCenter(QPoint(self.width()/2, self.height()/2))
		pr_bar.setGeometry(g)
		self.pr_bar = pr_bar


		pr_label = QLabel(self.progressView)
		pr_label.setText('Detecting template! Please wait =)')
		pr_label_geometry = pr_bar.geometry()
		pr_label_geometry.moveCenter(QPoint(self.width()/2, (self.height()/2)-50))
		pr_label.setGeometry(pr_label_geometry)

	def updateTable(self):
		self.table.setRowCount(len(self.images))
		for idx, img in enumerate(self.images):
			self.table.setItem(idx,0, QTableWidgetItem(img.name))

	def refreshImgViews(self):
		images = self.activeImg.visualImages()

		
		self.ig.populate(images)
		self.ig.show()

	def setImgForImgView(self, qimg, img_view):
		pixm = QPixmap(qimg)
		pixm = pixm.scaled(img_view.size(), Qt.KeepAspectRatio)
		img_view.setPixmap(pixm)

	
	def updateDetectionMode(self, mode):
		print "detection mode is set to", mode
	

	def double_clicked_cell(self, row, column):

		self.images[row].compute()
		if hasattr(self, "detector"):
			
			self.progressView.setVisible(True)

			self.workerThread = QThread()
			workerObject = Worker(self.detector.detect, (self.images[row],))
			workerObject.moveToThread(self.workerThread)
			self.workerThread.started.connect(workerObject.run)
			workerObject.finished.connect(self.finishDetection)
			workerObject.madeProgress.connect(self.pr_bar.setValue)
			self.workerThread.start()

		self.refreshImgViews()

	def finishDetection(self):
		self.refreshImgViews()
		self.progressView.setVisible(False)
		self.workerThread.quit()

	def clicked_cell(self, row, column):

		self.activeImg = self.images[row]
		self.setImgForImgView(imgToQImg(self.activeImg.original), self.main_img_view)
		self.refreshImgViews()


	def loadImages(self):
		
		fileDialog = QFileDialog(self)
		fileDialog.setFileMode(QFileDialog.ExistingFiles)
		fileDialog.setNameFilters(['Images (*.jpg *.png *.jpeg)'])
		fileDialog.show()

		if fileDialog.exec_():
			existing_imgs = [img.path for img in self.images]
			filenames =  fileDialog.selectedFiles() 
			images = [Image(str(f_n)) for f_n in filenames if f_n not in existing_imgs]
			self.images.extend(images)
			self.updateTable()

	def loadTemplate(self):
		fileDialog = QFileDialog(self)
		fileDialog.setFileMode(QFileDialog.ExistingFiles)
		fileDialog.setNameFilters(['Images (*.jpg *.png)'])
		fileDialog.show()

		if fileDialog.exec_():
			filename =  fileDialog.selectedFiles()[0]
			template = Image(str(filename))
			self.detector = HoughShapeDetector(template)
			self.setImgForImgView(imgToQImg(template.original), self.templ_img_view)


class RadioBox(QGroupBox):
    
	def __init__(self, title,  elements, parent=None):
		super(QWidget, self).__init__(parent)
		self.box = QVBoxLayout()
		self.group = QButtonGroup()
		self.setTitle(title)
		self.buttons = []
		self.createRadioButtons(elements)
		self.buttons[1].setChecked(True)
		self.setLayout(self.box)


	def createRadioButtons(self, titles):
		for idx, title in enumerate(titles):
			rb = QRadioButton(title)
			rb.toggled.connect(self.on_radio_button_toggled)
			self.box.addWidget(rb)
			self.group.addButton(rb, idx)
			self.buttons.append(rb)
                        

	def on_radio_button_toggled(self, event):

		radiobutton = self.sender()

		if radiobutton.isChecked():
			self.parentWidget().updateDetectionMode(radiobutton.text())
			
		
    
class ImageGallery(QWidget):
    
	def __init__(self, imagesPerRow=4,itemSize=80, parent=None):
		super(QWidget, self).__init__(parent)
                self.imagesPerRow = imagesPerRow
                self.itemSize = itemSize
		self.grid = QGridLayout()
		self.setLayout(self.grid)
		self.createLabels()
		self.pics = None
		
		self.mousePressEvent = self.changeActiveThumb


        def createLabels(self):
                row = col = 0
                for i in range(6):
                        label = ImageLabel(self)
                        self.grid.addWidget(label, row, col)
                        col +=1
			if col % self.imagesPerRow == 0:
				row += 1
				col = 0
                        
	def populate(self, pics, flags=Qt.KeepAspectRatio):
		row = col = 0
		self.pics = pics
		self.clearAllLabels()
		for idx, pic in enumerate(pics):
			label = self.grid.itemAt(idx).widget()
			pixmap = QPixmap(pic)
			pixmap = pixmap.scaled(label.size(), flags)
			label.setPixmap(pixmap)
			col +=1
			if col % self.imagesPerRow == 0:
				row += 1
				col = 0
		self.deactivateAll()
		self.grid.itemAt(0).widget().active = True

	def changeActiveThumb(self, event):

		clickedLab = self.childAt(event.pos())
		clickedIdx = self.grid.indexOf(clickedLab)

		if clickedLab is None or self.pics is None or clickedIdx >= len(self.pics):
			return
		if not clickedLab.active:
			clickedLab.active = True
			self.activeLabIdx = clickedIdx
			items = [self.grid.itemAt(i).widget() for i in range(self.grid.count())]
			self.deactivateAll()
			clickedLab.active = True
			
		self.parentWidget().setImgForImgView(self.pics[clickedIdx], self.parentWidget().main_img_view)
                
	def deactivateAll(self):
		items = [self.grid.itemAt(i).widget() for i in range(self.grid.count())]
		for i in items:
			i.active = False
                                   
	def clearAllLabels(self):
                items = [self.grid.itemAt(i).widget() for i in range(self.grid.count())]
                for lab in items:
			lab.clear()

class ImageLabel(QLabel):
	"""docstring for ImageLabel"""
	def __init__(self, parent=None):
		super(QLabel, self).__init__(parent)
		self._active = False

	@property
	def active(self):
		return self._active

	@active.setter
	def active(self, value):
		self._active = value
		if value:
			self.setStyleSheet('border: 2px solid blue')
		else:
			self.setStyleSheet('border:None')

class Worker(QObject):
	"""docstring for Worker"""
	finished = pyqtSignal()
	madeProgress = pyqtSignal([int])

	def __init__(self, task, args):
		super(Worker, self).__init__()
		self.task = task
		self.t_args = args

	def run(self):
		self.task(*self.t_args, progress_func=self.progressChanged)
		self.finished.emit()

	def progressChanged(self, progress):
		self.madeProgress.emit(progress)

def rotatePoint(centerPoint,point,angle):
	"""Rotates a point around another centerPoint. Angle is in degrees.
	Rotation is counter-clockwise"""
	angle = math.radians(angle)
	temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
	temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
	temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
	return temp_point
		
def main():
    
    app = QApplication(sys.argv)
    ex = Canny()
    sys.exit(app.exec_())
	

if __name__ == '__main__':
	main()
