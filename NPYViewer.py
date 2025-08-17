#!/usr/bin/env python3
import os.path
import sys
from PySide6.QtWidgets import *
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import *
from pathlib import Path
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.io import savemat
import networkx as nx
from typing import Literal
from os.path import exists as FileExists
from xml.etree.ElementTree import parse, Element

# from numpy import asarray
# from numpy import savetxt

class NPYfile():
    def __init__(self, data, filename):
        self.data = data
        self.filename = filename

    def __str__(self):
        global zLang
        Fields: str = zLang.Get('Label-NPY_Info-Infos').split('|-|')
        if hasattr(self.data, 'dtype'):
            return Fields[0] + str(self.filename) + f" \n{Fields[1]}" + str(self.data.dtype) + f"\n{Fields[2]}" + str(
                self.data.shape)
        else:
            return Fields[0] + str(self.filename) + f" \n{Fields[1]}" + f"\n{Fields[2]}" + str(self.data.shape)

class Lang():
    def __init__(self, LangType: str):
        self.LangType: str = LangType
        self.Translation: Element | None = None
        self.TranslationsPath: str = r'Lang\Translations.xml'

        self.LoadTranslation(LangType=LangType)

    def __call__(self, Key: str = ''):
        return self.Get(Key)

    def LoadTranslation(self, LangType: str):
        if not FileExists(self.TranslationsPath): raise FileExistsError(f'Translation File Not Exist In \'{self.TranslationsPath}\'.')
        for LangElem in parse(self.TranslationsPath).getroot().findall('language'):
            if LangElem.get('code') == LangType: self.Translation = LangElem; break
        else: raise KeyError(f'No Such TransType -> \'{LangType}\'.')

    def Get(self, Key: str = ''):
        return zElement.text if not (zElement:=self.Translation.find(Key)) is None else ''

class MainApp(QMainWindow):
    def __init__(self, App: QApplication, LangType: Literal['zh-CN', 'en-US'] = 'zh-CN'):
        super().__init__()
        global zLang
        zLang = Lang(LangType=LangType)

        self.App: QApplication = App
        self.Lang: Lang = zLang
        self.npyfile = None

        self.initUI()
        self.initMenu_Tittle()
        self.initMenu_Table()

    def initUI(self):
        # Main Win Config
        # Win Tittle
        self.setWindowTitle(self.Lang.Get("Win_Tittle") + " v."+version)

        # Win Label
        self.infoLb = QLabel(self.Lang.Get("Label-NPY_Info"))

        # Win Table
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(10)
        self.tableWidget.setColumnCount(10)
        self.tableWidget_ColHeader = QHeaderView(Qt.Horizontal, self.tableWidget)
        self.tableWidget_RowHeader = QHeaderView(Qt.Vertical, self.tableWidget)
        # table selection change
        # self.tableWidget.doubleClicked.connect(self.on_click)

        # Win Geometry
        Screen = self.App.primaryScreen()
        ScreenSize = Screen.size().toTuple()
        DPIRatio = Screen.devicePixelRatio()
        Width: int = int(ScreenSize[0] * 0.8)
        Height: int = int(ScreenSize[1] * Width/ScreenSize[0])
        PosX: int = int((ScreenSize[0] - Width) / 2)
        PosY: int = int((ScreenSize[1] - Height) / 2)
        self.setGeometry(PosX, PosY, Width, Height)

        # Layout
        self.widget = QWidget(self)
        layout = QGridLayout()
        layout.addWidget(self.infoLb)
        layout.addWidget(self.tableWidget)
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)
        # self.tableWidget.setItesetTextAlignmentmDelegate(AlignDelegate())
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setWindowIcon(QIcon('npyviewer_128x128.png'))

        self.show()

    def initMenu_Tittle(self):
        if True: # Menu - File
            exitAct = QAction(QIcon('exit.png'), f'&{self.Lang.Get("Menu-File-Exit")}', self)
            exitAct.setShortcut('Ctrl+Shift+Q')
            exitAct.setStatusTip(self.Lang.Get("Menu-File-Exit-Tips"))
            exitAct.triggered.connect(QApplication.instance().quit)

            openAct = QAction(QIcon('Open.png'), f'&{self.Lang.Get("Menu-File-Open")}', self)
            openAct.setShortcut('Ctrl+O')
            openAct.setStatusTip(self.Lang.Get("Menu-File-Open-Tips"))
            openAct.triggered.connect(self.openNPY)
            self.statusBar()

            saveAct = QAction(QIcon('Save.png'), f'&{self.Lang.Get("Menu-File-Save_As")}', self)
            saveAct.setShortcut('Ctrl+S')
            saveAct.setStatusTip(self.Lang.Get("Menu-File-Save_As-Tips"))
            saveAct.triggered.connect(self.saveAs)
            self.statusBar()

            menubar = self.menuBar()
            fileMenu = menubar.addMenu(f'&{self.Lang.Get("Menu-File")}')
            fileMenu.addAction(openAct)
            fileMenu.addAction(saveAct)
            fileMenu.addSeparator()
            fileMenu.addAction(exitAct)

        if True: # Menu - Function
            grayscalevVewAct = QAction(QIcon(None), f'&{self.Lang.Get("Menu-Display-Grayscale")}', self)
            grayscalevVewAct.setShortcut('Ctrl+G')
            grayscalevVewAct.setStatusTip(self.Lang.Get("Menu-Display-Grayscale-Tips"))
            grayscalevVewAct.triggered.connect(self.grayscaleView)
            self.statusBar()

            View3dAct = QAction(QIcon(None), f'&{self.Lang.Get("Menu-Display-3D_Point_Cloud")}', self)
            View3dAct.setShortcut('Ctrl+H')
            View3dAct.setStatusTip(self.Lang.Get("Menu-Display-3D_Point_Cloud-Tips"))
            View3dAct.triggered.connect(self.View3dPoints)
            self.statusBar()

            View3dImgAct = QAction(QIcon(None), f'&{self.Lang.Get("Menu-Display-HeightMap")}', self)
            View3dImgAct.setShortcut('Ctrl+J')
            View3dImgAct.setStatusTip(self.Lang.Get("Menu-Display-HeightMap-Tips"))
            View3dImgAct.triggered.connect(self.ViewImageHeightMap)
            self.statusBar()

            ViewTimeSeriesAct = QAction(QIcon(None), f'&{self.Lang.Get("Menu-Display-Time_Series")}', self)
            ViewTimeSeriesAct.setShortcut('Ctrl+K')  # 注意：原为 Ctrl+S，但已被“保存”占用，建议改为 Ctrl+T
            ViewTimeSeriesAct.setStatusTip(self.Lang.Get("Menu-Display-Time_Series-Tips"))
            ViewTimeSeriesAct.triggered.connect(self.ViewTimeseries)
            self.statusBar()

            ViewGraphSeriesAct = QAction(QIcon(None), f'&{self.Lang.Get("Menu-Display-Directional_Graph")}', self)
            ViewGraphSeriesAct.setShortcut('Ctrl+L')
            ViewGraphSeriesAct.setStatusTip(self.Lang.Get("Menu-Display-Directional_Graph-Tips"))
            ViewGraphSeriesAct.triggered.connect(self.ViewGraphSeriesAct)
            self.statusBar()
            
            menubar = self.menuBar()
            displayMenu = menubar.addMenu(f'&{self.Lang.Get("Menu-Display")}')
            displayMenu.addAction(grayscalevVewAct)
            displayMenu.addAction(View3dAct)
            displayMenu.addAction(View3dImgAct)
            displayMenu.addAction(ViewTimeSeriesAct)
            displayMenu.addAction(ViewGraphSeriesAct)

        if True:  # Menu - Settings
            menubar = self.menuBar()
            settingsMenu = menubar.addMenu(f'&{self.Lang.Get("Menu-Settings")}')

            # 主语言菜单项（带图标）
            langAct = QAction(QIcon('Language.png'), f'&{self.Lang.Get("Menu-Settings-Language")}', self)
            langAct.setStatusTip(self.Lang.Get("Menu-Settings-Language-Tips"))

            # 创建语言子菜单
            langMenu = QMenu(f'&{self.Lang.Get("Menu-Settings-Language")}', self)
            langMenu.setIcon(QIcon('Language.png'))

            # 中文选项
            zhAct = QAction(QIcon('Language-zh.png'), f'&{self.Lang.Get("Menu-Settings-Language-zh-CN")}', self)
            zhAct.setCheckable(True)
            zhAct.setChecked(self.Lang.LangType == 'zh-CN')
            zhAct.triggered.connect(lambda: self.ReSetLanguage(LangType='zh-CN'))

            # 英文选项
            enAct = QAction(QIcon('Language-en.png'), f'&{self.Lang.Get("Menu-Settings-Language-en-US")}', self)
            enAct.setCheckable(True)
            enAct.setChecked(self.Lang.LangType == 'en-US')
            enAct.triggered.connect(lambda: self.ReSetLanguage(LangType='en-US'))

            # 将语言选项加入子菜单
            langMenu.addAction(zhAct)
            langMenu.addAction(enAct)

            # 设置主菜单项的子菜单
            langAct.setMenu(langMenu)
            settingsMenu.addAction(langAct)

    def initMenu_Table(self, mode: Literal['ColHeader', 'RowHeader', None] = None, position: QPoint | None = None):
        if position is None:
            self.tableWidget.setHorizontalHeader(self.tableWidget_ColHeader)
            self.tableWidget_ColHeader.setContextMenuPolicy(Qt.CustomContextMenu)
            self.tableWidget_ColHeader.customContextMenuRequested.connect(lambda pos: self.initMenu_Table('ColHeader', pos))

            self.tableWidget.setVerticalHeader(self.tableWidget_RowHeader)
            self.tableWidget_RowHeader.setContextMenuPolicy(Qt.CustomContextMenu)
            self.tableWidget_RowHeader.customContextMenuRequested.connect(lambda pos: self.initMenu_Table('RowHeader', pos))
            return

        match mode:
            case 'ColHeader':
                currentCol = self.tableWidget_ColHeader.logicalIndexAt(position)
                if currentCol < 0: return

                insertLeftAct = QAction(QIcon('Insert-Right.png'), f'&{self.Lang.Get("Table-Col-Insert-Left")}', self.tableWidget_ColHeader)
                insertLeftAct.triggered.connect(lambda: self.TableFeatures('Insert-Left', currentCol=currentCol))

                insertRightAct = QAction(QIcon('Insert-Right.png'), f'&{self.Lang.Get("Table-Col-Insert-Right")}', self.tableWidget_ColHeader)
                insertRightAct.triggered.connect(lambda: self.TableFeatures('Insert-Right', currentCol=currentCol))

                autoAdjustSelectedAct = QAction(QIcon('AutoAdjust-ColWidth-Selected.png'), f'&{self.Lang.Get("Table-Col-AutoAdjust-Selected")}', self.tableWidget_ColHeader)
                autoAdjustSelectedAct.triggered.connect(lambda: self.TableFeatures('AutoAdjust-ColWidth-Selected', currentCol=currentCol))

                autoAdjustALLAct = QAction(QIcon('AutoAdjust-ColWidth-ALL.png'), f'&{self.Lang.Get("Table-Col-AutoAdjust-All")}', self.tableWidget_ColHeader)
                autoAdjustALLAct.triggered.connect(lambda: self.TableFeatures('AutoAdjust-ColWidth-ALL'))

                delCurrentCollAct = QAction(QIcon('Delete-Col.png'), f'&{self.Lang.Get("Table-Col-Delete")}', self.tableWidget_ColHeader)
                delCurrentCollAct.triggered.connect(lambda: self.TableFeatures('Delete-Col', currentCol=currentCol))

                ColHeaderMenu = QMenu()
                ColHeaderMenu.addAction(insertLeftAct)
                ColHeaderMenu.addAction(insertRightAct)
                ColHeaderMenu.addSeparator()
                ColHeaderMenu.addAction(autoAdjustSelectedAct)
                ColHeaderMenu.addAction(autoAdjustALLAct)
                ColHeaderMenu.addSeparator()
                ColHeaderMenu.addAction(delCurrentCollAct)

                ColHeaderMenu.exec(self.tableWidget_ColHeader.viewport().mapToGlobal(position))

            case 'RowHeader':
                currentRow = self.tableWidget_RowHeader.logicalIndexAt(position)
                if currentRow < 0: return

                insertTopAct = QAction(QIcon('InsertTop.png'), f'&{self.Lang.Get("Table-Row-Insert-Top")}', self.tableWidget_RowHeader)
                insertTopAct.triggered.connect(lambda: self.TableFeatures('Insert-Top', currentRow=currentRow))

                insertDownAct = QAction(QIcon('InsertDown.png'), f'&{self.Lang.Get("Table-Row-Insert-Bottom")}', self.tableWidget_RowHeader)
                insertDownAct.triggered.connect(lambda: self.TableFeatures('Insert-Down', currentRow=currentRow))

                autoAdjustSelectedAct = QAction(QIcon('AutoAdjust-RowWidth-Selected.png'), f'&{self.Lang.Get("Table-Row-AutoAdjust-Selected")}', self.tableWidget_RowHeader)
                autoAdjustSelectedAct.triggered.connect(lambda: self.TableFeatures('AutoAdjust-RowWidth-Selected', currentRow=currentRow))

                autoAdjustALLAct = QAction(QIcon('AutoAdjust-RowWidth-ALL.png'), f'&{self.Lang.Get("Table-Row-AutoAdjust-All")}', self.tableWidget_RowHeader)
                autoAdjustALLAct.triggered.connect(lambda: self.TableFeatures('AutoAdjust-RowWidth-ALL'))

                delCurrentRowlAct = QAction(QIcon('DelCurrentRow.png'), f'&{self.Lang.Get("Table-Row-Delete")}', self.tableWidget_RowHeader)
                delCurrentRowlAct.triggered.connect(lambda: self.TableFeatures('Delete-Row', currentRow=currentRow))

                RowHeaderMenu = QMenu()
                RowHeaderMenu.addAction(insertTopAct)
                RowHeaderMenu.addAction(insertDownAct)
                RowHeaderMenu.addSeparator()
                RowHeaderMenu.addAction(autoAdjustSelectedAct)
                RowHeaderMenu.addAction(autoAdjustALLAct)
                RowHeaderMenu.addSeparator()
                RowHeaderMenu.addAction(delCurrentRowlAct)

                RowHeaderMenu.exec(self.tableWidget_RowHeader.viewport().mapToGlobal(position))

    def TableFeatures(self, mode: Literal[
        'Delete-Col', 'Delete-Row'
        'Insert-Top', 'Insert-Down', 'Insert-Left', 'Insert-Right',
        'AutoAdjust-ColWidth-Selected', 'AutoAdjust-ColWidth-ALL', 'AutoAdjust-RowWidth-Selected', 'AutoAdjust-RowWidth-ALL', 
        ], **kwargs):
        match mode:
            case 'Delete-Col':
                if self.tableWidget.columnCount() > 1:
                    self.tableWidget.removeColumn(kwargs['currentCol'])
                # for currentCol in range(self.tableWidget.columnCount()):
                #     self.tableWidget.setHorizontalHeaderItem(currentCol, QTableWidgetItem(str(currentCol+1)))
            case 'Delete-Row':
                if self.tableWidget.rowCount() > 1:
                    self.tableWidget.removeRow(kwargs['currentRow'])
                # for currentRow in range(self.tableWidget.rowCount()):
                #     self.tableWidget.setHorizontalHeaderItem(currentRow, QTableWidgetItem(str(currentRow+1)))
            case 'Insert-Top':
                self.tableWidget.insertRow(kwargs['currentRow'])
                # for currentRow in range(kwargs['currentRow'], self.tableWidget.rowCount()):
                #     self.tableWidget.setHorizontalHeaderItem(currentRow, QTableWidgetItem(str(currentRow+1)))
            case 'Insert-Down':
                self.tableWidget.insertRow(kwargs['currentRow']+1)
                # for currentRow in range(kwargs['currentRow']+1, self.tableWidget.rowCount()):
                #     self.tableWidget.setHorizontalHeaderItem(currentRow, QTableWidgetItem(str(currentRow+1)))
            case 'Insert-Left':
                self.tableWidget.insertColumn(kwargs['currentCol'])
                # for currentCol in range(kwargs['currentCol'], self.tableWidget.columnCount()):
                #     self.tableWidget.setHorizontalHeaderItem(currentCol, QTableWidgetItem(str(currentCol+1)))
            case 'Insert-Right':
                self.tableWidget.insertColumn(kwargs['currentCol']+1)
                # for currentCol in range(kwargs['currentCol']+1, self.tableWidget.columnCount()):
                #     self.tableWidget.setHorizontalHeaderItem(currentCol, QTableWidgetItem(str(currentCol+1)))
            case 'AutoAdjust-ColWidth-Selected':
                self.tableWidget.resizeColumnToContents(kwargs['currentCol'])
            case 'AutoAdjust-ColWidth-ALL':
                self.tableWidget.resizeColumnsToContents()
            case 'AutoAdjust-RowWidth-Selected':
                self.tableWidget.resizeRowToContents(kwargs['currentRow'])
            case 'AutoAdjust-RowWidth-ALL':
                self.tableWidget.resizeRowsToContents()

    def saveAs(self):
        home = str(Path.home())
        if self.npyfile is not None:
            home = os.path.dirname(self.npyfile.filename)
        path = QFileDialog.getSaveFileName(
            self, 'Save File', home, 'NPY (*.npy);;CSV(*.csv);;MAT(*.mat)')[0]
        # path = QFileDialog.getSaveFileName(
        #    self, 'Save File', home, 'CSV(*.csv)')[0]
        if path != "" and ".csv" in path:
            with open((path.replace(".csv", "") + ".csv"), 'w') as stream:
                writer = csv.writer(stream)
                for row in range(self.tableWidget.rowCount()):
                    rowdata = []
                    for column in range(self.tableWidget.columnCount()):
                        item = self.tableWidget.item(row, column)
                        if item is not None:
                            rowdata.append(item.text())

                        else:
                            rowdata.append('')
                    writer.writerow(rowdata)
        else:
            OutMatrix = []
            for row in range(self.tableWidget.rowCount()):
                rowdata = []
                for column in range(self.tableWidget.columnCount()):
                    item = self.tableWidget.item(row, column)
                    if item is not None:
                        #if item.text().isnumeric():
                        rowdata.append(float(item.text()))

                if rowdata != []:
                    OutMatrix.append(rowdata)
            OutMatrix = np.array(OutMatrix)
            if ".csv" in path:
                np.save(path, np.array(OutMatrix))
            if ".mat" in path:
                mdic = {"ans": OutMatrix}
                print(OutMatrix)
                savemat(path, mdic)

    def openNPY(self):
        if os.path.exists("lastpath.npy"):
            home = str(np.load("lastpath.npy"))
            print(home)
        else:
            home = str(Path.home())
        if self.npyfile is not None:
            home = os.path.dirname(self.npyfile.filename)
        filename = QFileDialog.getOpenFileName(self, 'Open .NPY file', home, ".NPY files (*.npy);;.CSV files (*.csv)")[
            0]
        if filename != "":
            if ".npy" in filename:
                data = np.load(filename, allow_pickle=True)
            else:
                data = np.array(pd.read_csv(filename).values.tolist())

            npyfile = NPYfile(data, filename)
            print(npyfile)
            self.setWindowTitle("NPYViewer v." +version+": "+ npyfile.filename)
            self.infoLb.setText(f'{self.Lang.Get("Label-NPY_Info")}\n' + str(npyfile))
            self.tableWidget.clear()

            # initialise table
            self.tableWidget.setRowCount(data.shape[0])
            dtype_dim = len(npyfile.data.dtype)  # 0, if plain dtype, 1 or bigger if compound dtype
            if data.ndim > 1:
                self.tableWidget.setColumnCount(data.shape[1])
            elif dtype_dim > 0:
                self.tableWidget.setColumnCount(dtype_dim)
            else:
                self.tableWidget.setColumnCount(1)

            # fill data
            if data.ndim > 1:
                for i, value1 in enumerate(npyfile.data):  # loop over items in first column
                    for j, value in enumerate(value1):
                        self.tableWidget.setItem(i, j, QTableWidgetItem(str(value)))
            elif dtype_dim > 0:
                for i, value1 in enumerate(npyfile.data):
                    for j, col_name in enumerate(npyfile.data.dtype.names):
                        self.tableWidget.setItem(i, j, QTableWidgetItem(str(value1[col_name])))
            else:
                for i, value1 in enumerate(npyfile.data):  # loop over items in first column
                    self.tableWidget.setItem(i, 0, QTableWidgetItem(str(value1)))

            self.npyfile = npyfile
            path = os.path.dirname(filename)
            np.save("lastpath.npy", path)

    def openNPY_CLI(self,filename):

            if ".npy" in filename:
                data = np.load(filename, allow_pickle=True)
            else:
                data = np.array(pd.read_csv(filename).values.tolist())

            npyfile = NPYfile(data, filename)
            print(npyfile)
            self.setWindowTitle("NPYViewer v." +version+": " +npyfile.filename)
            self.infoLb.setText("NPY Properties:\n" + str(npyfile))
            self.tableWidget.clear()

            # initialise table
            self.tableWidget.setRowCount(data.shape[0])
            dtype_dim = len(npyfile.data.dtype)  # 0, if plain dtype, 1 or bigger if compound dtype
            if data.ndim > 1:
                self.tableWidget.setColumnCount(data.shape[1])
            elif dtype_dim > 0:
                self.tableWidget.setColumnCount(dtype_dim)
            else:
                self.tableWidget.setColumnCount(1)

            # fill data
            if data.ndim > 1:
                for i, value1 in enumerate(npyfile.data):  # loop over items in first column
                    for j, value in enumerate(value1):
                        self.tableWidget.setItem(i, j, QTableWidgetItem(str(value)))
            elif dtype_dim > 0:
                for i, value1 in enumerate(npyfile.data):
                    for j, col_name in enumerate(npyfile.data.dtype.names):
                        self.tableWidget.setItem(i, j, QTableWidgetItem(str(value1[col_name])))
            else:
                for i, value1 in enumerate(npyfile.data):  # loop over items in first column
                    self.tableWidget.setItem(i, 0, QTableWidgetItem(str(value1)))

            self.npyfile = npyfile
            path = os.path.dirname(filename)
            np.save("lastpath.npy", path)

    def grayscaleView(self):
        OutMatrix = []
        for row in range(self.tableWidget.rowCount()):
            rowdata = []
            for column in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, column)
                # print(item.text())
                if item is not None:
                    rowdata.append(np.float32(item.text()))

            if len(rowdata) > 0 and rowdata != None:
                OutMatrix.append(rowdata)

        OutMatrix = np.array(OutMatrix)
        plt.imshow(OutMatrix, cmap='gray')
        plt.show()
        return

    def ViewGraphSeriesAct(self):
        OutMatrix = []
        for row in range(self.tableWidget.rowCount()):
            rowdata = []
            for column in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, column)
                # print(item.text())
                if item is not None:
                    rowdata.append(np.float32(item.text()))

            if len(rowdata) > 0 and rowdata != None:
                OutMatrix.append(rowdata)

        OutMatrix = np.array(OutMatrix)


        G = nx.DiGraph(OutMatrix)

        # Set the position of the nodes in the graph
        pos = nx.spring_layout(G)

        # Draw the graph with labels and edges
        labels = {node: str(int(node)+1) for node in G.nodes()}

        nx.draw_networkx_labels(G, pos,labels=labels)
        nx.draw_networkx_edges(G, pos)

        nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): OutMatrix[i][j] for i, j in G.edges()})

        # Draw the nodes with the same color
        nx.draw_networkx_nodes(G, pos, node_color='b')


        # Set the title and show the plot
        plt.title("Directional Graph")
        plt.show()

    def ViewImageHeightMap(self):
        OutMatrix = []
        for row in range(self.tableWidget.rowCount()):
            rowdata = []
            for column in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, column)

                if item is not None:
                    if item.text():
                        rowdata.append(np.float32(item.text()))
            if len(rowdata) > 0 and rowdata != None:
                OutMatrix.append(rowdata)
        # print(OutMatrix)
        HeightMap = []
        for x, row in enumerate(OutMatrix):
            for y, val in enumerate(row):
                HeightMap.append([x, y, val])
        OutMatrix = np.array(HeightMap)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = OutMatrix[:, 0]

        ys = OutMatrix[:, 1]
        zs = OutMatrix[:, 2]
        ax.plot_trisurf(xs, ys, zs,
                        cmap='Greys_r', edgecolor='none');

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        plt.show()
        return

    def View3dPoints(self):
        OutMatrix = []
        for row in range(self.tableWidget.rowCount()):
            rowdata = []
            for column in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, column)

                if item is not None:
                    if item.text():
                        rowdata.append(np.float32(item.text()))
            if len(rowdata) > 0 and rowdata != None:
                OutMatrix.append(rowdata)
        # print(OutMatrix)
        OutMatrix = np.array(OutMatrix)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = OutMatrix[:, 0]

        ys = OutMatrix[:, 1]
        zs = OutMatrix[:, 2]
        ax.scatter(xs, ys, zs, c='r', marker='o')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        plt.show()
        return
    
    def ViewTimeseries(self):
        OutMatrix = []
        for row in range(self.tableWidget.rowCount()):
            rowdata = []
            for column in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, column)

                if item is not None:
                    if item.text():
                        rowdata.append(np.float32(item.text()))
            if len(rowdata) > 0 and rowdata != None:
                OutMatrix.append(rowdata)
        # print(OutMatrix)
        #OutMatrix = np.array(OutMatrix)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        indices=range(0,len(OutMatrix))
        plt.plot(indices, OutMatrix, label='values', linewidth=3)

        ax.set_xlabel('Time Unit')
        ax.set_ylabel('Values')

        plt.show()
        return

    def ReSetLanguage(self, LangType: str):
        global zLang
        zLang = Lang(LangType=LangType)
        self.Lang = zLang

        # UpDate窗口标题
        self.setWindowTitle(self.Lang.Get("Win_Tittle") + "v." + version)

        # 更新信息标签
        if self.npyfile is None: self.infoLb.setText(self.Lang.Get("Label-NPY_Info"))
        else: self.infoLb.setText(f'{self.Lang.Get("Label-NPY_Info")}\n' + str(self.npyfile))

        # 重建菜单栏（语言相关的文本）
        menubar = self.menuBar()
        menubar.clear()
        self.initMenu_Tittle()

version="1.28"
zLang: Lang

def main():
    App: QApplication = QApplication(sys.argv)
    Window: MainApp = MainApp(App=App, LangType='zh-CN')
    sys.exit(App.exec())

if __name__ == '__main__':
    main()