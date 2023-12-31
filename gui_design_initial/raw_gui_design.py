# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sas.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1074, 660)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy)
        self.stackedWidget.setMinimumSize(QtCore.QSize(0, 0))
        self.stackedWidget.setObjectName("stackedWidget")
        self.Home = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Home.sizePolicy().hasHeightForWidth())
        self.Home.setSizePolicy(sizePolicy)
        self.Home.setObjectName("Home")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.Home)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.home_grid = QtWidgets.QGridLayout()
        self.home_grid.setVerticalSpacing(50)
        self.home_grid.setObjectName("home_grid")
        self.b2 = QtWidgets.QPushButton(self.Home)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b2.sizePolicy().hasHeightForWidth())
        self.b2.setSizePolicy(sizePolicy)
        self.b2.setMinimumSize(QtCore.QSize(300, 40))
        self.b2.setMaximumSize(QtCore.QSize(300, 40))
        self.b2.setObjectName("b2")
        self.home_grid.addWidget(self.b2, 1, 0, 1, 1)
        self.b1 = QtWidgets.QPushButton(self.Home)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b1.sizePolicy().hasHeightForWidth())
        self.b1.setSizePolicy(sizePolicy)
        self.b1.setMinimumSize(QtCore.QSize(300, 40))
        self.b1.setMaximumSize(QtCore.QSize(300, 40))
        self.b1.setObjectName("b1")
        self.home_grid.addWidget(self.b1, 0, 0, 1, 1)
        self.b3 = QtWidgets.QPushButton(self.Home)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b3.sizePolicy().hasHeightForWidth())
        self.b3.setSizePolicy(sizePolicy)
        self.b3.setMinimumSize(QtCore.QSize(300, 40))
        self.b3.setMaximumSize(QtCore.QSize(300, 40))
        self.b3.setObjectName("b3")
        self.home_grid.addWidget(self.b3, 2, 0, 1, 1)
        self.verticalLayout_9.addLayout(self.home_grid)
        self.stackedWidget.addWidget(self.Home)


        self.Informations = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Informations.sizePolicy().hasHeightForWidth())
        self.Informations.setSizePolicy(sizePolicy)
        self.Informations.setObjectName("Informations")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.Informations)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.student_info_layout = QtWidgets.QVBoxLayout()
        self.student_info_layout.setObjectName("student_info_layout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.class_selection = QtWidgets.QComboBox(self.Informations)
        self.class_selection.setObjectName("class_selection")
        self.horizontalLayout.addWidget(self.class_selection)
        self.save_student_info_btn = QtWidgets.QPushButton(self.Informations)
        self.save_student_info_btn.setObjectName("save_student_info_btn")
        self.horizontalLayout.addWidget(self.save_student_info_btn)
        self.close_student_info_btn = QtWidgets.QPushButton(self.Informations)
        self.close_student_info_btn.setObjectName("close_student_info_btn")
        self.horizontalLayout.addWidget(self.close_student_info_btn)
        self.student_info_layout.addLayout(self.horizontalLayout)
        self.student_info = QtWidgets.QTableWidget(self.Informations)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.student_info.setFont(font)
        self.student_info.setObjectName("student_info")
        self.student_info.setColumnCount(0)
        self.student_info.setRowCount(0)
        self.student_info_layout.addWidget(self.student_info)
        self.student_info_layout.setStretch(0, 1)
        self.student_info_layout.setStretch(1, 15)
        self.verticalLayout_4.addLayout(self.student_info_layout)
        self.stackedWidget.addWidget(self.Informations)


        self.Record = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Record.sizePolicy().hasHeightForWidth())
        self.Record.setSizePolicy(sizePolicy)
        self.Record.setObjectName("Record")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.Record)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.record_grid = QtWidgets.QHBoxLayout()
        self.record_grid.setObjectName("record_grid")
        self.webcam_layout = QtWidgets.QVBoxLayout()
        self.webcam_layout.setObjectName("webcam_layout")
        self.webcam = QtWidgets.QLabel(self.Record)
        self.webcam.setText("")
        self.webcam.setObjectName("webcam")
        self.webcam_layout.addWidget(self.webcam)
        self.close_button = QtWidgets.QPushButton(self.Record)
        self.close_button.setMinimumSize(QtCore.QSize(200, 40))
        self.close_button.setMaximumSize(QtCore.QSize(200, 40))
        self.close_button.setObjectName("close_button")
        self.webcam_layout.addWidget(self.close_button, 0, QtCore.Qt.AlignHCenter)
        self.record_grid.addLayout(self.webcam_layout)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.date_layout = QtWidgets.QHBoxLayout()
        self.date_layout.setObjectName("date_layout")
        self.date_label = QtWidgets.QLabel(self.Record)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.date_label.setFont(font)
        self.date_label.setObjectName("date_label")
        self.date_layout.addWidget(self.date_label)
        self.select_date = QtWidgets.QDateEdit(self.Record)
        self.select_date.setMaximumDate(QtCore.QDate(2030, 12, 31))
        self.select_date.setMinimumDate(QtCore.QDate(2020, 9, 14))
        self.select_date.setCalendarPopup(True)
        self.select_date.setDate(QtCore.QDate(2020, 9, 14))
        self.select_date.setObjectName("select_date")
        self.date_layout.addWidget(self.select_date)
        self.date_layout.setStretch(0, 1)
        self.date_layout.setStretch(1, 2)
        self.verticalLayout.addLayout(self.date_layout)
        self.class_select = QtWidgets.QComboBox(self.Record)
        self.class_select.setCurrentText("")
        self.class_select.setObjectName("class_select")
        self.verticalLayout.addWidget(self.class_select)
        self.table = QtWidgets.QTableWidget(self.Record)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.table.sizePolicy().hasHeightForWidth())
        self.table.setSizePolicy(sizePolicy)
        self.table.setObjectName("table")
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        self.verticalLayout.addWidget(self.table)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 8)
        self.record_grid.addLayout(self.verticalLayout)
        self.record_grid.setStretch(0, 2)
        self.record_grid.setStretch(1, 1)
        self.verticalLayout_2.addLayout(self.record_grid)
        self.stackedWidget.addWidget(self.Record)


        self.New_Student = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.New_Student.sizePolicy().hasHeightForWidth())
        self.New_Student.setSizePolicy(sizePolicy)
        self.New_Student.setObjectName("New_Student")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.New_Student)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.new_entry_grid = QtWidgets.QHBoxLayout()
        self.new_entry_grid.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.new_entry_grid.setObjectName("new_entry_grid")
        self.webcam_entry_layout = QtWidgets.QVBoxLayout()
        self.webcam_entry_layout.setObjectName("webcam_entry_layout")
        self.webcam2 = QtWidgets.QLabel(self.New_Student)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.webcam2.sizePolicy().hasHeightForWidth())
        self.webcam2.setSizePolicy(sizePolicy)
        self.webcam2.setText("")
        self.webcam2.setObjectName("webcam2")
        self.webcam_entry_layout.addWidget(self.webcam2)
        self.buttons_layout = QtWidgets.QHBoxLayout()
        self.buttons_layout.setObjectName("buttons_layout")
        self.capture_button = QtWidgets.QPushButton(self.New_Student)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.capture_button.sizePolicy().hasHeightForWidth())
        self.capture_button.setSizePolicy(sizePolicy)
        self.capture_button.setObjectName("capture_button")
        self.buttons_layout.addWidget(self.capture_button)
        self.save_button = QtWidgets.QPushButton(self.New_Student)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_button.sizePolicy().hasHeightForWidth())
        self.save_button.setSizePolicy(sizePolicy)
        self.save_button.setObjectName("save_button")
        self.buttons_layout.addWidget(self.save_button)
        self.webcam_entry_layout.addLayout(self.buttons_layout)
        self.webcam_entry_layout.setStretch(0, 15)
        self.webcam_entry_layout.setStretch(1, 1)
        self.new_entry_grid.addLayout(self.webcam_entry_layout)
        self.new_info_layout = QtWidgets.QVBoxLayout()
        self.new_info_layout.setObjectName("new_info_layout")
        self.class_layout = QtWidgets.QGridLayout()
        self.class_layout.setObjectName("class_layout")
        self.class_select_2 = QtWidgets.QLabel(self.New_Student)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.class_select_2.setFont(font)
        self.class_select_2.setObjectName("class_select_2")
        self.class_layout.addWidget(self.class_select_2, 0, 0, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.class_prompt = QtWidgets.QComboBox(self.New_Student)
        self.class_prompt.setObjectName("class_prompt")
        self.class_layout.addWidget(self.class_prompt, 1, 0, 1, 1, QtCore.Qt.AlignTop)
        self.new_info_layout.addLayout(self.class_layout)
        self.name_layout = QtWidgets.QGridLayout()
        self.name_layout.setObjectName("name_layout")
        self.name_prompt = QtWidgets.QLabel(self.New_Student)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.name_prompt.sizePolicy().hasHeightForWidth())
        self.name_prompt.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.name_prompt.setFont(font)
        self.name_prompt.setObjectName("name_prompt")
        self.name_layout.addWidget(self.name_prompt, 0, 0, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.name = QtWidgets.QLineEdit(self.New_Student)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.name.sizePolicy().hasHeightForWidth())
        self.name.setSizePolicy(sizePolicy)
        self.name.setMinimumSize(QtCore.QSize(0, 40))
        self.name.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.name.setFont(font)
        self.name.setText("")
        self.name.setObjectName("name")
        self.name_layout.addWidget(self.name, 1, 0, 1, 1, QtCore.Qt.AlignTop)
        self.new_info_layout.addLayout(self.name_layout)
        self.id_layout = QtWidgets.QGridLayout()
        self.id_layout.setObjectName("id_layout")
        self.id = QtWidgets.QLineEdit(self.New_Student)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.id.sizePolicy().hasHeightForWidth())
        self.id.setSizePolicy(sizePolicy)
        self.id.setMinimumSize(QtCore.QSize(0, 40))
        self.id.setMaximumSize(QtCore.QSize(16777215, 40))
        self.id.setObjectName("id")
        self.id_layout.addWidget(self.id, 1, 0, 1, 1, QtCore.Qt.AlignTop)
        self.id_prompt = QtWidgets.QLabel(self.New_Student)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.id_prompt.sizePolicy().hasHeightForWidth())
        self.id_prompt.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.id_prompt.setFont(font)
        self.id_prompt.setObjectName("id_prompt")
        self.id_layout.addWidget(self.id_prompt, 0, 0, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.new_info_layout.addLayout(self.id_layout)
        self.new_entry_grid.addLayout(self.new_info_layout)
        self.new_entry_grid.setStretch(0, 2)
        self.new_entry_grid.setStretch(1, 1)
        self.verticalLayout_8.addLayout(self.new_entry_grid)
        self.stackedWidget.addWidget(self.New_Student)
        
        
        self.verticalLayout_3.addWidget(self.stackedWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1074, 26))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.menubar.sizePolicy().hasHeightForWidth())
        self.menubar.setSizePolicy(sizePolicy)
        self.menubar.setObjectName("menubar")
        self.menuHow_To = QtWidgets.QMenu(self.menubar)
        self.menuHow_To.setObjectName("menuHow_To")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.statusbar.sizePolicy().hasHeightForWidth())
        self.statusbar.setSizePolicy(sizePolicy)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuHow_To.menuAction())

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Automatic Student Attendance System"))
        self.b2.setText(_translate("MainWindow", "Record"))
        self.b1.setText(_translate("MainWindow", "Student Information"))
        self.b3.setText(_translate("MainWindow", "New Student"))
        self.save_student_info_btn.setText(_translate("MainWindow", "Save"))
        self.close_student_info_btn.setText(_translate("MainWindow", "Close"))
        self.close_button.setText(_translate("MainWindow", "Close"))
        self.date_label.setText(_translate("MainWindow", "Date :"))
        self.capture_button.setText(_translate("MainWindow", "Capture"))
        self.save_button.setText(_translate("MainWindow", "Save"))
        self.class_select_2.setText(_translate("MainWindow", "Enter Class"))
        self.name_prompt.setText(_translate("MainWindow", "Enter Student Name"))
        self.id_prompt.setText(_translate("MainWindow", "Enter Student ID"))
        self.menuHow_To.setTitle(_translate("MainWindow", "How To"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
