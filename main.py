from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, \
    QMessageBox, QHBoxLayout, QColorDialog, QShortcut, \
    QVBoxLayout
from PyQt5 import QtWidgets
import MainWindow_3D_PP
from utils.coords2labels import Coord_to_Label, Coord_to_Label_v1, label_gen_show
from utils.normalization import InputNorm, norm_show
from utils.coord_gen import coords_gen, coords_gen_show
import json
# from pyqtgraph.graphicsItems.ViewBox import axisCtrlTemplate_pyqt5
# from pyqtgraph.graphicsItems.PlotItem import plotConfigTemplate_pyqt5
# from pyqtgraph.imageview import ImageViewTemplate_pyqt5
import pyqtgraph as pg
import mrcfile
import numpy as np
import cv2
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import pyqtSlot
from skimage import filters
import os
import threading, inspect, time
from train import *
from test import *
from utils.coordFormatConvert import *
from utils.utils import *


"""Particle color for different categories."""
colors = [(244, 67, 54),  # 1
          (255, 235, 59),  # 2
          (156, 39, 176),  # 3
          (33, 150, 243),  # 4
          (0, 188, 212),  # 5
          (139, 195, 74),  # 6
          (255, 152, 0),  # 7
          (63, 81, 181),  # 8
          (255, 193, 7),  # 9
          (255, 0, 0),  # 10
          (0, 255, 0),  # 11
          (0, 0, 255),  # 12
          (255, 255, 0),  # 13
          (255, 0, 255),  # 14
          (0, 255, 255),  # 15
          ]


class Stats(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = MainWindow_3D_PP.Ui_DeepETPicker()
        self.ui.setupUi(self)
        self.ui.retranslateUi(self)

        self.ui.edit_dset_name.textChanged.connect(self.DsetNameChanged)
        self.ui.pBut_add_coord.clicked.connect(self.openCoordPath)
        self.ui.pBut_add_base.clicked.connect(self.openBasePath)
        self.ui.pBut_sel_tomo.clicked.connect(self.openTomoFile)
        self.ui.bb_c2l_ok.clicked.connect(self.c2l_ok)
        self.ui.cbox_coord_format.currentIndexChanged.connect(self.coord_format_change)
        self.ui.cbox_tomo_format.currentIndexChanged.connect(self.lable_format_change)

        self.ui.cbox_label_type.currentIndexChanged.connect(self.lable_type_change)
        self.label_type = self.ui.cbox_label_type.currentText()

        self.ui.sb_label_diameter.valueChanged.connect(self.label_diameter_change)
        self.label_diameter = self.ui.sb_label_diameter.value()

        self.ui.cbox_ocp_type.currentIndexChanged.connect(self.ocp_type_change)
        self.ocp_type = self.ui.cbox_ocp_type.currentText()

        self.ui.edit_ocp_diameter.textChanged.connect(self.ocp_diameter_change)
        self.ocp_diameter = self.ui.edit_ocp_diameter.text()

        self.ui.sb_cls_num.valueChanged.connect(self.cls_num_change)
        self.cls_num = self.ui.sb_cls_num.value()

        self.input_norm = self.ui.butG_pre_norm.checkedButton().text()
        self.ui.butG_pre_norm.buttonClicked.connect(self.RadButtonClicked)

        self.ui.pBut_save_configs.clicked.connect(self.saveConfigs)
        self.ui.pBut_load_configs.clicked.connect(self.loadConfigs)

        # Show results
        self.sectionView("")
        # self.vbox = QVBoxLayout()
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.win, stretch=3)
        self.hbox.addWidget(self.ui.gBox_res_load, stretch=2)
        # self.hbox.setStretch(5, 1)
        # self.vbox.addLayout(self.hbox)

        # set global layout
        self.ui.tab_res_show.setLayout(self.hbox)
        self.sub1.scene().sigMouseClicked.connect(self.mouseClicked)
        pg.SignalProxy(self.sub1.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClicked)

        self.ui.pBut_show_tomo.clicked.connect(self.showTomo)
        self.ui.pBut_show_label.clicked.connect(self.showLabel)
        self.ui.pBut_show_res.clicked.connect(self.showTomoLabel)

        self.res_label = []
        self.res_label_type = self.ui.butG_res_label_type.checkedButton().text()
        if self.res_label_type == 'Mask':
            self.ui.hSlider_circle_diameter.setEnabled(False)
            self.ui.cbox_circle.setEnabled(False)
            self.ui.sb_circle_width.setEnabled(False)
        else:
            self.ui.hSlider_circle_diameter.setEnabled(True)
            self.ui.cbox_circle.setEnabled(True)
            self.ui.sb_circle_width.setEnabled(True)

        self.ui.butG_res_label_type.buttonClicked.connect(self.RBut_LabelClicked)

        self.circle_width = self.ui.sb_circle_width.value()
        self.ui.sb_circle_width.valueChanged.connect(self.circle_width_change)

        self.ui.hSlider_circle_diameter.valueChanged.connect(self.diameterChange)
        self.ui.hSlider_circle_diameter.setValue(7)
        self.circle_diameter = self.ui.hSlider_circle_diameter.value()
        self.old_diameter = self.circle_diameter
        self.ui.edit_circle_diameter.setText(str(self.circle_diameter))

        self.ui.hSlider_mask_alpha.valueChanged.connect(self.maskAlphaChange)
        self.ui.hSlider_mask_alpha.setValue(20)
        self.mask_alpha = self.ui.hSlider_mask_alpha.value()
        self.old_alpha = self.mask_alpha
        self.ui.edit_mask_alpha.setText(f"{float(self.mask_alpha) / 100.:0.2f}")

        self.ui.cbox_circle.stateChanged.connect(self.isPlotCircle)
        if self.ui.cbox_circle.isChecked():
            self.flag_show_circle = True
            self.ui.hSlider_circle_diameter.setEnabled(True)
        else:
            self.flag_show_circle = False
            self.ui.hSlider_circle_diameter.setEnabled(False)

        self.ui.cbox_mask_show.stateChanged.connect(self.isShowMask)
        self.flag_show_mask = True

        self.ui.pBut_setColor.clicked.connect(self.setColor)
        self.color = (255, 0, 0)

        # Res: params adjust
        self.ui.cbox_gaus_filter.stateChanged.connect(self.isGauFilter)
        self.ui.edit_gaussian_kernel.textChanged.connect(self.isGauFilter)
        self.ui.edit_gaussian_sigma.textChanged.connect(self.isGauFilter)
        if self.ui.cbox_gaus_filter.isChecked():
            self.ui.edit_gaussian_kernel.setEnabled(True)
            self.ui.edit_gaussian_sigma.setEnabled(True)
            self.ui.label_29.setEnabled(True)
            self.ui.label_30.setEnabled(True)
            self.gau_kernel = int(self.ui.edit_gaussian_kernel.text())
            self.gau_sigma = float(self.ui.edit_gaussian_sigma.text())
        else:
            self.ui.edit_gaussian_kernel.setEnabled(False)
            self.ui.edit_gaussian_sigma.setEnabled(False)
            self.ui.label_29.setEnabled(False)
            self.ui.label_30.setEnabled(False)
        self.ui.pBut_paramAdjust_ok.clicked.connect(self.paramAdjust_ok)

        self.ui.hSlider_x.valueChanged.connect(self.xyzChange)
        self.ui.hSlider_y.valueChanged.connect(self.xyzChange)
        self.ui.hSlider_z.valueChanged.connect(self.xyzChange)

        # QShortcut(QKeySequence(self.tr("Command+")))
        QShortcut(QKeySequence(self.tr("Escape")), self, self.close)
        QShortcut(QKeySequence(self.tr("Ctrl+D")), self, self.changeShowMask)
        QShortcut(QKeySequence(self.tr("Up")), self, self.changeZ_up)
        QShortcut(QKeySequence(self.tr("Down")), self, self.changeZ_down)

        self.ui.pBut_SAV_save.clicked.connect(self.saveVideo)

        # Training
        self.train_dsetName = self.ui.train_edit_dsetName.text()
        self.ui.train_edit_dsetName.textChanged.connect(self.train_dsetName_change)
        self.train_modelName = self.ui.train_cbox_modelName.currentText()
        self.ui.train_cbox_modelName.currentIndexChanged.connect(self.train_modelName_change)
        self.train_cls_num = self.ui.train_sb_clsNum.value()
        self.ui.train_sb_clsNum.valueChanged.connect(self.train_cls_num_change)
        self.train_batchSize = self.ui.train_sb_batchSize.value()
        self.ui.train_sb_batchSize.valueChanged.connect(self.train_batchSize_change)
        self.train_patchSize = self.ui.train_sb_patchSize.value()
        self.ui.train_sb_patchSize.valueChanged.connect(self.train_patchSize_change)
        self.train_paddingSize = self.ui.train_sb_paddingSize.value()
        self.ui.train_sb_paddingSize.valueChanged.connect(self.train_paddingSize_change)
        self.train_maxEpochs = self.ui.train_sb_maxEpochs.value()
        self.ui.train_sb_maxEpochs.valueChanged.connect(self.train_maxEpochs_change)
        self.ui.train_edit_segThresh.textChanged.connect(self.train_segThresh_change)
        self.train_segThresh = float(self.ui.train_edit_segThresh.text())
        self.ui.train_edit_lr.textChanged.connect(self.train_lr_change)
        self.train_lr = float(self.ui.train_edit_lr.text())
        self.ui.train_edit_gpuIds.textChanged.connect(self.train_gpuIds_change)
        self.train_gpuIds = self.ui.train_edit_gpuIds.text()
        self.ui.train_pBut_saveConfigs.clicked.connect(self.train_saveConfigs)
        self.ui.train_pBut_loadConfigs.clicked.connect(self.train_loadConfigs)
        self.ui.train_pBut_clear.clicked.connect(self.train_show_clear)
        self.ui.train_pBut_ok.clicked.connect(self.train_ok)
        self.ui.train_pBut_stop.clicked.connect(self.train_stop)
        self.ui.edit_train_set_ids.textChanged.connect(self.train_set_ids_change)
        self.train_set_ids = self.ui.edit_train_set_ids.text()
        self.ui.edit_val_set_ids.textChanged.connect(self.val_set_ids_change)
        self.val_set_ids = self.ui.edit_val_set_ids.text()
        self.ui.train_pBut_dsetList.clicked.connect(self.train_dsetList)

        # Hiding
        self.ui.train_edit_segThresh.setVisible(False)  # train_seg_edit
        self.ui.label_89.setVisible(False)  # train_seg label
        self.ui.radB_norm.setVisible(False)  # normalization
        self.ui.test_edit_segThresh.setVisible(False)
        self.ui.label_15.setVisible(False)
        self.ui.page.setVisible(False)
        self.ui.tab_show_video.removeItem(5)
        self.ui.edit_test_set_ids.setVisible(False)
        self.ui.cbox_ocp_type.setVisible(False)
        self.ui.label_22.setVisible(False)

        # test
        self.ui.test_pBut_trainConfigs.clicked.connect(self.test_load_trainConfigs)
        self.ui.test_pBut_weightPath.clicked.connect(self.test_load_weightPath)
        self.test_patchSize = self.ui.test_sb_patchSize.value()
        self.ui.test_sb_patchSize.valueChanged.connect(self.test_patchSize_change)
        self.test_paddingSize = self.ui.test_sb_paddingSize.value()
        self.ui.test_sb_paddingSize.valueChanged.connect(self.test_paddingSize_change)
        self.ui.test_edit_segThresh.textChanged.connect(self.test_segThresh_change)
        self.test_segThresh = float(self.ui.test_edit_segThresh.text())
        self.ui.test_edit_gpuIds.textChanged.connect(self.test_gpuIds_change)
        self.test_gpuIds = self.ui.test_edit_gpuIds.text()
        self.ui.test_pBut_ok.clicked.connect(self.test_ok)
        self.ui.test_pBut_stop.clicked.connect(self.test_stop)
        self.ui.edit_test_set_ids.textChanged.connect(self.test_set_ids_change)
        self.test_set_ids = self.ui.edit_test_set_ids.text()
        self.ui.test_cbox_format.currentIndexChanged.connect(self.test_format_change)

        # format conversion
        self.ui.test_pBut_coordPath.clicked.connect(self.test_select_CoordPath)
        self.ui.test_edit_clsId.textChanged.connect(self.test_clsId_change)
        self.test_clsId = self.ui.test_edit_clsId.text()
        self.ui.test_pBut_convertOk.clicked.connect(self.convert_ok)

        # manual picking
        self.ui.mpick_ckb_enable.stateChanged.connect(self.mpick_enable_stateChanged)
        if self.ui.mpick_ckb_enable.isChecked():
            self.ui.mpick_slider_labelDiameter.setEnabled(True)
            self.ui.mpick_edit_labelDameter.setEnabled(True)
            self.ui.mpick_sb_classId.setEnabled(True)
            self.ui.mpick_sb_labelWidth.setEnabled(True)
            self.ui.mpick_pBut_setColor.setEnabled(True)
        else:
            self.ui.mpick_slider_labelDiameter.setEnabled(False)
            self.ui.mpick_edit_labelDameter.setEnabled(False)
            self.ui.mpick_sb_classId.setEnabled(False)
            self.ui.mpick_sb_labelWidth.setEnabled(False)
            self.ui.mpick_pBut_setColor.setEnabled(False)
        self.ui.mpick_slider_labelDiameter.valueChanged.connect(self.mpick_labelDiameterChange)
        self.ui.mpick_slider_labelDiameter.setValue(7)
        self.mpick_circle_diameter = self.ui.mpick_slider_labelDiameter.value()
        self.mpick_old_diameter = self.mpick_circle_diameter
        self.ui.mpick_edit_labelDameter.setText(str(self.mpick_circle_diameter))
        self.mpick_circle_width = self.ui.mpick_sb_labelWidth.value()
        self.ui.mpick_sb_labelWidth.valueChanged.connect(self.mpick_labelWidth_change)
        self.mpick_clsId = self.ui.mpick_sb_classId.value()
        self.ui.mpick_sb_classId.valueChanged.connect(self.mpick_classId_change)
        self.ui.mpick_pBut_setColor.clicked.connect(self.mpick_setColor)
        self.color = (255, 0, 0)
        self.mouse_double = False
        self.ui.mpick_pBut_savePath.clicked.connect(self.mpick_savePath)
        self.ui.mpick_edit_savePath.textChanged.connect(self.mpick_savePathChange)
        self.mpick_save_path = self.ui.mpick_edit_savePath.text()
        self.ui.mpick_edit_saveName.textChanged.connect(self.mpick_saveNameChange)
        self.mpick_save_name = self.ui.mpick_edit_saveName.text()
        self.ui.mpick_pBut_save.clicked.connect(self.mpick_save)
        self.ui.mpick_ckb_clear.stateChanged.connect(self.mpick_clear_stateChanged)
        if self.ui.mpick_ckb_clear.isChecked():
            self.self.mpick_coords = []
            self.mpick_coords_np = np.array(self.mpick_coords)

    # manual picking
    def mpick_enable_stateChanged(self):
        if self.ui.mpick_ckb_enable.isChecked():
            self.ui.mpick_slider_labelDiameter.setEnabled(True)
            self.ui.mpick_edit_labelDameter.setEnabled(True)
            self.ui.mpick_sb_classId.setEnabled(True)
            self.ui.mpick_sb_labelWidth.setEnabled(True)
            self.ui.mpick_pBut_setColor.setEnabled(True)
            self.mpick_coords = []
            self.mpick_coords_np = np.array(self.mpick_coords)

            if self.res_label_type == 'Coords' and self.res_label != []:
                self.mpick_coords = self.res_label.tolist()
                self.mpick_coords_np = np.array(self.res_label)
                for idx, xyz in enumerate(self.mpick_coords_np):
                    self.res_show_info(f"{idx}:{xyz}")
                self.mpick_circle_diameter = self.circle_diameter
                self.ui.mpick_slider_labelDiameter.setValue(self.circle_diameter)
                self.mpick_circle_width = self.circle_width
                self.ui.mpick_sb_labelWidth.setValue(self.circle_width)
                self.ui.cbox_circle.setChecked(False)
        else:
            self.ui.mpick_slider_labelDiameter.setEnabled(False)
            self.ui.mpick_edit_labelDameter.setEnabled(False)
            self.ui.mpick_sb_classId.setEnabled(False)
            self.ui.mpick_sb_labelWidth.setEnabled(False)
            self.ui.mpick_pBut_setColor.setEnabled(False)

    def mpick_clear_stateChanged(self):
        if self.ui.mpick_ckb_clear.isChecked():
            self.mpick_coords = []
            self.mpick_coords_np = np.array(self.mpick_coords)
            self.MC_updata()

    def mpick_labelDiameterChange(self):
        self.mpick_circle_diameter = self.ui.mpick_slider_labelDiameter.value()
        self.ui.mpick_edit_labelDameter.setText(str(self.mpick_circle_diameter))
        self.MC_updata()

    def mpick_labelWidth_change(self):
        self.mpick_circle_width = self.ui.mpick_sb_labelWidth.value()
        self.MC_updata()

    def mpick_classId_change(self):
        self.mpick_clsId = self.ui.mpick_sb_classId.value()
        self.MC_updata()

    def mpick_setColor(self):
        Qcolor = QColorDialog.getColor()
        self.color = (Qcolor.red(), Qcolor.green(), Qcolor.blue())
        if self.ui.mpick_ckb_enable:
            colors[self.mpick_clsId - 1] = self.color
        self.MC_updata()

    def res_show_info(self, info):
        self.ui.res_txtB.insertPlainText(f"{info}\n")
        self.ui.res_txtB.ensureCursorVisible()

    def mpick_savePath(self):
        self.mpick_save_path = QFileDialog.getExistingDirectory(self, 'Select the path')
        if self.mpick_save_path != "":
            self.ui.mpick_edit_savePath.setText(self.mpick_save_path)
            self.res_show_info(f"Save path of manual picking: {self.mpick_save_path}")

    def mpick_savePathChange(self):
        self.mpick_save_path = self.ui.mpick_edit_savePath.text()

    def mpick_saveNameChange(self):
        self.mpick_save_name = self.ui.mpick_edit_saveName.text()

    def mpick_save(self):
        if self.mpick_save_path == "" \
                or self.mpick_save_name == "" \
                or self.mpick_coords_np.shape[0] < 1:
            QMessageBox.critical(self, 'Error', 'Incomplete information')
        else:
            os.makedirs(self.mpick_save_path, exist_ok=True)
            np.savetxt(os.path.join(self.mpick_save_path, self.mpick_save_name),
                       self.mpick_coords_np,
                       fmt='%s', delimiter='\t', newline='\n')

    # Train components
    def train_show_info(self, info):
        self.ui.train_txtB.insertPlainText(f"{info}\n")
        self.ui.train_txtB.ensureCursorVisible()

    def train_dsetName_change(self):
        self.train_dsetName = self.ui.train_edit_dsetName.text()
        self.train_show_info(f"Training dataset Name: {self.train_dsetName}")

    def train_modelName_change(self):
        if self.ui.train_cbox_modelName.currentIndex != 0:
            self.train_modelName = self.ui.train_cbox_modelName.currentText()
            self.train_show_info(f"Training model name: {self.train_modelName}")

    def train_cls_num_change(self):
        self.train_cls_num = self.ui.train_sb_clsNum.value()
        self.train_show_info(f"Number of training classes: {self.train_cls_num}")

    def train_batchSize_change(self):
        self.train_batchSize = self.ui.train_sb_batchSize.value()
        self.train_show_info(f"Training - batch size: {self.train_batchSize}")

    def train_patchSize_change(self):
        self.train_patchSize = self.ui.train_sb_patchSize.value()
        self.train_show_info(f"Training -  patch size: {self.train_patchSize}")

    def train_paddingSize_change(self):
        self.train_paddingSize = self.ui.train_sb_paddingSize.value()
        self.train_show_info(f"Training -  padding size: {self.train_paddingSize}")

    def train_maxEpochs_change(self):
        self.train_maxEpochs = self.ui.train_sb_maxEpochs.value()
        self.train_show_info(f"Training - max epochs: {self.train_maxEpochs}")

    def train_segThresh_change(self):
        self.train_segThresh = float(self.ui.train_edit_segThresh.text())
        self.train_show_info(f"Training - segmentation threshold: {self.train_segThresh}")

    def train_lr_change(self):
        self.train_lr = float(self.ui.train_edit_lr.text())
        self.train_show_info(f"Training - learning rate: {self.train_lr}")

    def train_gpuIds_change(self):
        self.train_gpuIds = self.ui.train_edit_gpuIds.text()
        self.train_show_info(f"Training - gpu ids: {self.train_gpuIds}")

    def train_set_ids_change(self):
        self.train_set_ids = self.ui.edit_train_set_ids.text()
        self.train_show_info(f"Train dataset ids: {self.train_set_ids}")

    def val_set_ids_change(self):
        self.val_set_ids = self.ui.edit_val_set_ids.text()
        self.train_show_info(f"Val dataset ids: {self.val_set_ids}")

    def train_saveConfigs(self):
        if self.ui.train_edit_segThresh.text() == "" \
                or self.ui.edit_train_set_ids.text() == "" \
                or self.ui.edit_val_set_ids.text() == "" \
                or self.ui.train_edit_gpuIds.text() == "" \
                or self.ui.train_edit_dsetName.text() == "" \
                or (self.ui.edit_load_configs.text() == "" and self.ui.train_edit_loadConfigs == ""):
            QMessageBox.critical(self, 'Error', 'Incomplete information')
        else:
            base_path = self.c2l_basePath if isinstance(self.c2l_basePath, str) else self.c2l_basePath[0]
            input_norm = self.input_norm if isinstance(self.input_norm, str) else self.input_norm[0]
            tomo_name = 'data_std' if 'standardization' in input_norm else 'data_norm'
            label_name = (self.label_type if isinstance(self.label_type, str) else self.label_type[0]) + \
                         str(self.label_diameter if isinstance(self.label_diameter, int) else self.label_diameter[0])
            # ocp_name = (self.ocp_type if isinstance(self.ocp_type, str) else self.ocp_type[0]) + \
            #            str(self.ocp_diameter if isinstance(self.ocp_diameter, int) else self.ocp_diameter[0])
            ocp_name = 'data_ocp'  # + str(self.ocp_diameter if isinstance(self.ocp_diameter, int) else self.ocp_diameter[0])
            self.train_configs = dict(
                dset_name=(self.train_dsetName if isinstance(self.train_dsetName, str) else self.train_dsetName[
                    0]),
                base_path=self.c2l_basePath if isinstance(self.c2l_basePath, str) else self.c2l_basePath[0],
                coord_path=os.path.join(base_path, 'coords'),
                coord_format=self.coord_format if isinstance(self.coord_format, str) else self.coord_format[0],
                tomo_path=os.path.join(base_path, tomo_name),
                tomo_format=self.tomo_format if isinstance(self.tomo_format, str) else self.tomo_format[0],
                num_cls=self.train_cls_num if isinstance(self.train_cls_num, int) else self.train_cls_num[0],
                label_name=(self.label_type if isinstance(self.label_type, str) else self.label_type[0]) + \
                           str(self.label_diameter if isinstance(self.label_diameter, int) else self.label_diameter[0]),
                label_path=os.path.join(base_path, label_name),
                label_type=self.label_type if isinstance(self.label_type, str) else self.label_type[0],
                label_diameter=self.label_diameter if isinstance(self.label_diameter, int) else self.label_diameter[0],
                ocp_type=self.ocp_type if isinstance(self.ocp_type, str) else self.ocp_type[0],
                ocp_diameter=self.ocp_diameter if isinstance(self.ocp_diameter, str) else self.ocp_diameter[0],
                ocp_name=ocp_name,
                ocp_path=os.path.join(base_path, ocp_name),
                norm_type=self.input_norm if isinstance(self.input_norm, str) else self.input_norm[0],
                model_name=self.train_modelName if isinstance(self.train_modelName, str) else self.train_modelName[0],
                train_set_ids=self.train_set_ids if isinstance(self.train_set_ids, str) else self.train_set_ids[0],
                val_set_ids=self.val_set_ids if isinstance(self.val_set_ids, str) else self.val_set_ids[0],
                batch_size=self.train_batchSize if isinstance(self.train_batchSize, int) else self.train_batchSize[0],
                patch_size=self.train_patchSize if isinstance(self.train_patchSize, int) else self.train_patchSize[0],
                padding_size=self.train_paddingSize if isinstance(self.train_paddingSize, int) else
                self.train_paddingSize[0],
                lr=self.train_lr if isinstance(self.train_lr, float) else self.train_lr[0],
                max_epochs=self.train_maxEpochs if isinstance(self.train_maxEpochs, int) else self.train_maxEpochs[0],
                seg_thresh=self.train_segThresh if isinstance(self.train_segThresh, float) else self.train_segThresh[0],
                gpu_ids=self.train_gpuIds if isinstance(self.train_gpuIds, str) else self.train_gpuIds[0],
            )
            config_save_path = os.path.join(base_path, 'configs')
            os.makedirs(config_save_path, exist_ok=True)
            self.train_config_save_path = f"{config_save_path}/{self.train_dsetName}.py"
            with open(self.train_config_save_path, 'w') as f:
                f.write("train_configs=")
                json.dump(self.train_configs, f, separators=(',\n'+' '*len('train_configs={'), ': '))
            self.train_show_info(f"save train configs to '{config_save_path}/{self.train_dsetName}.py'")

    def train_loadConfigs(self):
        self.train_config_file, _ = QFileDialog.getOpenFileName(self, 'Select the training configs')

        if self.train_config_file != "":
            self.ui.train_edit_loadConfigs.setText(self.train_config_file)
            self.train_show_info(f"Load training configs: {self.train_config_file}")

        train_config_name = self.train_config_file.split('/')[-1][:-3]
        base_config = '.'.join(self.train_config_file.split('/')[:-1])

        # print(sys.modules.keys())
        # # delete the package in the sys.modules
        # if f"{base_config}.{train_config_name}" in list(sys.modules.keys()):
        #     del sys.modules[f"{base_config}.{train_config_name}"]
        # config = importlib.import_module(f"{base_config}.{train_config_name}")
        # self.train_configs = config.train_configs
        with open(self.train_config_file, 'r') as f:
            self.train_configs = json.loads(''.join(f.readlines()).lstrip('train_configs='))

        self.train_dsetName = self.train_configs['dset_name']
        self.base_path = self.train_configs['base_path']
        self.coord_path = self.train_configs['coord_path']
        self.coord_format = self.train_configs['coord_format']
        self.tomo_path = self.train_configs['tomo_path']
        self.tomo_format = self.train_configs['tomo_format']
        self.train_numCls = self.train_configs['num_cls']
        self.label_name = self.train_configs['label_name']
        self.label_path = self.train_configs['label_path']
        self.label_type = self.train_configs['label_type']
        self.label_diameter = self.train_configs['label_diameter']
        self.ocp_type = self.train_configs['ocp_type']
        self.ocp_diameter = self.train_configs['ocp_diameter']
        self.ocp_name = self.train_configs['ocp_name']
        self.ocp_path = self.train_configs['ocp_path']
        self.norm_type = self.train_configs['norm_type']
        self.train_modelName = self.train_configs['model_name']
        self.train_set_ids = self.train_configs['train_set_ids']
        self.val_set_ids = self.train_configs['val_set_ids']
        self.train_batchSize = self.train_configs['batch_size']
        self.train_patchSize = self.train_configs['patch_size']
        self.train_paddingSize = self.train_configs['padding_size']
        self.train_lr = self.train_configs['lr']
        self.train_maxEpochs = self.train_configs['max_epochs']
        self.train_segThresh = self.train_configs['seg_thresh']
        self.train_gpuIds = self.train_configs['gpu_ids']

        self.train_dsetName = self.train_dsetName if isinstance(self.train_dsetName, str) else self.train_dsetName[0]
        self.base_path = self.base_path if isinstance(self.base_path, str) else self.base_path[0]
        self.coord_path = self.coord_path if isinstance(self.coord_path, str) else self.coord_path[0]
        self.coord_format = self.coord_format if isinstance(self.coord_format, str) else self.coord_format[0]
        self.tomo_path = self.tomo_path if isinstance(self.tomo_path, str) else self.tomo_path[0]
        self.tomo_format = self.tomo_format if isinstance(self.tomo_format, str) else self.tomo_format[0]
        self.train_numCls = self.train_numCls if isinstance(self.train_numCls, int) else self.train_numCls[0]
        self.label_name = self.label_name if isinstance(self.label_name, str) else self.label_name[0]
        self.label_path = self.label_path if isinstance(self.label_path, str) else self.label_path[0]
        self.label_type = self.label_type if isinstance(self.label_type, str) else self.label_type[0]
        self.label_diameter = self.label_diameter if isinstance(self.label_diameter, int) else self.label_diameter[0]
        self.ocp_type = self.ocp_type if isinstance(self.ocp_type, str) else self.ocp_type[0]
        self.ocp_diameter = self.ocp_diameter if isinstance(self.ocp_diameter, str) else self.ocp_diameter[0]
        self.ocp_name = self.ocp_name if isinstance(self.ocp_name, str) else self.ocp_name[0]
        self.ocp_path = self.ocp_path if isinstance(self.ocp_path, str) else self.ocp_path[0]
        self.norm_type = self.norm_type if isinstance(self.norm_type, str) else self.norm_type[0]
        self.val_set_ids = self.val_set_ids if isinstance(self.val_set_ids, str) else self.val_set_ids[
            0]
        self.train_set_ids = self.train_set_ids if isinstance(self.train_set_ids, str) else self.train_set_ids[
            0]
        self.train_modelName = self.train_modelName if isinstance(self.train_modelName, str) else self.train_modelName[
            0]
        self.train_batchSize = self.train_batchSize if isinstance(self.train_batchSize, int) else self.train_batchSize[
            0]
        self.train_patchSize = self.train_patchSize if isinstance(self.train_patchSize, int) else self.train_patchSize[
            0]
        self.train_paddingSize = self.train_paddingSize if isinstance(self.train_paddingSize, int) else \
            self.train_paddingSize[0]
        self.train_lr = self.train_lr if isinstance(self.train_lr, float) else \
            self.train_lr[0]
        self.train_maxEpochs = self.train_maxEpochs if isinstance(self.train_maxEpochs, int) else self.train_maxEpochs[
            0]
        self.train_segThresh = self.train_segThresh if isinstance(self.train_segThresh, float) else \
            self.train_segThresh[0]
        self.train_gpuIds = self.train_gpuIds if isinstance(self.train_gpuIds, str) else self.train_gpuIds[0]
        self.c2l_basePath = self.base_path

        self.ui.edit_base_path.setText(self.base_path)
        self.ui.cbox_coord_format.setCurrentText(self.coord_format)
        self.ui.cbox_tomo_format.setCurrentText(self.tomo_format)
        self.ui.cbox_label_type.setCurrentText(self.label_type)
        self.ui.cbox_ocp_type.setCurrentText(self.ocp_type)
        self.ui.sb_cls_num.setValue(self.train_numCls)
        self.ui.sb_label_diameter.setValue(self.label_diameter)
        self.ui.edit_ocp_diameter.setText(self.ocp_diameter)
        self.ui.train_edit_dsetName.setText(self.train_dsetName.split('.')[0])
        self.ui.train_cbox_modelName.setCurrentText(self.train_modelName)
        self.ui.train_sb_clsNum.setValue(self.train_numCls)
        self.ui.train_sb_batchSize.setValue(self.train_batchSize)
        self.ui.train_sb_patchSize.setValue(self.train_patchSize)
        self.ui.train_sb_paddingSize.setValue(self.train_paddingSize)
        self.ui.train_edit_lr.setText(str(self.train_lr))
        self.ui.train_sb_maxEpochs.setValue(self.train_maxEpochs)
        self.ui.train_edit_segThresh.setText(str(self.train_segThresh))
        self.ui.train_edit_gpuIds.setText(self.train_gpuIds)
        self.ui.edit_train_set_ids.setText(self.train_set_ids)
        self.ui.edit_val_set_ids.setText(self.val_set_ids)

        if self.input_norm == "standardization":
            self.ui.radB_norm.setChecked(False)
            self.ui.radB_std.setChecked(True)
        elif self.input_norm == "normalization":
            self.ui.radB_std.setChecked(False)
            self.ui.radB_norm.setChecked(True)

        self.train_show_info('*' * 100)

        for i in self.train_configs.keys():
            self.train_show_info(f'{i}: {self.train_configs[i]}')

        self.train_show_info('*' * 100)

    def train_loadConfigs_v1(self):
        self.train_config_file = self.train_config_save_path

        if self.train_config_file != "":
            self.ui.train_edit_loadConfigs.setText(self.train_config_file)
            self.train_show_info(f"Load training configs: {self.train_config_file}")

        train_config_name = self.train_config_file.split('/')[-1][:-3]
        base_config = '.'.join(self.train_config_file.split('/')[:-1])

        # print(sys.modules.keys())
        # # delete the package in the sys.modules
        # if f"{base_config}.{train_config_name}" in list(sys.modules.keys()):
        #     del sys.modules[f"{base_config}.{train_config_name}"]
        # config = importlib.import_module(f"{base_config}.{train_config_name}")
        # self.train_configs = config.train_configs
        with open(self.train_config_file, 'r') as f:
            self.train_configs = json.loads(''.join(f.readlines()).lstrip('train_configs='))

        self.train_dsetName = self.train_configs['dset_name']
        self.base_path = self.train_configs['base_path']
        self.coord_path = self.train_configs['coord_path']
        self.coord_format = self.train_configs['coord_format']
        self.tomo_path = self.train_configs['tomo_path']
        self.tomo_format = self.train_configs['tomo_format']
        self.train_numCls = self.train_configs['num_cls']
        self.label_name = self.train_configs['label_name']
        self.label_path = self.train_configs['label_path']
        self.label_type = self.train_configs['label_type']
        self.label_diameter = self.train_configs['label_diameter']
        self.ocp_type = self.train_configs['ocp_type']
        self.ocp_diameter = self.train_configs['ocp_diameter']
        self.ocp_name = self.train_configs['ocp_name']
        self.ocp_path = self.train_configs['ocp_path']
        self.norm_type = self.train_configs['norm_type']
        self.train_set_ids = self.train_configs['train_set_ids']
        self.val_set_ids = self.train_configs['val_set_ids']
        self.train_modelName = self.train_configs['model_name']
        self.train_batchSize = self.train_configs['batch_size']
        self.train_patchSize = self.train_configs['patch_size']
        self.train_paddingSize = self.train_configs['padding_size']
        self.train_lr = self.train_configs['lr']
        self.train_maxEpochs = self.train_configs['max_epochs']
        self.train_segThresh = self.train_configs['seg_thresh']
        self.train_gpuIds = self.train_configs['gpu_ids']

        self.train_dsetName = self.train_dsetName if isinstance(self.train_dsetName, str) else self.train_dsetName[0]
        self.base_path = self.base_path if isinstance(self.base_path, str) else self.base_path[0]
        self.coord_path = self.coord_path if isinstance(self.coord_path, str) else self.coord_path[0]
        self.coord_format = self.coord_format if isinstance(self.coord_format, str) else self.coord_format[0]
        self.tomo_path = self.tomo_path if isinstance(self.tomo_path, str) else self.tomo_path[0]
        self.tomo_format = self.tomo_format if isinstance(self.tomo_format, str) else self.tomo_format[0]
        self.train_numCls = self.train_numCls if isinstance(self.train_numCls, int) else self.train_numCls[0]
        self.label_name = self.label_name if isinstance(self.label_name, str) else self.label_name[0]
        self.label_path = self.label_path if isinstance(self.label_path, str) else self.label_path[0]
        self.label_type = self.label_type if isinstance(self.label_type, str) else self.label_type[0]
        self.label_diameter = self.label_diameter if isinstance(self.label_diameter, int) else self.label_diameter[0]
        self.ocp_type = self.ocp_type if isinstance(self.ocp_type, str) else self.ocp_type[0]
        self.ocp_diameter = self.ocp_diameter if isinstance(self.ocp_diameter, str) else self.ocp_diameter[0]
        self.ocp_name = self.ocp_name if isinstance(self.ocp_name, str) else self.ocp_name[0]
        self.ocp_path = self.ocp_path if isinstance(self.ocp_path, str) else self.ocp_path[0]
        self.norm_type = self.norm_type if isinstance(self.norm_type, str) else self.norm_type[0]
        self.val_set_ids = self.val_set_ids if isinstance(self.val_set_ids, str) else self.val_set_ids[
            0]
        self.train_set_ids = self.train_set_ids if isinstance(self.train_set_ids, str) else self.train_set_ids[
            0]
        self.train_modelName = self.train_modelName if isinstance(self.train_modelName, str) else self.train_modelName[
            0]
        self.train_batchSize = self.train_batchSize if isinstance(self.train_batchSize, int) else self.train_batchSize[
            0]
        self.train_patchSize = self.train_patchSize if isinstance(self.train_patchSize, int) else self.train_patchSize[
            0]
        self.train_paddingSize = self.train_paddingSize if isinstance(self.train_paddingSize, int) else \
            self.train_paddingSize[0]
        self.train_lr = self.train_lr if isinstance(self.train_lr, float) else \
            self.train_lr[0]
        self.train_maxEpochs = self.train_maxEpochs if isinstance(self.train_maxEpochs, int) else self.train_maxEpochs[
            0]
        self.train_segThresh = self.train_segThresh if isinstance(self.train_segThresh, float) else \
            self.train_segThresh[0]
        self.train_gpuIds = self.train_gpuIds if isinstance(self.train_gpuIds, str) else self.train_gpuIds[0]
        self.c2l_basePath = self.base_path

        self.ui.edit_base_path.setText(self.base_path)
        self.ui.cbox_coord_format.setCurrentText(self.coord_format)
        self.ui.cbox_tomo_format.setCurrentText(self.tomo_format)
        self.ui.cbox_label_type.setCurrentText(self.label_type)
        self.ui.cbox_ocp_type.setCurrentText(self.ocp_type)
        self.ui.sb_cls_num.setValue(self.train_numCls)
        self.ui.sb_label_diameter.setValue(self.label_diameter)
        self.ui.edit_ocp_diameter.setText(self.ocp_diameter)
        self.ui.train_edit_dsetName.setText(self.train_dsetName.split('.')[0])
        self.ui.train_cbox_modelName.setCurrentText(self.train_modelName)
        self.ui.train_sb_clsNum.setValue(self.train_numCls)
        self.ui.train_sb_batchSize.setValue(self.train_batchSize)
        self.ui.train_sb_patchSize.setValue(self.train_patchSize)
        self.ui.train_sb_paddingSize.setValue(self.train_paddingSize)
        self.ui.train_edit_lr.setText(str(self.train_lr))
        self.ui.train_sb_maxEpochs.setValue(self.train_maxEpochs)
        self.ui.train_edit_segThresh.setText(str(self.train_segThresh))
        self.ui.train_edit_gpuIds.setText(self.train_gpuIds)
        self.ui.edit_train_set_ids.setText(self.train_set_ids)
        self.ui.edit_val_set_ids.setText(self.val_set_ids)

        if self.input_norm == "standardization":
            self.ui.radB_norm.setChecked(False)
            self.ui.radB_std.setChecked(True)
        elif self.input_norm == "normalization":
            self.ui.radB_std.setChecked(False)
            self.ui.radB_norm.setChecked(True)

        self.train_show_info('*' * 100)

        for i in self.train_configs.keys():
            self.train_show_info(f'{i}: {self.train_configs[i]}')

        self.train_show_info('*' * 100)

    def train_dsetList(self):
        coord_path = f"{self.c2l_basePath}/coords/num_name.csv"
        coord_data = np.loadtxt(coord_path, delimiter='\t', dtype=str).reshape(-1, 3)
        self.train_show_info(f'*' * 100)
        self.train_show_info(f"Dataset list:")
        self.train_show_info(f"Number\t Name \t Cls_id")
        for item in coord_data:
            self.train_show_info(f"{item[0].split('.')[0]}\t{item[1]}\t{item[2]}")
        self.train_show_info(f'*' * 100)

    def train_show_clear(self):
        self.ui.train_txtB.clear()

    def train_stop(self):
        try:
            # self.train_thread.n = 0
            # self.train_thread.join()
            # os.system(f"kill -9 {self.train_thread.pid_num}")
            stop_thread(self.train_thread)
            # self.train_t.pause()
            # os.system(f"kill {self.train_pid}")
            # self.train_thread.terminate()
        except:
            pass
        self.train_show_info('*' * 100)
        self.train_show_info('Training Stopped')
        self.train_show_info('*' * 100)

    def train_ok(self):
        self.train_saveConfigs()
        self.train_loadConfigs_v1()
        if self.ui.train_edit_segThresh.text() == "" \
                or self.ui.train_edit_gpuIds.text() == "" \
                or self.ui.train_edit_dsetName.text() == "" \
                or (self.ui.edit_load_configs.text() == "" and self.ui.train_edit_loadConfigs == ""):
            QMessageBox.critical(self, 'Error', 'Incomplete information')
        else:
            self.train_show_info('*' * 100)
            self.train_show_info('Final training configuration parameters')
            self.train_show_info('*' * 100)
            self.train_show_info(f"Training dataset Name: {self.train_dsetName}")
            self.train_show_info(f"Training model name: {self.train_modelName}")
            self.train_show_info(f"Number of training classes: {self.train_cls_num}")
            self.train_show_info(f"Training - batch size: {self.train_batchSize}")
            self.train_show_info(f"Training -  patch size: {self.train_patchSize}")
            self.train_show_info(f"Training -  padding size: {self.train_paddingSize}")
            self.train_show_info(f"Training - max epochs: {self.train_maxEpochs}")
            self.train_show_info(f"Training - segmentation threshold: {self.train_segThresh}")
            self.train_show_info(f"Training - gpu ids: {self.train_gpuIds}")
            self.train_show_info('*' * 100)

            self.csv_path = f"{self.base_path}/coords/num_name.csv"
            self.csv_data = np.loadtxt(self.csv_path, delimiter='\t', dtype=str).reshape(-1, 3)

            train_list = []
            for item in self.train_set_ids.split(','):
                if '-' in item:
                    tmp = [int(i) for i in item.split('-')]
                    train_list.extend(np.arange(tmp[0], tmp[1] + 1).tolist())
                else:
                    train_list.append(int(item))

            val_list = []
            for item in self.val_set_ids.split(','):
                if '-' in item:
                    tmp = [int(i) for i in item.split('-')]
                    val_list.extend(np.arange(tmp[0], tmp[1] + 1).tolist())
                else:
                    val_list.append(int(item))

            csv_data_new = []
            others = []

            for item in self.csv_data:
                if (int(item[2]) in train_list) and int(item[2]) not in val_list:
                    csv_data_new.insert(0, item.tolist())
                elif int(item[2]) not in val_list:
                    others.append(item.tolist())

            for item in self.csv_data:
                if int(item[2]) in val_list:
                    csv_data_new.append(item.tolist())

            csv_data_new.extend(others)
            np.savetxt(self.csv_path,
                       np.array(csv_data_new),
                       delimiter='\t',
                       newline='\n',
                       fmt='%s')

            if self.train_cls_num == 1:
                use_sigmoid = True
                use_softmax = False
                train_cls_num = self.train_cls_num
            elif self.train_cls_num > 1:
                train_cls_num = self.train_cls_num + 1
                use_sigmoid = False
                use_softmax = True

            options = BaseOptions()
            args = options.gather_options()
            args.block_size = self.train_patchSize
            args.num_classes = train_cls_num
            args.loss_func_seg = 'Dice'
            args.optim = 'AdamW'
            args.weight_decay = 0.01
            args.learning_rate = self.train_lr
            args.batch_size = self.train_batchSize
            args.max_epoch = self.train_maxEpochs
            args.network = self.train_modelName
            args.use_bg = True
            args.use_IP = True
            args.use_coord = True
            args.use_sigmoid = use_sigmoid
            args.use_softmax = use_softmax
            args.threshold = self.train_segThresh
            # args.data_split = [0, 1, 0, 1, 0, 1]
            # print(train_list, val_list)
            val_first = len(train_list) if val_list[0] not in train_list else len(train_list) - 1
            self.train_show_info(f"val_first:{val_first:.0f}")
            args.data_split = [0, len(train_list),  # train
                               val_first, val_first + 1,  # val
                               val_first, val_first + 1]  # test_val
            self.train_show_info(f"data_split:{args.data_split}")
            args.f_maps = [24, 48, 72, 108]
            args.random_num = 0
            args.configs = os.path.join(self.base_path, 'configs', f'{self.train_dsetName}.py')
            args.loader_type = 'dataloader_DynamicLoad'
            args.test_use_pad = True
            args.pad_size = [self.train_paddingSize]
            args.test_mode = 'val'
            args.val_batch_size = self.train_batchSize
            args.val_block_size = self.train_patchSize
            args.scheduler = 'OneCycleLR'
            args.gpu_id = [int(i) for i in self.train_gpuIds.split(',')]
            args.meanPool_NMS = True

            """
            threading.Thread
            """
            self.train_emit = EmittingStr()
            self.train_emit.textWritten.connect(self.train_show_info)
            self.train_thread = threading.Thread(target=train_func, args=(args, self.train_emit))
            # self.train_thread = myThread(1, train, args, self.train_emit)
            self.train_thread.start()

            # self.train_thread.join()
            # print(self.train_pid)

            """
            Class threading
            """
            # self.train_t = Concur(train, args, self.train_emit)
            # self.train_t.start()
            # self.train_t.resume()

            """
            QThread
            """
            # class Qthread_job(QThread):
            #     def __init__(self, job, args, stdout):
            #         super(Qthread_job, self).__init__()
            #         self.job = job
            #         self.args = args
            #         self.stdout = stdout
            #     def run(self):
            #         self.job(self.args, self.stdout)
            #
            # self.train_thread = Qthread_job(train, args, self.train_emit)
            # self.train_thread.start()

    """
    test
    """

    def test_set_ids_change(self):
        self.test_set_ids = self.ui.edit_test_set_ids.text()
        self.test_show_info(f"Test dataset ids: {self.test_set_ids}")

    def test_load_trainConfigs(self):
        self.train_config_file, _ = QFileDialog.getOpenFileName(self, 'Select the training configs')

        if self.train_config_file != "":
            self.ui.train_edit_loadConfigs.setText(self.train_config_file)
            self.test_show_info(f"Load training configs: {self.train_config_file}")

        with open(self.train_config_file, 'r') as f:
            self.train_configs = json.loads(''.join(f.readlines()).lstrip('train_configs='))

        self.train_dsetName = self.train_configs['dset_name']
        self.base_path = self.train_configs['base_path']
        self.coord_path = self.train_configs['coord_path']
        self.coord_format = self.train_configs['coord_format']
        self.tomo_path = self.train_configs['tomo_path']
        self.tomo_format = self.train_configs['tomo_format']
        self.train_numCls = self.train_configs['num_cls']
        self.label_name = self.train_configs['label_name']
        self.label_path = self.train_configs['label_path']
        self.label_type = self.train_configs['label_type']
        self.label_diameter = self.train_configs['label_diameter']
        self.ocp_type = self.train_configs['ocp_type']
        self.ocp_diameter = self.train_configs['ocp_diameter']
        self.ocp_name = self.train_configs['ocp_name']
        self.ocp_path = self.train_configs['ocp_path']
        self.norm_type = self.train_configs['norm_type']
        self.train_modelName = self.train_configs['model_name']
        self.train_batchSize = self.train_configs['batch_size']
        self.train_patchSize = self.train_configs['patch_size']
        self.train_paddingSize = self.train_configs['padding_size']
        self.train_maxEpochs = self.train_configs['max_epochs']
        self.train_segThresh = self.train_configs['seg_thresh']
        self.train_gpuIds = self.train_configs['gpu_ids']
        self.test_gpuIds = self.train_configs['gpu_ids']

        self.train_dsetName = self.train_dsetName if isinstance(self.train_dsetName, str) else self.train_dsetName[0]
        self.base_path = self.base_path if isinstance(self.base_path, str) else self.base_path[0]
        self.coord_path = self.coord_path if isinstance(self.coord_path, str) else self.coord_path[0]
        self.coord_format = self.coord_format if isinstance(self.coord_format, str) else self.coord_format[0]
        self.tomo_path = self.tomo_path if isinstance(self.tomo_path, str) else self.tomo_path[0]
        self.tomo_format = self.tomo_format if isinstance(self.tomo_format, str) else self.tomo_format[0]
        self.train_numCls = self.train_numCls if isinstance(self.train_numCls, int) else self.train_numCls[0]
        self.label_name = self.label_name if isinstance(self.label_name, str) else self.label_name[0]
        self.label_path = self.label_path if isinstance(self.label_path, str) else self.label_path[0]
        self.label_type = self.label_type if isinstance(self.label_type, str) else self.label_type[0]
        self.label_diameter = self.label_diameter if isinstance(self.label_diameter, int) else self.label_diameter[0]
        self.ocp_type = self.ocp_type if isinstance(self.ocp_type, str) else self.ocp_type[0]
        self.ocp_diameter = self.ocp_diameter if isinstance(self.ocp_diameter, str) else self.ocp_diameter[0]
        self.ocp_name = self.ocp_name if isinstance(self.ocp_name, str) else self.ocp_name[0]
        self.ocp_path = self.ocp_path if isinstance(self.ocp_path, str) else self.ocp_path[0]
        self.norm_type = self.norm_type if isinstance(self.norm_type, str) else self.norm_type[0]
        self.train_modelName = self.train_modelName if isinstance(self.train_modelName, str) else self.train_modelName[
            0]
        self.train_batchSize = self.train_batchSize if isinstance(self.train_batchSize, int) else self.train_batchSize[
            0]
        self.train_patchSize = self.train_patchSize if isinstance(self.train_patchSize, int) else self.train_patchSize[
            0]
        self.train_paddingSize = self.train_paddingSize if isinstance(self.train_paddingSize, int) else \
            self.train_paddingSize[0]
        self.train_maxEpochs = self.train_maxEpochs if isinstance(self.train_maxEpochs, int) else self.train_maxEpochs[
            0]
        self.train_segThresh = self.train_segThresh if isinstance(self.train_segThresh, float) else \
            self.train_segThresh[0]
        self.train_gpuIds = self.train_gpuIds if isinstance(self.train_gpuIds, str) else self.train_gpuIds[0]
        self.c2l_basePath = self.base_path

        self.ui.edit_base_path.setText(self.base_path)
        self.ui.cbox_coord_format.setCurrentText(self.coord_format)
        self.ui.cbox_tomo_format.setCurrentText(self.tomo_format)
        self.ui.cbox_label_type.setCurrentText(self.label_type)
        self.label_type = self.ui.cbox_label_type.currentText()
        self.ui.cbox_ocp_type.setCurrentText(self.ocp_type)
        self.ui.sb_cls_num.setValue(self.train_numCls)
        self.ui.sb_label_diameter.setValue(self.label_diameter)
        self.ui.edit_ocp_diameter.setText(self.ocp_diameter)
        self.ui.train_edit_dsetName.setText(self.train_dsetName.split('.')[0])
        self.ui.train_cbox_modelName.setCurrentText(self.train_modelName)
        self.ui.train_sb_clsNum.setValue(self.train_numCls)
        self.ui.train_sb_batchSize.setValue(self.train_batchSize)
        self.ui.train_sb_patchSize.setValue(self.train_patchSize)
        self.ui.train_sb_paddingSize.setValue(self.train_paddingSize)
        self.ui.train_sb_maxEpochs.setValue(self.train_maxEpochs)
        self.ui.train_edit_segThresh.setText(str(self.train_segThresh))
        self.ui.train_edit_gpuIds.setText(self.train_gpuIds)
        self.ui.test_sb_patchSize.setValue(self.train_patchSize)
        self.ui.test_sb_paddingSize.setValue(self.train_paddingSize)
        self.ui.test_edit_segThresh.setText(str(self.train_segThresh))
        self.ui.test_edit_trainConfigs.setText(self.train_config_file)
        self.ui.test_edit_gpuIds.setText(self.train_configs['gpu_ids'])

        if self.input_norm == "standardization":
            self.ui.radB_norm.setChecked(False)
            self.ui.radB_std.setChecked(True)
        elif self.input_norm == "normalization":
            self.ui.radB_std.setChecked(False)
            self.ui.radB_norm.setChecked(True)

        self.test_show_info('*' * 100)

        for i in self.train_configs.keys():
            self.test_show_info(f'{i}: {self.train_configs[i]}')

        self.test_show_info('*' * 100)

    def test_stop(self):
        try:
            # self.test_thread.quit()
            stop_thread(self.test_thread)
        except:
            pass

        self.test_show_info('*' * 100)
        self.test_show_info('Testing stopped!')
        self.test_show_info('*' * 100)

    def test_ok(self):
        if self.ui.test_sb_patchSize.text() == "" \
                or self.ui.test_sb_paddingSize.text() == "" \
                or self.ui.test_edit_segThresh.text() == "" \
                or self.ui.test_edit_weightPath.text() == "" \
                or self.ui.train_edit_loadConfigs.text() == "":
            QMessageBox.critical(self, 'Error', 'Incomplete information')
            return 0
        else:
            self.test_show_info('*' * 100)
            self.train_show_info('Final inference configuration parameters')
            self.train_show_info('*' * 100)
            self.train_show_info(f"Dataset Name: {self.train_dsetName}")
            self.train_show_info(f"Model name: {self.train_modelName}")
            self.train_show_info(f"Weight path: {self.train_weightPath}")
            self.train_show_info(f"Number of training classes: {self.train_cls_num}")
            # self.train_show_info(f"Inference - batch size: {self.test_batchSize}")
            self.train_show_info(f"Testing -  patch size: {self.test_patchSize}")
            self.train_show_info(f"Testing -  padding size: {self.test_paddingSize}")
            self.train_show_info(f"Testing - segmentation threshold: {self.test_segThresh}")
            self.train_show_info(f"Testing - gpu ids: {self.test_gpuIds}")
            self.train_show_info('*' * 100)

        self.train_configs['label_path'] = None
        self.train_configs['ocp_path'] = None

        if self.train_cls_num == 1:
            use_sigmoid = True
            use_softmax = False
            train_cls_num = self.train_cls_num
        elif self.train_cls_num > 1:
            use_sigmoid = False
            use_softmax = True
            train_cls_num = self.train_cls_num + 1

        dset_list = np.array(
            [i[:-(len(i.split('.')[-1]) + 1)] for i in os.listdir(self.tomo_path) if self.tomo_format in i])
        dset_num = dset_list.shape[0]
        num_name = np.concatenate([np.arange(dset_num).reshape(-1, 1), dset_list.reshape(-1, 1)], axis=1)
        np.savetxt(os.path.join(self.tomo_path, 'num_name.csv'),
                   num_name,
                   delimiter='\t',
                   fmt='%s',
                   newline='\n')

        options = BaseOptions()
        args = options.gather_options()
        args.configs = self.train_config_file
        args.loader_type = 'dataloader_DynamicLoad'
        args.num_classes = train_cls_num
        args.use_bg = True
        args.use_IP = True
        args.use_coord = True
        args.test_use_pad = True
        args.network = self.train_modelName
        args.f_maps = [24, 48, 72, 108]
        args.batch_size = self.train_batchSize
        args.block_size = self.test_patchSize
        args.use_sigmoid = use_sigmoid
        args.use_softmax = use_softmax
        args.checkpoints = self.train_weightPath
        args.threshold = self.test_segThresh
        args.pad_size = [self.test_paddingSize]
        args.data_split = [0, 1, 0, 1, 0, 1]
        args.use_seg = True
        args.use_eval = False
        args.save_pred = False
        args.test_idxs = np.arange(dset_num)
        args.save_mrc = False
        args.test_mode = 'test_only'
        args.out_name = 'PredictedLabels'
        args.de_duplication = True
        args.de_dup_fmt = 'fmt4'
        args.mini_dist = sorted([int(i) // 2 + 1 for i in self.ocp_diameter.split(',')])[0]
        args.gpu_id = [int(i) for i in self.test_gpuIds.split(',')]
        args.meanPool_NMS = True

        """
        QThread
        """
        # class Qthread_job(QThread):
        #     def __init__(self, job, args):
        #         super(Qthread_job, self).__init__()
        #         self.job = job
        #         self.args = args
        #     def run(self):
        #         self.job(self.args)
        #
        # self.test_thread = Qthread_job(test, args)
        # self.test_thread.start()

        """
        threading.Thread
        """
        self.test_emit = EmittingStr()
        self.test_emit.textWritten.connect(self.test_show_info)
        self.test_thread = threading.Thread(target=test_func, args=(args, self.test_emit))
        self.test_thread.start()

    def test_gpuIds_change(self):
        self.test_gpuIds = self.ui.test_edit_gpuIds.text()
        self.test_show_info(f"Testing - gpu ids: {self.test_gpuIds}")

    def test_show_info(self, info):
        self.ui.test_txtB.insertPlainText(f"{info}\n")
        self.ui.test_txtB.ensureCursorVisible()

    def test_load_weightPath(self):
        self.train_weightPath, _ = QFileDialog.getOpenFileName(self, 'Select the training weight')

        if self.train_weightPath != "":
            self.ui.test_edit_weightPath.setText(self.train_weightPath)
            self.test_show_info(f"Load training weight: {self.train_weightPath}")

    def test_patchSize_change(self):
        self.test_patchSize = self.ui.test_sb_patchSize.value()
        self.test_show_info(f"Testing -  patch size: {self.test_patchSize}")

    def test_paddingSize_change(self):
        self.test_paddingSize = self.ui.test_sb_paddingSize.value()
        self.test_show_info(f"Testing -  padding size: {self.test_paddingSize}")

    def test_segThresh_change(self):
        self.test_segThresh = float(self.ui.test_edit_segThresh.text())
        self.test_show_info(f"Testing - segmentation threshold: {self.test_segThresh}")

    def test_format_change(self):
        self.test_coord_format = self.ui.test_cbox_format.currentText()
        self.test_show_info(f"Output coord format: {self.test_coord_format}")

    def test_select_CoordPath(self):
        self.test_coordPath = QFileDialog.getExistingDirectory(self, 'Select the coordination path')
        if self.test_coordPath != "":
            self.ui.test_edit_coord_path.setText(self.test_coordPath)
            self.test_show_info(f"Open coord path: {self.test_coordPath}")

    def test_clsId_change(self):
        self.test_clsId = self.ui.test_edit_clsId.text()
        self.test_show_info(f"Coord Format Conversion - cls id: {self.test_clsId}")

    def convert_ok(self):
        if self.ui.test_edit_coord_path.text() == "" \
                or self.ui.test_cbox_format.currentIndex == -1 \
                or self.ui.test_edit_clsId.text() == "":
            QMessageBox.critical(self, 'Error', 'Incomplete information')
            return 0
        else:
            self.test_show_info('*' * 100)
            self.test_show_info('Final coord conversion parameters')
            self.test_show_info('*' * 100)
            self.test_show_info(f"Coords path: {self.test_coordPath}")
            self.test_show_info(f"Output format: {self.test_coord_format}")
            self.test_show_info(f"Output cls id: {self.test_clsId}")
            self.test_show_info('*' * 100)

            self.test_clsId = int(self.test_clsId)
            coords_list = [i for i in os.listdir(self.test_coordPath) if i.endswith('.coords')]
            for coord_name in coords_list:
                try:
                    data = np.loadtxt(os.path.join(self.test_coordPath, coord_name),
                                  delimiter='\t').astype(np.float)
                except:
                    data = np.loadtxt(os.path.join(self.test_coordPath, coord_name),
                                      delimiter='\t').astype(np.float32)
                cls_ids = np.unique(data[:, 0]).astype(int)
                if self.test_clsId not in cls_ids:
                    print(self.test_clsId, cls_ids)
                    QMessageBox.critical(self, 'Error', 'Unknown class id.')
                    return 0
                else:
                    data = data[data[:, 0].astype(int) == self.test_clsId][:, 1:]
                    if self.test_coord_format == '.star':
                        save_path = f"{self.test_coordPath}_cls{self.test_clsId}_star/{coord_name.replace('.coords', '.star')}"
                        os.makedirs(f"{self.test_coordPath}_cls{self.test_clsId}_star", exist_ok=True)
                        coords2star(data, save_path)

                    elif self.test_coord_format == '.box':
                        save_path = f"{self.test_coordPath}_cls{self.test_clsId}_box/{coord_name.replace('.coords', '.box')}"
                        os.makedirs(f"{self.test_coordPath}_cls{self.test_clsId}_box", exist_ok=True)
                        coords2box(data, save_path)
                    elif self.test_coord_format == '.coords':
                        save_path = f"{self.test_coordPath}_cls{self.test_clsId}_coords/{coord_name}"
                        os.makedirs(f"{self.test_coordPath}_cls{self.test_clsId}_coords", exist_ok=True)
                        coords2coords(data, save_path)
            self.test_show_info('*' * 100)
            self.test_show_info(
                f"Coordinates of cls_id={self.test_clsId} have been converted into '{self.test_coord_format}'.")
            self.test_show_info(f"Details can be found in '{'/'.join(save_path.split('/')[:-1])}'")
            self.test_show_info('*' * 100)

    @pyqtSlot()
    def test_use(self):
        print('hahaha')

    def changeShowMask(self):
        if self.ui.cbox_mask_show.isChecked():
            self.ui.cbox_mask_show.setChecked(False)
        else:
            self.ui.cbox_mask_show.setChecked(True)

    def isShowMask(self):
        if self.ui.cbox_mask_show.isChecked():
            self.flag_show_mask = True
            self.ui.hSlider_mask_alpha.setEnabled(True)
            self.mask_alpha = self.old_alpha
            self.ui.hSlider_mask_alpha.setValue(self.mask_alpha)
            self.MC_updata()
        else:
            self.flag_show_mask = False
            self.ui.hSlider_mask_alpha.setEnabled(False)
            self.old_alpha = self.mask_alpha
            self.mask_alpha = 0
            self.MC_updata()

    def isHistEqu(self):
        if self.ui.cbox_hist_equ.isChecked():
            self.tomo_old = self.tomo_data
            if self.ui.cbox_gaus_filter.isChecked():
                self.tomo_gau = stretch(self.tomo_gau)
                self.tomo_data = hist_equ(self.tomo_gau)
                self.tomo_data = stretch(self.tomo_data)
            else:
                self.tomo_data = stretch(self.tomo_data)
                self.tomo_data = hist_equ(self.tomo_orig)
                self.tomo_data = stretch(self.tomo_data)

            self.MC_updata()
        else:
            if self.ui.cbox_gaus_filter.isChecked():
                self.tomo_data = self.tomo_gau
            else:
                self.tomo_data = self.tomo_orig
            self.MC_updata()

    def isGauFilter(self):
        if self.ui.cbox_gaus_filter.isChecked():
            self.ui.edit_gaussian_kernel.setEnabled(True)
            self.ui.edit_gaussian_sigma.setEnabled(True)
            self.ui.label_29.setEnabled(True)
            self.ui.label_30.setEnabled(True)
            self.gau_kernel = int(self.ui.edit_gaussian_kernel.text())
            self.gau_sigma = float(self.ui.edit_gaussian_sigma.text())
        else:
            self.ui.edit_gaussian_kernel.setEnabled(False)
            self.ui.edit_gaussian_sigma.setEnabled(False)
            self.ui.label_29.setEnabled(False)
            self.ui.label_30.setEnabled(False)

    def paramAdjust_ok(self):
        if self.ui.cbox_gaus_filter.isChecked():
            self.tomo_gau = filters.gaussian(self.tomo_orig, self.gau_sigma)
            self.tomo_data = self.tomo_gau
            self.tomo_data = stretch(self.tomo_data)
            self.tomo_gau = stretch(self.tomo_gau)
            self.isHistEqu()
            self.MC_updata()
        else:
            self.tomo_data = self.tomo_orig
            self.tomo_gau = self.tomo_orig
            self.MC_updata()

    def isPlotCircle(self):
        if self.ui.cbox_circle.isChecked():
            self.flag_show_circle = True
            self.ui.hSlider_circle_diameter.setEnabled(True)
            self.circle_diameter = self.old_diameter
            self.ui.hSlider_circle_diameter.setValue(self.circle_diameter)
            self.MC_updata()
        else:
            self.flag_show_circle = False
            self.ui.hSlider_circle_diameter.setEnabled(False)
            self.old_diameter = self.circle_diameter
            self.circle_diameter = 0
            self.MC_updata()

    def setColor(self):
        Qcolor = QColorDialog.getColor()
        self.color = (Qcolor.red(), Qcolor.green(), Qcolor.blue())
        if self.ui.mpick_ckb_enable:
            colors[self.mpick_clsId - 1] = self.color
        self.MC_updata()

    def DsetNameChanged(self):
        self.dset_name = self.ui.edit_dset_name.text()
        self.c2l_show_info(f"Dataset Name: {self.dset_name}")

    def RadButtonClicked(self):
        self.input_norm = self.ui.butG_pre_norm.checkedButton().text()
        self.c2l_show_info(f"Normalization type: {self.input_norm}")

    def openBasePath(self):
        self.c2l_basePath = QFileDialog.getExistingDirectory(self, 'Select the coordination path')
        if self.c2l_basePath != "":
            self.ui.edit_base_path.setText(self.c2l_basePath)
            self.c2l_show_info(f"Base dir: {self.c2l_basePath}")

    def openCoordPath(self):
        if self.c2l_basePath != "":
            self.c2l_coordPath = QFileDialog.getExistingDirectory(self,
                                                                  'Select the coordination path',
                                                                  self.c2l_basePath)
        else:
            self.c2l_coordPath = QFileDialog.getExistingDirectory(self,
                                                                  'Select the coordination path')

        if self.c2l_coordPath != "":
            self.ui.edit_coord_path.setText(self.c2l_coordPath)
            self.c2l_show_info(f"Coords Path: {self.c2l_coordPath}")

    def openTomoFile(self):
        if self.c2l_basePath != "":
            self.c2l_tomoFile = QFileDialog.getExistingDirectory(self,
                                                                 "Select the tomogram path",
                                                                 self.c2l_basePath)
        else:
            self.c2l_tomoFile = QFileDialog.getExistingDirectory(self,
                                                                 "Select the tomogram path")

        if self.c2l_tomoFile != "":
            self.ui.edit_tomo_file.setText(self.c2l_tomoFile)
            self.c2l_show_info(f"Tomo path: {self.c2l_tomoFile}")
            # self.input_format = self.c2l_tomoFile.split('.')[-1]

    def coord_format_change(self):
        if self.ui.cbox_coord_format.currentIndex != 0:
            self.coord_format = self.ui.cbox_coord_format.currentText()
            self.c2l_show_info(f"Coord format: {self.coord_format}")

    def lable_format_change(self):
        if self.ui.cbox_tomo_format.currentIndex != 0:
            self.tomo_format = self.ui.cbox_tomo_format.currentText()
            self.c2l_show_info(f"Label format: {self.tomo_format}")

    def lable_type_change(self):
        if self.ui.cbox_label_type.currentIndex != 0:
            self.label_type = self.ui.cbox_label_type.currentText()
            self.c2l_show_info(f"Label type: {self.label_type}")

    def label_diameter_change(self):
        self.label_diameter = self.ui.sb_label_diameter.value()
        self.c2l_show_info(f"Label diameter: {self.label_diameter}")

    def ocp_type_change(self):
        if self.ui.cbox_ocp_type.currentIndex != 0:
            self.ocp_type = self.ui.cbox_ocp_type.currentText()
            self.c2l_show_info(f"Ocp type: {self.ocp_type}")

    def ocp_diameter_change(self):
        self.ocp_diameter = self.ui.edit_ocp_diameter.text()
        self.c2l_show_info(f"Ocp diameter: {self.ocp_diameter}")

    def cls_num_change(self):
        self.cls_num = self.ui.sb_cls_num.value()
        self.c2l_show_info(f"Number of classes: {self.cls_num}")

    def c2l_show_info(self, info):
        self.ui.txtB_c2l.insertPlainText(f"{info}\n")
        self.ui.txtB_c2l.ensureCursorVisible()

    def loadConfigs(self):
        self.c2l_config_file, _ = QFileDialog.getOpenFileName(self,
                                                              'Select the coordination path')

        if self.c2l_config_file != "":
            self.ui.edit_load_configs.setText(self.c2l_config_file)
            self.c2l_show_info(f"Load configs: {self.c2l_config_file}")

        # config_name = self.c2l_config_file.split('/')[-1][:-3]

        # delete the package in the sys.modules
        # if f"configs.{config_name}" in list(sys.modules.keys()):
        #     del sys.modules[f"configs.{config_name}"]
        # config = importlib.import_module(f"configs.{config_name}")
        # self.preconfig = config.pre_config

        with open(self.c2l_config_file, 'r') as f:
            self.preconfig = json.loads(''.join(f.readlines()).lstrip('pre_config='))

        self.dset_name = self.preconfig["dset_name"]
        self.c2l_basePath = self.preconfig["base_path"],
        self.c2l_coordPath = self.preconfig["coord_path"],
        self.coord_format = self.preconfig["coord_format"],
        self.c2l_tomoFile = self.preconfig["tomo_path"],
        self.tomo_format = self.preconfig["tomo_format"],
        self.cls_num = self.preconfig["num_cls"],
        self.label_type = self.preconfig["label_type"],
        self.label_diameter = self.preconfig["label_diameter"],
        self.ocp_type = self.preconfig["ocp_type"],
        self.ocp_diameter = self.preconfig["ocp_diameter"],
        self.input_norm = self.preconfig["norm_type"]

        self.c2l_basePath = self.c2l_basePath[0] if isinstance(self.c2l_basePath[0], str) else self.c2l_basePath[0][0]
        self.c2l_coordPath = self.c2l_coordPath[0] if isinstance(self.c2l_coordPath[0], str) else self.c2l_coordPath[0][
            0]
        self.coord_format = self.coord_format[0] if isinstance(self.coord_format[0], str) else self.coord_format[0][0]
        self.c2l_tomoFile = self.c2l_tomoFile[0] if isinstance(self.c2l_tomoFile[0], str) else self.c2l_tomoFile[0][0]
        self.tomo_format = self.tomo_format[0] if isinstance(self.tomo_format[0], str) else self.tomo_format[0][0]
        self.label_type = self.label_type[0] if isinstance(self.label_type[0], str) else self.label_type[0][0]
        self.ocp_type = self.ocp_type[0] if isinstance(self.ocp_type[0], str) else self.ocp_type[0][0]
        self.cls_num = self.cls_num[0] if isinstance(self.cls_num[0], int) else self.cls_num[0][0]
        self.label_diameter = self.label_diameter[0] if isinstance(self.label_diameter[0], int) else \
            self.label_diameter[0][0]
        self.ocp_diameter = self.ocp_diameter[0] if isinstance(self.ocp_diameter[0], str) else self.ocp_diameter[0][0]

        self.ui.edit_dset_name.setText(self.dset_name)
        self.ui.edit_base_path.setText(self.c2l_basePath)
        self.ui.edit_coord_path.setText(self.c2l_coordPath)
        self.ui.cbox_coord_format.setCurrentText(self.coord_format)
        self.ui.edit_tomo_file.setText(self.c2l_tomoFile)
        self.ui.cbox_tomo_format.setCurrentText(self.tomo_format)
        self.ui.cbox_label_type.setCurrentText(self.label_type)
        self.ui.cbox_ocp_type.setCurrentText(self.ocp_type)
        self.ui.sb_cls_num.setValue(self.cls_num)
        self.ui.sb_label_diameter.setValue(self.label_diameter)
        self.ui.edit_ocp_diameter.setText(self.ocp_diameter)

        if self.input_norm == "standardization":
            self.ui.radB_norm.setChecked(False)
            self.ui.radB_std.setChecked(True)
        elif self.input_norm == "normalization":
            self.ui.radB_std.setChecked(False)
            self.ui.radB_norm.setChecked(True)

        self.c2l_show_info('*' * 100)
        self.c2l_show_info(f"Coords Path: {self.c2l_coordPath}")
        self.c2l_show_info(f"Coord format: {self.coord_format}")
        self.c2l_show_info(f"Tomo file: {self.c2l_tomoFile}")
        self.c2l_show_info(f"Number of classes: {self.cls_num}")
        self.c2l_show_info(f"Label format: {self.tomo_format}")
        self.c2l_show_info(f"Label type: {self.label_type}")
        self.c2l_show_info(f"Label diameter: {self.label_diameter}")
        self.c2l_show_info(f"Normalization type: {self.input_norm}")
        self.c2l_show_info(f"Ocp type: {self.ocp_type}")
        self.c2l_show_info(f"Ocp diameter: {self.ocp_diameter}")

    def saveConfigs(self):
        pre_config = dict(
            dset_name=self.dset_name if isinstance(self.dset_name, str) else self.dset_name[0],
            base_path=self.c2l_basePath if isinstance(self.c2l_basePath, str) else self.c2l_basePath[0],
            coord_path=self.c2l_coordPath if isinstance(self.c2l_coordPath, str) else self.c2l_coordPath[0],
            coord_format=self.coord_format if isinstance(self.coord_format, str) else self.coord_format[0],
            tomo_path=self.c2l_tomoFile if isinstance(self.c2l_tomoFile, str) else self.c2l_tomoFile[0],
            tomo_format=self.tomo_format if isinstance(self.tomo_format, str) else self.tomo_format[0],
            num_cls=self.cls_num if isinstance(self.cls_num, int) else self.cls_num[0],
            label_type=self.label_type if isinstance(self.label_type, str) else self.label_type[0],
            label_diameter=self.label_diameter if isinstance(self.label_diameter, int) else self.label_diameter[0],
            ocp_type=self.ocp_type if isinstance(self.ocp_type, str) else self.ocp_type[0],
            ocp_diameter=self.ocp_diameter if isinstance(self.ocp_diameter, str) else self.ocp_diameter[0],
            norm_type=self.input_norm if isinstance(self.input_norm, str) else self.input_norm[0]
        )
        config_save_path = os.path.join(
            self.c2l_basePath if isinstance(self.c2l_basePath, str) else self.c2l_basePath[0], 'configs')
        os.makedirs(config_save_path, exist_ok=True)
        if self.cls_num == len(self.ocp_diameter.split(',')):
            with open(f"{config_save_path}/{self.dset_name}.py", 'w') as f:
                f.write("pre_config=")
                json.dump(pre_config, f, separators=(',\n'+' '*len('pre_config={'), ': '))
            self.c2l_show_info(f"save configs to {config_save_path}, Finished!")
        else:
            self.c2l_show_info(f"The number of classes is not consistent with the 'Ocp diameter'")

    def c2l_ok(self):
        if self.ui.edit_dset_name.text() == "" \
                or self.ui.edit_base_path.text() == "" \
                or self.ui.edit_coord_path.text() == "" \
                or self.ui.edit_tomo_file.text() == "" \
                or self.ui.cbox_coord_format.currentIndex() == -1 \
                or self.ui.cbox_tomo_format.currentIndex() == -1:

            QMessageBox.critical(self, 'Error', 'Incomplete information')
        else:
            self.ui.gBox_output.setEnabled(False)
            self.ui.gBox_input.setEnabled(False)
            self.c2l_show_info('*' * 100)
            self.c2l_show_info('Final configuration parameters')
            self.c2l_show_info('*' * 100)
            self.c2l_show_info(f"Coords Path: {self.c2l_coordPath}")
            self.c2l_show_info(f"Coord format: {self.coord_format}")
            self.c2l_show_info(f"Tomo file: {self.c2l_tomoFile}")
            self.c2l_show_info(f"Number of classes: {self.cls_num}")
            self.c2l_show_info(f"Label format: {self.tomo_format}")
            self.c2l_show_info(f"Label type: {self.label_type}")
            self.c2l_show_info(f"Label diameter: {self.label_diameter}")
            self.c2l_show_info(f"Normalization type: {self.input_norm}")
            self.c2l_show_info(f"Ocp type: {self.ocp_type}")
            self.c2l_show_info(f"Ocp diameter: {self.ocp_diameter}")
            self.c2l_show_info('*' * 100)

            # Initial coords
            coords_gen_emit = EmittingStr()
            coords_gen_emit.textWritten.connect(self.c2l_show_info)
            thread = ThreadShowInfo(func=coords_gen_show,
                                     args=(self.c2l_coordPath,
                                           self.coord_format,
                                           self.c2l_basePath,
                                           coords_gen_emit))
            thread.start()
            while not thread.isFinished():
                pass

            # Normalization of input data
            norm_emit = EmittingStr()
            norm_emit.textWritten.connect(self.c2l_show_info)
            thread1 = ThreadShowInfo(func=norm_show,
                                      args=(self.c2l_tomoFile,
                                            self.tomo_format,
                                            self.c2l_basePath,
                                            self.input_norm,
                                            norm_emit))
            thread1.start()
            while not thread1.isFinished():
                pass

            # Generate labes according to the coords
            emit = EmittingStr()
            emit.textWritten.connect(self.c2l_show_info)
            thread2 = ThreadShowInfo(func=label_gen_show,
                                      args=(self.c2l_basePath, self.c2l_coordPath, self.coord_format, self.c2l_tomoFile,
                                            self.tomo_format, self.cls_num, self.label_type, str(self.label_diameter),
                                            emit))
            thread2.start()
            while not thread2.isFinished():
                pass

            # Generate ocps according to the coords
            emit = EmittingStr()
            emit.textWritten.connect(self.c2l_show_info)
            thread3 = ThreadShowInfo(func=label_gen_show,
                                      args=(self.c2l_basePath, self.c2l_coordPath, self.coord_format, self.c2l_tomoFile,
                                            self.tomo_format, self.cls_num, "data_ocp", self.ocp_diameter, emit))
            thread3.start()
            while not thread3.isFinished():
                pass

            self.ui.gBox_output.setEnabled(True)
            self.ui.gBox_input.setEnabled(True)
            self.c2l_show_info('Preprocess Finished.')

    def setSliderXYZ(self, z, y, x, tomo_shape):
        z_max, y_max, x_max = tomo_shape
        self.ui.hSlider_z.setMinimum(0)
        self.ui.hSlider_y.setMinimum(0)
        self.ui.hSlider_x.setMinimum(0)
        self.ui.hSlider_z.setMaximum(z_max - 1)
        self.ui.hSlider_y.setMaximum(y_max - 1)
        self.ui.hSlider_x.setMaximum(x_max - 1)
        self.ui.hSlider_z.setValue(z)
        self.ui.hSlider_y.setValue(y)
        self.ui.hSlider_x.setValue(x)
        self.ui.edit_z.setText(f"{z:.0f}")
        self.ui.edit_y.setText(f"{y:.0f}")
        self.ui.edit_x.setText(f"{x:.0f}")

    def sectionView(self, tomo_path):
        if tomo_path == "":
            self.tomo_data = np.random.randn(200, 400, 400)
        else:
            with mrcfile.open(tomo_path, permissive=True) as tomo:
                try:
                    self.tomo_data = np.array(tomo.data).astype(np.float)
                except:
                    self.tomo_data = np.array(tomo.data).astype(np.float32)
                self.tomo_data = stretch(self.tomo_data)
        z_max, y_max, x_max = self.tomo_data.shape

        self.data_xy = np.transpose(self.tomo_data[int(z_max // 2), :, :])
        self.data_zy = self.tomo_data[:, :, x_max // 2]
        self.data_xz = np.transpose(self.tomo_data[:, y_max // 2, :])
        # print(z_max, y_max, x_max)

        self.win = pg.GraphicsLayoutWidget()
        self.win.show()  ## show widget alone in its own window
        self.win.setWindowTitle('pyqtgraph example: ImageItem')
        self.win.resize(x_max + z_max, y_max + z_max)
        # win.ci.setBorder((5, 5, 10))

        self.win.nextRow()
        self.sub1 = self.win.addLayout(border=(100, 10, 10))
        self.sub1.setContentsMargins(0, 0, 0, 0)
        # self.p_xy = self.sub1.addViewBox(row=0, col=0, rowspan=y_max, colspan=x_max)
        self.p_xy = self.sub1.addViewBox()
        self.p_xy.disableAutoRange()
        self.p_xy.setAspectLocked(True)  ## lock the aspect ratio so pixels are always square
        self.p_xy.setXRange(0, x_max)
        self.p_xy.setYRange(0, y_max)
        # self.p_xy.setLimits(xMin=-50, xMax=x_max+50, yMin=-50, yMax=y_max+50)
        self.img_xy = pg.ImageItem(border='b')
        self.p_xy.addItem(self.img_xy)
        self.img_xy.setImage(self.data_xy)

        # self.p_zy = self.sub1.addViewBox(row=0, col=x_max, rowspan=1, colspan=1)
        self.p_zy = self.sub1.addViewBox()
        self.p_zy.disableAutoRange()
        self.p_zy.setAspectLocked(True)  ## lock the aspect ratio so pixels are always square
        self.p_zy.setXRange(0, z_max)
        self.p_zy.setYRange(0, y_max)
        self.img_zy = pg.ImageItem(border='b')
        self.p_zy.addItem(self.img_zy)
        self.img_zy.setImage(self.data_zy)
        self.p_zy.linkView(self.p_xy.YAxis, self.p_xy)

        self.sub1.nextRow()
        # self.p_xz = self.sub1.addViewBox(row=y_max, col=0, rowspan=z_max, colspan=z_max)
        self.p_xz = self.sub1.addViewBox()
        self.p_xz.disableAutoRange()
        self.p_xz.setAspectLocked(True)  ## lock the aspect ratio so pixels are always square
        self.p_xz.setXRange(0, x_max)
        self.p_xz.setYRange(0, z_max)
        self.img_xz = pg.ImageItem(border='b')
        self.p_xz.addItem(self.img_xz)
        self.img_xz.setImage(self.data_xz)
        self.p_xz.linkView(self.p_xy.XAxis, self.p_xy)
        # self.p_xz.translateBy(y=-80)
        # xz_h, xz_w = self.p_xz.height(), self.p_xz.width()
        # print(xz_h, xz_w)

        # self.coords_ = self.sub1.addViewBox(row=y_max, col=x_max, rowspan=z_max, colspan=z_max)
        # self.sub1.nextCol()
        self.text = pg.LabelItem(justify='center')
        self.sub1.addItem(self.text)
        self.text.setText(f"({z_max // 2}, {y_max // 2}, {x_max // 2})")
        self.setSliderXYZ(z_max // 2, y_max // 2, x_max // 2, self.tomo_data.shape)

        # self.label_adjust = self.sub1.addViewBox(enableMouse=False)
        # self.label_adjust.disableAutoRange()
        # self.label_adjust.setAspectLocked(True)
        # self.label_adjust.setXRange(0, z_max)
        # self.label_adjust.setYRange(0, z_max)
        # self.text = pg.TextItem(rotateAxis=(1,0)) #anchor=(z_max//2, 0)
        # self.label_adjust.addItem(self.text)
        # self.text.setText(text=f"({z_max // 2}, {y_max // 2}, {x_max // 2})", color=(0,255,0))
        # self.label = pg.LabelItem(justify='center')
        # self.label_adjust.addItem(self.label)
        # self.label.setText(f"({z_max // 2}, {y_max // 2}, {x_max // 2})")

        # cross hair
        self.vLine_xy = pg.InfiniteLine(angle=90, movable=False)
        self.hLine_xy = pg.InfiniteLine(angle=0, movable=False)
        self.p_xy.addItem(self.vLine_xy, ignoreBounds=True)
        self.p_xy.addItem(self.hLine_xy, ignoreBounds=True)

        self.vLine_zy = pg.InfiniteLine(angle=90, movable=False)
        self.hLine_zy = pg.InfiniteLine(angle=0, movable=False)
        self.p_zy.addItem(self.vLine_zy, ignoreBounds=True)
        self.p_zy.addItem(self.hLine_zy, ignoreBounds=True)

        self.vLine_xz = pg.InfiniteLine(angle=90, movable=False)
        self.hLine_xz = pg.InfiniteLine(angle=0, movable=False)
        self.p_xz.addItem(self.vLine_xz, ignoreBounds=True)
        self.p_xz.addItem(self.hLine_xz, ignoreBounds=True)

        self.x, self.y, self.z = x_max // 2, y_max // 2, z_max // 2

    def mouseMoved(self, evt):
        pos = evt[0]
        print(pos)
        if self.p_xy.sceneBoundingRect().contains(pos):
            mousePoint = self.p_xy.mapSceneToView(pos)
            self.x, self.y = mousePoint.x(), mousePoint.y()
        elif self.p_zy.sceneBoundingRect().contains(pos):
            mousePoint = self.p_zy.mapSceneToView(pos)
            self.z, self.y = mousePoint.x(), mousePoint.y()
        elif self.p_xz.sceneBoundingRect().contains(pos):
            mousePoint = self.p_xz.mapSceneToView(pos)
            self.x, self.z = mousePoint.x(), mousePoint.y()

        self.text.setText(f"({self.x:.0f}, {self.y:.0f}, {self.z:.0f}), "
                          f"{self.tomo_data[int(self.z), int(self.y), int(self.x)]:.2f}")
        self.setSliderXYZ(self.z, self.y, self.x, self.tomo_data.shape)
        self.vLine_xy.setPos(self.x)
        self.hLine_xy.setPos(self.y)
        self.vLine_zy.setPos(self.z)
        self.hLine_zy.setPos(self.y)
        self.vLine_xz.setPos(self.x)
        self.hLine_xz.setPos(self.z)

    def mouseClicked(self, evt):
        print(evt)
        pos = evt.scenePos()

        if self.p_xy.sceneBoundingRect().contains(pos):
            mousePoint = self.p_xy.mapSceneToView(pos)
            self.x, self.y = mousePoint.x(), mousePoint.y()
        elif self.p_zy.sceneBoundingRect().contains(pos):
            mousePoint = self.p_zy.mapSceneToView(pos)
            self.z, self.y = mousePoint.x(), mousePoint.y()
        elif self.p_xz.sceneBoundingRect().contains(pos):
            mousePoint = self.p_xz.mapSceneToView(pos)
            self.x, self.z = mousePoint.x(), mousePoint.y()

        if self.ui.mpick_ckb_enable.isChecked() and evt.double():
            xyz = [self.mpick_clsId, self.x, self.y, self.z]
            if self.mpick_coords_np.shape[0] > 0:
                dist = np.linalg.norm(
                    self.mpick_coords_np[:, -3:] - np.array(xyz).reshape(1, -1)[:, -3:].astype(float), ord=2, axis=1)
                dist_idx = (dist < self.mpick_circle_diameter // 2)
                if len(np.nonzero(dist_idx)[0]) >= 1:
                    temp_idx = np.nonzero(dist_idx)[0].tolist()
                    self.mpick_coords_np = np.delete(self.mpick_coords_np, temp_idx, axis=0)
                    self.mpick_coords = self.mpick_coords_np.tolist()
                else:
                    self.mpick_coords.append(xyz)
                    self.mpick_coords_np = np.array(self.mpick_coords)
            else:
                self.mpick_coords.append(xyz)
                self.mpick_coords_np = np.array(self.mpick_coords)
            self.mouse_double = True

            self.res_show_info('coords')
            self.res_show_info('*' * 100)
            for idx, xyz in enumerate(self.mpick_coords_np):
                self.res_show_info(f"{idx}:{xyz}")
            self.res_show_info('*' * 100)
            self.res_show_info(colors[self.mpick_clsId - 1])
            self.mouse_double = False
        else:
            self.mouse_double = False
        self.MC_updata()

    def MC_updata(self):
        self.vLine_xy.setPos(self.x)
        self.hLine_xy.setPos(self.y)
        self.vLine_zy.setPos(self.z)
        self.hLine_zy.setPos(self.y)
        self.vLine_xz.setPos(self.x)
        self.hLine_xz.setPos(self.z)

        self.text.setText(f"({self.x:.0f}, {self.y:.0f}, {self.z:.0f}), "
                          f"{self.tomo_data[int(self.z), int(self.y), int(self.x)]:.2f}")
        self.setSliderXYZ(self.z, self.y, self.x, self.tomo_data.shape)
        self.data_xy = np.transpose(self.tomo_data[int(self.z), :, :])
        self.data_zy = self.tomo_data[:, :, int(self.x)]
        self.data_xz = np.transpose(self.tomo_data[:, int(self.y), :])

        # print(np.min(self.data_xy), np.max(self.data_xy)
        if self.ui.mpick_ckb_enable.isChecked() \
                and self.mpick_coords_np.shape[0] > 0:
            color = [self.color] * self.mpick_coords_np.shape[0] if self.mpick_coords_np.shape[1] == 3 else [
                colors[int(i) - 1] for i in self.mpick_coords_np[:, 0]]
            self.img_xy.setImage(
                annotate_particle(self.data_xy, self.mpick_coords_np[:, -3:], self.mpick_circle_diameter, self.z, 0,
                               self.mpick_circle_width, color))
            self.img_zy.setImage(
                annotate_particle(self.data_zy, self.mpick_coords_np[:, -3:], self.mpick_circle_diameter, self.x, 1,
                               self.mpick_circle_width, color))
            self.img_xz.setImage(
                annotate_particle(self.data_xz, self.mpick_coords_np[:, -3:], self.mpick_circle_diameter, self.y, 2,
                               self.mpick_circle_width, color))
        elif self.res_label != []:
            if self.res_label_type == 'Mask':
                if self.flag_show_mask:
                    self.label_xy = np.transpose(self.res_label[int(self.z), :, :])
                    self.label_zy = self.res_label[:, :, int(self.x)]
                    self.label_xz = np.transpose(self.res_label[:, int(self.y), :])

                    self.img_xy.setImage(
                        add_transparency(self.data_xy, self.label_xy, float(self.mask_alpha) / 100., self.color, 0.5))
                    self.img_zy.setImage(
                        add_transparency(self.data_zy, self.label_zy, float(self.mask_alpha) / 100., self.color, 0.5))
                    self.img_xz.setImage(
                        add_transparency(self.data_xz, self.label_xz, float(self.mask_alpha) / 100., self.color, 0.5))
                else:
                    self.img_xy.setImage(self.data_xy)
                    self.img_zy.setImage(self.data_zy)
                    self.img_xz.setImage(self.data_xz)

            elif self.res_label_type == 'Coords':
                if self.flag_show_circle:
                    color = [self.color] * self.res_label.shape[0] if self.res_label.shape[1] == 3 else [
                        colors[int(i) - 1] for i in self.res_label[:, 0]]
                    self.img_xy.setImage(
                        annotate_particle(self.data_xy, self.res_label[:, -3:], self.circle_diameter, self.z, 0,
                                       self.circle_width, color))
                    self.img_zy.setImage(
                        annotate_particle(self.data_zy, self.res_label[:, -3:], self.circle_diameter, self.x, 1,
                                       self.circle_width, color))
                    self.img_xz.setImage(
                        annotate_particle(self.data_xz, self.res_label[:, -3:], self.circle_diameter, self.y, 2,
                                       self.circle_width, color))
                else:
                    self.img_xy.setImage(self.data_xy)
                    self.img_zy.setImage(self.data_zy)
                    self.img_xz.setImage(self.data_xz)
        else:
            self.img_xy.setImage(self.data_xy)
            self.img_zy.setImage(self.data_zy)
            self.img_xz.setImage(self.data_xz)

    def showTomo(self):
        self.show_tomoFile, _ = QFileDialog.getOpenFileName(self,
                                                            "Select the tomogram path")

        if self.show_tomoFile != "":
            self.ui.edit_show_tomo.setPlainText(self.show_tomoFile)
            self.c2l_show_info(f"Showing tomo file: {self.show_tomoFile}")
            with mrcfile.open(self.show_tomoFile, permissive=True) as tomo:
                try:
                    self.tomo_data = np.array(tomo.data).astype(np.float)
                except:
                    self.tomo_data = np.array(tomo.data).astype(np.float32)

                self.tomo_data = stretch(self.tomo_data)
                self.tomo_shape = self.tomo_data.shape
                self.tomo_orig = self.tomo_data
            self.MC_updata()

    def showLabel(self):
        self.show_labelFile, _ = QFileDialog.getOpenFileName(self,
                                                             "Select the tomogram path")

        if self.show_labelFile != "":
            self.ui.edit_show_label.setPlainText(self.show_labelFile)
            self.c2l_show_info(f"Showing label file: {self.show_labelFile}")

    def showTomoLabel(self):
        c2l_1 = Coord_to_Label_v1(self.show_tomoFile,
                                  self.show_labelFile,
                                  1,
                                  7,
                                  self.res_label_type)
        self.res_label = c2l_1.gen_labels()

    def circle_width_change(self):
        self.circle_width = self.ui.sb_circle_width.value()
        self.MC_updata()

    def RBut_LabelClicked(self):
        self.res_label_type = self.ui.butG_res_label_type.checkedButton().text()
        self.showTomoLabel()
        self.MC_updata()
        if self.res_label_type == 'Mask':
            self.ui.hSlider_circle_diameter.setEnabled(False)
            self.ui.cbox_circle.setEnabled(False)
            self.ui.sb_circle_width.setEnabled(False)
        else:
            self.ui.hSlider_circle_diameter.setEnabled(True)
            self.ui.cbox_circle.setEnabled(True)
            self.ui.sb_circle_width.setEnabled(True)

    def diameterChange(self):
        self.circle_diameter = self.ui.hSlider_circle_diameter.value()
        self.ui.edit_circle_diameter.setText(str(self.circle_diameter))
        self.MC_updata()

    def xyzChange(self):
        self.x = self.ui.hSlider_x.value()
        self.y = self.ui.hSlider_y.value()
        self.z = self.ui.hSlider_z.value()
        self.ui.edit_x.setText(f"{self.x:.0f}")
        self.ui.edit_y.setText(f"{self.y:.0f}")
        self.ui.edit_z.setText(f"{self.z:.0f}")
        self.MC_updata()

    def changeZ_up(self):
        self.z = self.ui.hSlider_z.value()
        self.z += 1
        self.ui.edit_z.setText(f"{self.z:.0f}")
        self.ui.hSlider_z.setValue(self.z)

    def changeZ_down(self):
        self.z = self.ui.hSlider_z.value()
        self.z -= 1
        self.ui.edit_z.setText(f"{self.z:.0f}")
        self.ui.hSlider_z.setValue(self.z)

    def maskAlphaChange(self):
        self.mask_alpha = self.ui.hSlider_mask_alpha.value()
        self.ui.edit_mask_alpha.setText(f"{float(self.mask_alpha) / 100.:0.2f}")
        self.MC_updata()

    def saveVideo(self):
        zmin = int(self.ui.edit_SAV_zmin.text())
        zmax = int(self.ui.edit_SAV_zmax.text())
        ymin = int(self.ui.edit_SAV_ymin.text())
        ymax = int(self.ui.edit_SAV_ymax.text())
        xmin = int(self.ui.edit_SAV_xmin.text())
        xmax = int(self.ui.edit_SAV_xmax.text())
        fps = int(self.ui.edit_SAV_FPS.text())
        z_interval = int(self.ui.edit_SAV_zInterval.text())
        save_path, _ = QFileDialog.getSaveFileName(self, 'Select save path for video')

        if 'mp4' in save_path:
            fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        elif 'avi' in save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        videowriter = cv2.VideoWriter(save_path,
                                      fourcc,
                                      fps,
                                      (ymax - ymin, xmax - xmin))
        for z_idx in range(zmin, zmax + 1, z_interval):
            data_xy = np.transpose(self.tomo_data[int(z_idx), :, :])
            color = [self.color] * self.res_label.shape[0] if self.res_label.shape[1] == 3 else [
                colors[i - 1] for i in self.res_label[:, 0]]
            img = annotate_particle(data_xy, self.res_label[:, -3:], self.circle_diameter, z_idx, 0,
                                 self.circle_width, color)[xmin:xmax, ymin:ymax, :]
            print(img.shape)
            videowriter.write(img)
        videowriter.release()


app = QApplication([])
stats = Stats()
stats.show()
app.exec_()
