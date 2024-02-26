from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from train_yolo_v9.train_yolo_v9_process import TrainYoloV9Param

# PyQt GUI framework
from PyQt5.QtWidgets import *
from train_yolo_v9.ikutils import model_zoo

# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class TrainYoloV9Widget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = TrainYoloV9Param()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
    
        # Dataset folder
        self.browse_dataset_folder = pyqtutils.append_browse_file(self.grid_layout, label="Dataset folder",
                                                                  path=self.parameters.cfg["dataset_folder"],
                                                                  tooltip="Select folder",
                                                                  mode=QFileDialog.Directory)

        # Model name
        self.combo_model_name = pyqtutils.append_combo(self.grid_layout, "Model name")
        for model_name in model_zoo.keys():
            self.combo_model_name.addItem(model_name)

        self.combo_model_name.setCurrentText(self.parameters.cfg["model_name"])

        # Epochs
        self.spin_epochs = pyqtutils.append_spin(self.grid_layout, "Epochs", self.parameters.cfg["epochs"])

        # Batch size
        self.spin_batch = pyqtutils.append_spin(self.grid_layout, "Batch size", self.parameters.cfg["batch_size"])

        # Input size
        self.spin_train_imgsz = pyqtutils.append_spin(self.grid_layout, "Train image size",
                                                  self.parameters.cfg["train_imgsz"])
        self.spin_test_imgsz = pyqtutils.append_spin(self.grid_layout, "Test image size",
                                                  self.parameters.cfg["test_imgsz"])

        # Hyper-parameters
        custom_hyp = bool(self.parameters.cfg["config_file"])
        self.check_hyp = QCheckBox("Custom hyper-parameters")
        self.check_hyp.setChecked(custom_hyp)
        self.grid_layout.addWidget(self.check_hyp, self.grid_layout.rowCount(), 0, 1, 2)
        self.check_hyp.stateChanged.connect(self.on_custom_hyp_changed)

        self.label_hyp = QLabel("Hyper-parameters file")
        self.browse_hyp_file = pyqtutils.BrowseFileWidget(path=self.parameters.cfg["config_file"],
                                                          tooltip="Select file",
                                                          mode=QFileDialog.ExistingFile)

        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_hyp, row, 0)
        self.grid_layout.addWidget(self.browse_hyp_file, row, 1)

        self.label_hyp.setVisible(custom_hyp)
        self.browse_hyp_file.setVisible(custom_hyp)

        # Model weight file
        self.browse_model_weight_file = pyqtutils.append_browse_file(self.grid_layout, label="Model weight file",
                                                                     path=self.parameters.cfg["model_weight_file"],
                                                                     tooltip="Select file", mode=QFileDialog.ExistingFile)
        # Output folder
        self.browse_out_folder = pyqtutils.append_browse_file(self.grid_layout, label="Output folder",
                                                              path=self.parameters.cfg["output_folder"],
                                                              tooltip="Select folder",
                                                              mode=QFileDialog.Directory)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_custom_hyp_changed(self, int):
        self.label_hyp.setVisible(self.check_hyp.isChecked())
        self.browse_hyp_file.setVisible(self.check_hyp.isChecked())

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.cfg["dataset_folder"] = self.browse_dataset_folder.path
        self.parameters.cfg["model_name"] = self.combo_model_name.currentText()
        self.parameters.cfg["epochs"] = self.spin_epochs.value()
        self.parameters.cfg["batch_size"] = self.spin_batch.value()
        self.parameters.cfg["train_imgsz"] = self.spin_train_imgsz.value()
        self.parameters.cfg["test_imgsz"] = self.spin_test_imgsz.value()
        self.parameters.cfg["model_weight_file"] = self.browse_model_weight_file.path

        if self.check_hyp.isChecked():
            self.parameters.cfg["config_file"] = self.browse_hyp_file.path

        self.parameters.cfg["output_folder"] = self.browse_out_folder.path

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class TrainYoloV9WidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "train_yolo_v9"

    def create(self, param):
        # Create widget object
        return TrainYoloV9Widget(param, None)
