"""
Useful napari list widget to go through images in a chosen folder

"""

from pathlib import Path
from typing import List

import napari
import nd2
import numpy as np
import scipy.ndimage as ndi
import skimage as sk
import tifffile
import torch
from magicgui import magicgui
from napari.types import ImageData, LayerDataTuple
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from SBsparsify import sparsify_denoise2, torch_estimate_background_idwt

FILE_FORMATS = ["[!.]*.tif", "[!.]*.tiff", "[!.]*.nd2"]
ROOTFOLDER = "/home/starrluxton/Data/Xiangyi/ER_quantification/mKateTRAM-1_ER"


def imread(filename: Path) -> np.ndarray:
    ext = filename.suffix

    if ext == ".nd2":
        img = nd2.imread(filename)
        return img

    elif ext == ".tif" or ext == ".tiff":
        img = tifffile.imread(filename)
        return img


class InputFileList(QListWidget):
    # custom list widget that supports drag-n-drop & delete
    itemDeleted = Signal(int)

    def __init__(self):
        super().__init__()
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(self.InternalMove)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            current_index = self.currentRow()
            self.takeItem(current_index)
            self.itemDeleted.emit(self.count())

        if e.key() == Qt.Key_Up:
            current_index = self.currentRow()
            moved_index = max(current_index, 0)
            self.setCurrentRow(moved_index)

        if e.key() == Qt.Key_Down:
            n_items = self.count()
            current_index = self.currentRow()
            moved_index = min(current_index, n_items - 1)
            self.setCurrentRow(moved_index)

        super().keyPressEvent(e)


class FilelistWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.layout = QVBoxLayout()

        self.choose_folder_button = QPushButton("Choose a folder")
        self.current_folder_label = QLabel("")
        self.flist_widget = InputFileList()
        self.flist_groupbox = QGroupBox("Input files")
        self.flist_vlayout = QVBoxLayout()
        self.flist_vlayout.addWidget(self.choose_folder_button)
        self.flist_vlayout.addWidget(self.current_folder_label)
        self.flist_vlayout.addWidget(self.flist_widget)
        self.flist_groupbox.setLayout(self.flist_vlayout)
        self.layout.addWidget(self.flist_groupbox)
        self.scrub_area = QLineEdit("300000")

        scrub_body_btn = QPushButton("Scrub body")
        save_mask_btn = QPushButton("Save masks")

        self.layout.addWidget(save_mask_btn)
        self.layout.addWidget(self.scrub_area)
        self.layout.addWidget(scrub_body_btn)
        self.setLayout(self.layout)

        self.choose_folder_button.clicked.connect(self._open_file_dialog)
        save_mask_btn.clicked.connect(self._save_masks)
        scrub_body_btn.clicked.connect(self._scrub_body)

    def _scrub_body(self):
        if "body" in self.viewer.layers:
            area_thres = float(self.scrub_area.text())
            body_img = self.viewer.layers["body"].data
            body_img = sk.morphology.remove_small_objects(
                (body_img > 0), min_size=area_thres
            )
            self.viewer.layers["body"].data = body_img

    def _open_file_dialog(self):
        self.img_folder = Path(
            QFileDialog.getExistingDirectory(
                self, "Choose a folder", ROOTFOLDER
            )
        )
        if self.img_folder:
            self.flist_widget.clear()
        else:
            return

        self.imgpath = Path(self.img_folder)
        flist = []
        for ext in FILE_FORMATS:
            _flist = [file for file in self.imgpath.glob(ext)]
            flist.extend(_flist)

        nfiles = len(flist)

        # construct abbreviated path
        abbr_path = f"...{self.img_folder.parent.name}/{self.img_folder.name}"

        # update current folder label
        self.current_folder_label.setText(f"{abbr_path:s} ({nfiles:d})")

        for f in sorted(flist):
            self.flist_widget.addItem(f.name)

        self.flist_widget.currentItemChanged.connect(self._load_image)
        self.flist_widget.itemDeleted.connect(self._update_count)

    def _load_image(self, key):
        fname = key.text()
        if fname:
            fpath = self.img_folder / fname
            # get extension and handle different formats here
            img = imread(fpath)
            self.viewer.layers.clear()
            self.viewer.add_image(
                img, name=f"{fpath.stem}", colormap="viridis"
            )

    def _update_count(self, n):
        abbr_path = f"...{self.img_folder.parent.name}/{self.img_folder.name}"
        # update current folder label
        self.current_folder_label.setText(f"{abbr_path:s} ({n:d})")

    def _save_masks(self):
        body_exists = "body" in self.viewer.layers
        er_exists = "ER" in self.viewer.layers

        if body_exists and er_exists:
            out_root = self.img_folder.parent / "masks"
            out_path = out_root / self.img_folder.name
            out_path.mkdir(exist_ok=True, parents=True)
            img_body = self.viewer.layers["body"].data.astype(int)
            img_er = self.viewer.layers["ER"].data.astype(int)
            # only retain ER within the body
            img_er = img_er * (img_body > 0)
            img_fn_str = self.flist_widget.currentItem().text().split(".")[0]
            er_area = (img_er > 0).sum()
            body_area = (img_body > 0).sum()

            viewer.layers["ER"].data = (img_er > 0) * 2

            fractional_area = er_area / (er_area + body_area)
            print(f"{img_fn_str}:\tArea occupancy = {fractional_area:12.3f}")
            tifffile.imwrite(
                out_path / f"{img_fn_str}_body.tif",
                np.uint8(img_body),
                compression="packbits",
            )
            tifffile.imwrite(
                out_path / f"{img_fn_str}_ER.tif",
                np.uint8(img_er),
                compression="packbits",
            )


@magicgui(
    call_button="Estimate background",
    tol={
        "widget_type": "FloatSpinBox",
        "step": 0.001,
        "value": 1e-3,
        "min": 1e-7,
    },
)
def background(
    image: ImageData,
    k: int = 5,
    max_iter: int = 25,
    tol: float = 1e-3,
) -> ImageData:
    bg = torch_estimate_background_idwt(
        image, max_k=k, niter=max_iter, tol=tol
    )
    net = np.maximum(image - bg, 0)

    return net


@magicgui(call_button="Sparsify")
def sparse(
    image: ImageData,
    smoothness: float = 20.0,
    sparsity: float = 30.0,
    rho: float = 0.1,
    max_iter: int = 100,
    GPU: bool = True,
) -> ImageData:
    if GPU:
        gpu = torch.device("cuda")
        image = torch.from_numpy(image.astype(np.float32)).to(gpu)

    return sparsify_denoise2(
        image,
        smoothness,
        sparsity,
        rho=rho,
        max_iter=max_iter,
    )


@magicgui(call_button="make mask")
def segment_er(image: ImageData) -> List[LayerDataTuple]:
    disk = sk.morphology.disk(10)
    blurred_image = ndi.gaussian_filter(image, 5.0)
    body_val = sk.filters.threshold_triangle(blurred_image)
    er_val = sk.filters.threshold_li(image)

    er_img = 2 * (image > er_val)
    body = image > body_val
    body = ndi.binary_fill_holes(body)
    body = sk.morphology.remove_small_objects(body, min_size=400)
    body = sk.morphology.binary_dilation(body, disk)
    body = ndi.binary_fill_holes(body)
    body = sk.morphology.convex_hull_image(body)
    return [
        (body, {"name": "body"}, "labels"),
        (er_img, {"name": "ER"}, "labels"),
    ]


if __name__ == "__main__":
    viewer = napari.Viewer()
    custom_widget = FilelistWidget(viewer)
    filelist_dock = viewer.window.add_dock_widget(
        custom_widget, name="custom list"
    )
    viewer.window.add_dock_widget(background, name="bg. estimation")
    viewer.window.add_dock_widget(sparse, name="S & D")
    viewer.window.add_dock_widget(segment_er, name="make mask")
    viewer.reset_view()
    napari.run()
