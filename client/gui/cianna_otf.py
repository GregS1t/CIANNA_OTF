#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIANNA OTF — Step 4 UI (English)

A lightweight PyQt6 GUI for inspecting FITS images, drawing an ROI, and
browsing available models read from a local `CIANNA_models.xml` file.

Key features
------------
- Reads immutable client config from `CODE/client/configs/param_cianna_rts_client.json`.
- Computes server URL from JSON: `CLIENT_CONNEXION` → local/remote.
- **Models are loaded from the local XML** pointed by `LOCAL_FILE_MODELS`.
  The displayed model names come from the `<Model id="...">` attribute.
- A **Models** menu lists available models and opens a detail sheet with all
  relevant metadata (ReleaseDate, dims, YOLO*, quantization/normalization,
  datasets, links, etc.). The menu always offers **Download latest
  CIANNA_models.xml from server…** to fetch/replace the local model list on
  demand.
- A **Server** menu shows the effective URL/host/port.
- A **log pane** at the bottom displays key runtime messages.

Run
---
    python step04_params_panel.py

Dependencies
------------
    conda install -y -c conda-forge pyqt6 pyqtgraph astropy numpy requests
"""
from __future__ import annotations

import os
import sys
import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QObject, pyqtSignal, QThread 
import pyqtgraph as pg


# --------------------
# Global UI settings
# --------------------
try:
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
except AttributeError:
    pass

pg.setConfigOptions(imageAxisOrder='row-major')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


#
# Class dedicated to retrieve prediction files 
# from the server in a separate thread
#

class DownloadWorker(QObject):
    finished = pyqtSignal(str)         # Signal émis quand le téléchargement réussit
    failed = pyqtSignal(str)           # Signal émis si erreur

    def __init__(self, server_url, process_id, destination_folder):
        super().__init__()
        self.server_url = server_url
        self.process_id = process_id
        self.destination_folder = destination_folder

    def run(self):
        url = f"{self.server_url}/jobs/{self.process_id}/results"
        timeout = 20  # secondes max pour attendre que le fichier soit dispo
        poll_interval = 1  # seconde
        start = time.time()

        while time.time() - start < timeout:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    filename = f"net0_rts_{self.process_id}.dat"
                    os.makedirs(self.destination_folder, exist_ok=True)
                    dest_path = os.path.join(self.destination_folder, filename)
                    with open(dest_path, "wb") as f:
                        f.write(r.content)
                    self.finished.emit(dest_path)
                    return
                elif r.status_code == 404:
                    time.sleep(poll_interval)
                else:
                    self.failed.emit(f"Unexpected status code: {r.status_code}")
                    return
            except Exception as e:
                self.failed.emit(str(e))
                return

        self.failed.emit("Timeout waiting for result.")


# --------------------
# XML models parsing
# --------------------

def _clean_text(text: Optional[str]) -> Optional[str]:
    """Return a stripped string without surrounding quotes, or None.

    Parameters
    ----------
    text : Optional[str]
        Raw text to clean.

    Returns
    -------
    Optional[str]
        Cleaned text or None if empty.
    """
    if text is None:
        return None
    s = text.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1]
    return s or None


@dataclass
class TrainingDataset:
    """Training dataset description parsed from the XML."""

    name: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None


@dataclass
class ModelInfo:
    """Aggregated model information for UI display and selection.

    Notes
    -----
    The `name` field is the model display name and is taken from the `<Model id>`
    attribute when available. If the `id` attribute is missing, we fall back to
    the `<Name>` tag.
    """

    # Display name (prefer `<Model id="...">`)
    name: str

    # Known XML fields
    release_date: Optional[str] = None
    base_memory_footprint: Optional[str] = None
    per_image_memory_footprint: Optional[str] = None
    original_input_dim: Optional[str] = None
    min_input_dim: Optional[str] = None
    max_input_dim: Optional[str] = None
    yolo_grid_elem_dim: Optional[str] = None
    yolo_box_count: Optional[str] = None
    yolo_param_count: Optional[str] = None
    yolo_grid_count: Optional[str] = None
    data_normalization: Optional[str] = None
    data_normalization_type: Optional[str] = None
    data_quantization: Optional[str] = None
    receptive_field: Optional[str] = None
    training_quantization: Optional[str] = None
    inference_mode: Optional[str] = None
    inference_quantization: Optional[str] = None
    inference_patch_shift: Optional[str] = None
    inference_orig_offset: Optional[str] = None
    url_link: Optional[str] = None
    training_datasets: List[TrainingDataset] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    comments: Optional[str] = None

    # For UI combo population
    quantizations: List[str] = field(default_factory=list)
    norms: List[str] = field(default_factory=list)

    # Other unmapped attributes
    attributes: Dict[str, Any] = field(default_factory=dict)


def xml_parse_models(xml_path: str) -> List[ModelInfo]:
    """
    Parse either a legacy CIANNA_models.xml file or a UWS job XML file
    and return a list of ModelInfo objects.
    """
    import xml.etree.ElementTree as ET
    import traceback

    models: List[ModelInfo] = []
    if not xml_path or not os.path.exists(xml_path):
        return models

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        tag_lower = root.tag.lower()
        ns = {"uws": "http://www.ivoa.net/xml/UWS/v1.0"}

        def _clean_text(text):
            if not text:
                return None
            s = text.strip().strip('"').strip("'")
            return s or None

        # --- Case 1: UWS job format ------------------------------------------
        if "uws" in tag_lower or "job" in tag_lower:
            params = {}
            for param in root.findall(".//uws:parameter", namespaces=ns):
                pid = param.attrib.get("id")
                if pid:
                    params[pid] = _clean_text(param.text)

            info = ModelInfo(name=params.get("ModelName", "UnknownModel"))

            tagmap = {
                "ReleaseDate": "release_date",
                "BaseMemoryFootprint": "base_memory_footprint",
                "PerImageMemoryFootprint": "per_image_memory_footprint",
                "OriginalInputDim": "original_input_dim",
                "MinInputDim": "min_input_dim",
                "MaxInputDim": "max_input_dim",
                "YOLOGridElemDim": "yolo_grid_elem_dim",
                "YOLOBoxCount": "yolo_box_count",
                "YOLOParamCount": "yolo_param_count",
                "YOLOGridCount": "yolo_grid_count",
                "DataNormalization": "data_normalization",
                "DataNormalizationType": "data_normalization_type",
                "DataQuantization": "data_quantization",
                "ReceptiveField": "receptive_field",
                "TrainingQuantization": "training_quantization",
                "InferenceMode": "inference_mode",
                "InferenceQuantization": "inference_quantization",
                "InferencePatchShift": "inference_patch_shift",
                "InferenceOrigOffset": "inference_orig_offset",
                "ModelURL": "url_link",
                "CheckpointPath": "checkpoint_path",
            }

            for key, attr in tagmap.items():
                val = params.get(key)
                if val is not None:
                    setattr(info, attr, val)

            # Training datasets
            ds = TrainingDataset(
                name=_clean_text(params.get("TrainingDatasetName")),
                description=_clean_text(params.get("TrainingDatasetDescription")),
                location=_clean_text(params.get("TrainingDatasetLocation")),
            )
            if ds.name or ds.description or ds.location:
                info.training_datasets.append(ds)

            # Quantization & normalization lists
            info.quantizations = [
                q for q in [
                    info.training_quantization,
                    info.inference_quantization,
                    info.data_quantization,
                ] if q
            ] or ["FP32C_FP32A"]
            info.norms = [info.data_normalization_type] if info.data_normalization_type else ["tanh"]

            # Extra UWS metadata
            info.attributes["jobId"] = root.findtext("uws:jobId", namespaces=ns)
            info.attributes["phase"] = root.findtext("uws:phase", namespaces=ns)
            info.attributes["creationTime"] = root.findtext("uws:creationTime", namespaces=ns)

            models.append(info)
            return models

    except Exception:
        traceback.print_exc()
        return []


# --------------------
# FITS / WCS helpers
# --------------------

def robust_first_2d_plane(data: np.ndarray) -> np.ndarray:
    """Return a 2D plane from N-D FITS data, preferring the first slice.

    Parameters
    ----------
    data : np.ndarray
        FITS data (2D or ND).

    Returns
    -------
    np.ndarray
        A 2D array.
    """
    arr = np.asarray(data)
    if arr.ndim == 2:
        return arr
    if arr.ndim >= 3:
        h, w = arr.shape[-2:]
        idx = (0,) * (arr.ndim - 2) + (slice(None), slice(None))
        try:
            out = arr[idx]
            if out.shape == (h, w):
                return out
        except Exception:
            pass
        out = arr.reshape((-1, h, w))[0]
        return out
    raise ValueError("Unsupported FITS dimensionality")


def reduce_wcs_to_celestial_2d(wcs: WCS) -> Optional[WCS]:
    """Reduce a WCS to a 2D celestial WCS when possible.

    Parameters
    ----------
    wcs : WCS
        Raw WCS from FITS header.

    Returns
    -------
    Optional[WCS]
        A 2D celestial WCS or None if unavailable.
    """
    if wcs is None:
        return None
    try:
        if getattr(wcs, 'has_celestial', False):
            try:
                w = wcs.celestial
                if getattr(w, 'naxis', 2) == 2:
                    return w
            except Exception:
                pass
        w = wcs
        if getattr(w, 'naxis', 2) > 2:
            try:
                while w.naxis > 2:
                    w = w.dropaxis(2)
            except Exception:
                return None
        if getattr(w, 'has_celestial', False) and getattr(w, 'naxis', 2) == 2:
            return w
    except Exception:
        pass
    return None


def load_fits_first_plane(path: str) -> Tuple[np.ndarray, Optional[WCS]]:
    """Load a FITS file and return the first 2D plane and a 2D celestial WCS.

    Parameters
    ----------
    path : str
        Path to a FITS file.

    Returns
    -------
    Tuple[np.ndarray, Optional[WCS]]
        Image and WCS (if available).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with fits.open(path, memmap=True) as hdul:
        if len(hdul) == 0:
            raise ValueError("Empty FITS (no HDU)")
        hdu = hdul[0]
        data = hdu.data
        if data is None:
            raise ValueError("Primary HDU has no data")
        img = robust_first_2d_plane(data)
        img = np.nan_to_num(img, copy=False)
        wcs2d = None
        try:
            wcs_full = WCS(hdu.header)
            wcs2d = reduce_wcs_to_celestial_2d(wcs_full)
        except Exception:
            wcs2d = None
        return img, wcs2d


# --------------------
# JSON config (read-only)
# --------------------

def find_config_upwards(start: str, relative: str) -> Optional[str]:
    """Search upwards from *start* for a *relative* path and return it if found.

    Parameters
    ----------
    start : str
        Starting directory.
    relative : str
        Relative path to look for while walking up the tree.

    Returns
    -------
    Optional[str]
        Found absolute path or None.
    """
    start = os.path.abspath(start)
    parts = start.split(os.sep)
    for i in range(len(parts), 0, -1):
        base = os.sep.join(parts[:i])
        cand = os.path.join(base, relative)
        if os.path.exists(cand):
            return cand
    return None


def fixed_config_path() -> Optional[str]:
    """Return the expected path to the client JSON config if present.

    The function checks a couple of explicit locations and otherwise searches up
    from both the script directory and the CWD for
    `CODE/client/configs/param_cianna_rts_client.json`.

    Returns
    -------
    Optional[str]
        Absolute path or None.
    """
    explicit = [
        os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'configs', 'param_cianna_rts_client.json')),
        os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', 'client', 'configs', 'param_cianna_rts_client.json')),
    ]
    for p in explicit:
        if os.path.exists(p):
            return p
    rel = os.path.join('CODE', 'client', 'configs', 'param_cianna_rts_client.json')
    for base in (SCRIPT_DIR, os.getcwd()):
        found = find_config_upwards(base, rel)
        if found:
            return found
    return None


@dataclass
class ServerConfig:
    """Server-related configuration derived from the client JSON."""

    url: str
    mode: str  # 'local' or 'remote'
    models_endpoint: str
    local_models_path: Optional[str] = None


def compute_server_config(path: Optional[str]) -> ServerConfig:
    """Compute server configuration from the client JSON.

    Rules
    -----
    - If `CLIENT_CONNEXION == "local"` → `http://127.0.0.1:5000`.
    - If `CLIENT_CONNEXION == "remote"` → `SERVER_URL`.
    - **Requested change**: resolve `LOCAL_FILE_MODELS` with
      `os.path.join("../..", conf.get("LOCAL_FILE_MODELS"))`.

    Parameters
    ----------
    path : Optional[str]
        Path to the client JSON config.

    Returns
    -------
    ServerConfig
        Computed config values.
    """
    default_local = "http://127.0.0.1:5000"
    default_endpoint_candidates = ["/model-files/CIANNA_models.xml", "/CIANNA_models.xml"]
    if not path:
        return ServerConfig(url=default_local, mode='local', models_endpoint=default_endpoint_candidates[0], local_models_path=None)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            conf = json.load(f)

        # IMPORTANT: per user request, force this resolution rule
        lfm_raw = conf.get('LOCAL_FILE_MODELS')
        if isinstance(lfm_raw, str) and lfm_raw.strip():
            # Join with "../.." exactly as specified
            models_path = os.path.normpath(os.path.join("../..", lfm_raw.strip()))
        else:
            models_path = None

        mode = str(conf.get('CLIENT_CONNEXION', '')).strip().lower()
        if mode == 'remote':
            url = str(conf.get('SERVER_URL', '')).strip() or default_local
            url = os.path.expandvars(url)
            endpoint = (
                str(conf.get('MODELS_XML_ENDPOINT') or conf.get('MODELS_ENDPOINT') or conf.get('MODELS_XML_URL_PATH') or "").strip()
            )
            if not endpoint:
                endpoint = default_endpoint_candidates[0]
            return ServerConfig(url=url, mode='remote', models_endpoint=endpoint, local_models_path=models_path)
        else:
            endpoint = (
                str(conf.get('MODELS_XML_ENDPOINT') or conf.get('MODELS_ENDPOINT') or conf.get('MODELS_XML_URL_PATH') or "").strip()
            )
            if not endpoint:
                endpoint = default_endpoint_candidates[0]
            return ServerConfig(url=default_local, mode='local', models_endpoint=endpoint, local_models_path=models_path)
    except Exception:
        return ServerConfig(url=default_local, mode='local', models_endpoint=default_endpoint_candidates[0], local_models_path=None)


def parse_host_port(url: str) -> Tuple[str, int]:
    """Return (host, port) extracted from a URL, with sensible defaults.

    Parameters
    ----------
    url : str
        URL like `http://host:port`.

    Returns
    -------
    Tuple[str, int]
        Host and port.
    """
    p = urlparse(url)
    host = p.hostname or '127.0.0.1'
    port = p.port or (5000 if (p.scheme or 'http') == 'http' else 443)
    return host, port


# --------------------
# Main window
# --------------------
class MainWindow(QtWidgets.QMainWindow):
    """Main application window for the CIANNA OTF step-4 GUI."""

    def __init__(self) -> None:
        """Initialize the main window and build the UI/layout.

        The window shows:
        - Top/left: FITS image viewer (pyqtgraph) + a live readout label.
        - Top/right: parameters (server mode, model/quantization/normalization,
            and a checkbox to enable/disable ROI usage).
        - Bottom: a log pane (QPlainTextEdit) for runtime messages.

        Models are loaded from the local XML pointed to by the client JSON
        (LOCAL_FILE_MODELS). A default model is provided if none can be parsed.
        """
        super().__init__()
        self.setWindowTitle("CIANNA OTF — DEMO - CLIENT GUI")
        self.resize(1480, 980)
        self.setMinimumSize(1360, 920)

        # ---- State ------------------------------------------------------------
        self.image: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.wcs: Optional[WCS] = None
        self.models: List[ModelInfo] = []
        self.use_roi: bool = True           # whether ROI is enabled/visible
        self.req_counter: int = 0           # unique counter for requests

        # Read client JSON and compute server config
        conf_path = fixed_config_path()
        self.server_conf = compute_server_config(conf_path)

        # ---- Central layout: vertical splitter (top: UI, bottom: logs) --------
        main_splitter = QtWidgets.QSplitter()
        main_splitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.setCentralWidget(main_splitter)

        # Top: horizontal splitter (left: image, right: params)
        top_splitter = QtWidgets.QSplitter()
        top_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        main_splitter.addWidget(top_splitter)

        # ---- Left panel: image + readout + ROI (added on demand) --------------
        left = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left)
        left_v.setContentsMargins(6, 6, 6, 6)

        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        left_v.addWidget(self.image_view, 1)

        self.readout = QtWidgets.QLabel("No image loaded.")
        self.readout.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        left_v.addWidget(self.readout)

        # ROI is created now but only added to the view when an image is loaded
        self.roi = pg.RectROI([20, 20], [100, 100], pen=pg.mkPen(width=2))
        self.roi.setZValue(10)
        self.roi.sigRegionChanged.connect(self._on_roi_changed)

        top_splitter.addWidget(left)

        # ---- Right panel: parameters ------------------------------------------
        right = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(right)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self.label_server_mode = QtWidgets.QLabel(self.server_conf.mode)
        form.addRow("Mode:", self.label_server_mode)

        self.combo_model = QtWidgets.QComboBox()
        form.addRow("Model:", self.combo_model)

        self.combo_quant = QtWidgets.QComboBox()
        form.addRow("Quantization:", self.combo_quant)

        self.combo_norm = QtWidgets.QComboBox()
        form.addRow("Normalization:", self.combo_norm)

        # Checkbox to toggle ROI usage
        self.chk_use_roi = QtWidgets.QCheckBox("Use ROI")
        self.chk_use_roi.setChecked(False)
        self.chk_use_roi.toggled.connect(self._on_toggle_roi)
        form.addRow("Selection:", self.chk_use_roi)
        self.use_roi = False

        # Identify button (disabled until an image is opened)
        self.btn_identify = QtWidgets.QPushButton("Identify sources")
        self.btn_identify.setEnabled(False)
        self.btn_identify.clicked.connect(self._on_identify_sources)
        form.addRow(self.btn_identify)

        top_splitter.addWidget(right)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 0)

        # ---- Bottom: log pane --------------------------------------------------
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        )
        self.log_view.setFont(font)
        self.log_view.setMaximumBlockCount(2000)
        main_splitter.addWidget(self.log_view)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)

        # ---- Status bar + menus -----------------------------------------------
        self.status = self.statusBar()
        self.status.showMessage("Ready.")
        self._build_menus()

        # ---- Models: load from local file, then ensure defaults ----------------
        self._load_models_from_local_file(self.server_conf.local_models_path)
        self._seed_default_model_if_needed()
        self._populate_combo_from_model()
        self.combo_model.currentIndexChanged.connect(
            lambda _i: self._populate_combo_from_model()
        )

        # Bind backend helpers (existing endpoints/functions from your client API)
        self._bind_backend_helpers()


        have_all = all([
            self.fn_get_model_info,
            self.fn_create_xml_param,
            self.fn_send_xml_fits_to_server,
            self.fn_poll_for_completion,
            self.fn_download_result,
        ])
        if not have_all:
            self.btn_identify.setToolTip("Client API helpers not found; request sending will be disabled until available.")

        # ---- Initial logs ------------------------------------------------------
        self.log(f"Config: mode={self.server_conf.mode} url={self.server_conf.url}")
        self.log(f"Local models: {self.server_conf.local_models_path or 'not set'}")
        # ---- Check models.xml presence ----
        models_file = self.server_conf.local_models_path
        if not models_file or not os.path.exists(models_file):
            self.log("[WARNING] No file 'CIANNA_models.xml' found, please update.")
            self.btn_identify.setEnabled(False)
        else:
            self.log(f"[INFO] CIANNA_models.xml file found (consider an update) : {models_file}")


        # ---- Utility: logging ----
    def log(self, msg: str) -> None:
        """Append a log line with timestamp to the bottom pane and status bar (safe during early init)."""
        ts = time.strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        # Pendant l'init, log_view peut ne pas encore exister
        if hasattr(self, "log_view") and isinstance(self.log_view, QtWidgets.QPlainTextEdit):
            self.log_view.appendPlainText(line)
        else:
            print(line)  # fallback console
        if hasattr(self, "status") and isinstance(self.status, QtWidgets.QStatusBar):
            self.status.showMessage(msg, 5000)


    # ---- Menus ----
    def _build_menus(self) -> None:
        """Create application menus (File, Server, Models, Help)."""
        menu = self.menuBar()

        # File
        m_file = menu.addMenu("&File")
        act_open = QtGui.QAction("Open FITS…", self); act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open_fits); m_file.addAction(act_open)
        m_file.addSeparator()
        act_quit = QtGui.QAction("Quit", self); act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close); m_file.addAction(act_quit)

        # Server
        m_server = menu.addMenu("&Server")
        act_info = QtGui.QAction("Server info…", self)
        act_info.triggered.connect(self._show_server_info)
        m_server.addAction(act_info)

        # Models
        self.m_models = menu.addMenu("&Models")
        self._refresh_models_menu()

        # Help
        m_help = menu.addMenu("&Help")
        act_welcome = QtGui.QAction("Welcome", self)
        act_welcome.triggered.connect(self._show_welcome_dialog)
        m_help.addAction(act_welcome)


    def _show_welcome_dialog(self) -> None:
        """Show a small 'Welcome' window with application info."""
        version = "0.4.0"
        build_date = "2025-09-01"
        author = "CIANNA Team"
        qt_ver = QtCore.QT_VERSION_STR
        pyqt_ver = QtCore.PYQT_VERSION_STR

        html = (
            "<h2 style='margin:0'>CIANNA OTF — Client GUI</h2>"
            f"<p><b>Version:</b> {version}<br>"
            f"<b>Date:</b> {build_date}<br>"
            f"<b>Author:</b> {author}<br>"
            f"<b>Qt:</b> {qt_ver} — <b>PyQt:</b> {pyqt_ver}</p>"
            "<p>This GUI lets you open FITS files, select some parameters, "
            "CIANNA models and send request to server. It's just a mockup !</p>"
        )

        QtWidgets.QMessageBox.about(self, "Welcome", html)

    def _refresh_models_menu(self) -> None:
        """Rebuild the Models menu with the current model list.

        """
        self.m_models.clear()
        act_dl = QtGui.QAction("Download CIANNA models from server…", self)
        act_dl.triggered.connect(self._download_models_xml_from_server)
        self.m_models.addAction(act_dl)
        self.m_models.addSeparator()

        if not self.models:
            self.models = [ModelInfo('SDC1_Cornu_2024', quantizations=['FP32C_FP32A'], norms=['tanh'])]

        seen = set(); uniq: List[ModelInfo] = []
        for m in self.models:
            if m.name not in seen:
                seen.add(m.name); uniq.append(m)
        self.models = uniq

        for m in self.models:
            act = QtGui.QAction(m.name, self)
            act.triggered.connect(lambda _=False, mm=m: self._show_model_details(mm))
            self.m_models.addAction(act)

        self._sync_combo_model_from_models()

    # ---- FITS ----
    def _on_open_fits(self) -> None:
        """Open a FITS file and display the first 2D plane."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open FITS", os.getcwd(), "FITS files (*.fits *.fit *.fz *.fits.gz);;All files (*)",
        )
        if not path:
            return
        try:
            img, wcs2d = load_fits_first_plane(path)
            self.image = img; self.image_path = path; self.wcs = wcs2d
            self.log(f"[INFO] FITS dimensions : {img.shape[0]} x {img.shape[1]}")
            self.image_view.setImage(img); self.image_view.autoRange()
            extra = " (WCS 2D OK)" if self.wcs is not None else " (no usable celestial WCS)"
            self.status.showMessage(
                f"Displayed: {os.path.basename(path)} | shape={img.shape} | min={np.min(img):.3g} max={np.max(img):.3g}{extra}"
            )
            self.log(f"Loaded FITS: {path}")

            # Prepare ROI geometry
            H, W = img.shape
            w0 = max(10, int(W * 0.3)); h0 = max(10, int(H * 0.3))
            x0 = int(W * 0.1); y0 = int(H * 0.1)
            self.roi.setPos([x0, y0]); self.roi.setSize([w0, h0])

            # Show/hide ROI based on checkbox
            try:
                self.image_view.getView().removeItem(self.roi)
            except Exception:
                pass
            if self.use_roi:
                self.image_view.addItem(self.roi)
                self._on_roi_changed()
            else:
                self._update_readout_full_image()
            
            # NEW: enable identify button when an image is loaded
            self.btn_identify.setEnabled(True)

        except Exception as e:
            self.log(f"FITS error: {e}")
            QtWidgets.QMessageBox.critical(self, "FITS error", str(e))

    # ---- ROI ----
    def _on_roi_changed(self) -> None:
        """Update readout when the ROI is moved/resized."""
        if self.image is None:
            self.readout.setText("No image loaded.")
            return
        pos = self.roi.pos(); size = self.roi.size()
        x = float(pos.x()); y = float(pos.y())
        w = float(size.x()); h = float(size.y())
        cx = x + w/2.0; cy = y + h/2.0

        H, W = self.image.shape
        x_d = max(0.0, min(x, W)); y_d = max(0.0, min(y, H))
        w_d = max(0.0, min(w, W - x_d)); h_d = max(0.0, min(h, H - y_d))
        cx_d = x_d + w_d/2.0; cy_d = y_d + h_d/2.0

        ra_deg_str = dec_deg_str = sexa_str = "—"
        if self.wcs is not None:
            try:
                world = self.wcs.pixel_to_world(cx_d, cy_d)
                if hasattr(world, 'ra') and hasattr(world, 'dec'):
                    ra_deg = float(world.ra.deg); dec_deg = float(world.dec.deg)
                    ra_deg_str = f"{ra_deg:.6f}°"; dec_deg_str = f"{dec_deg:.6f}°"
                    sc = SkyCoord(ra=world.ra, dec=world.dec, frame=world.frame)
                    sexa_str = sc.to_string('hmsdms', precision=2)
            except Exception:
                pass

        # self.readout.setText(
        #     (
        #         f"ROI — x={x_d:.1f}, y={y_d:.1f}, w={w_d:.1f}, h={h_d:.1f} | "
        #         f"center(px) → cx={cx_d:.1f}, cy={cy_d:.1f} | "
        #         f"center(WCS) → RA={ra_deg_str}, Dec={dec_deg_str}  [{sexa_str}]"
        #     )
        # )

    def _on_toggle_roi(self, checked: bool) -> None:
        """Handle ROI visibility toggle.

        If checked, the ROI is shown and used for requests; if unchecked, the
        entire image will be used for identification.
        """
        self.use_roi = bool(checked)
        if self.image is None:
            self.log("ROI toggled (no image loaded).")
            return

        # Ensure ROI geometry is sensible for the current image
        H, W = self.image.shape
        w0 = max(10, int(W * 0.3)); h0 = max(10, int(H * 0.3))
        x0 = int(W * 0.1); y0 = int(H * 0.1)
        self.roi.setPos([x0, y0]); self.roi.setSize([w0, h0])

        try:
            self.image_view.getView().removeItem(self.roi)
        except Exception:
            pass

        if self.use_roi:
            self.image_view.addItem(self.roi)
            self._on_roi_changed()
            self.log("ROI enabled (selection will use the ROI).")
        else:
            self._update_readout_full_image()
            self.log("ROI disabled (selection will use the full image).")


    def _update_readout_full_image(self) -> None:
        """Update the readout to reflect full-image selection (ROI disabled)."""
        if self.image is None:
            self.readout.setText("No image loaded.")
            return
        H, W = self.image.shape
        cx, cy = W / 2.0, H / 2.0

        ra_deg_str = dec_deg_str = sexa_str = "—"
        if self.wcs is not None:
            try:
                world = self.wcs.pixel_to_world(cx, cy)
                if hasattr(world, 'ra') and hasattr(world, 'dec'):
                    ra_deg = float(world.ra.deg); dec_deg = float(world.dec.deg)
                    ra_deg_str = f"{ra_deg:.6f}°"; dec_deg_str = f"{dec_deg:.6f}°"
                    sc = SkyCoord(ra=world.ra, dec=world.dec, frame=world.frame)
                    sexa_str = sc.to_string('hmsdms', precision=2)
            except Exception:
                pass

        # self.readout.setText(
        #     (
        #         f"FULL IMAGE — x=0.0, y=0.0, w={W:.1f}, h={H:.1f} | "
        #         f"center(px) → cx={cx:.1f}, cy={cy:.1f} | "
        #         f"center(WCS) → RA={ra_deg_str}, Dec={dec_deg_str}  [{sexa_str}]"
        #     )
        # )

    def get_current_selection(self) -> Tuple[float, float, float, float]:
        """Return the current selection (x, y, w, h) in pixel coordinates.

        When ROI is enabled, this returns the ROI rectangle (clamped to image
        bounds). When disabled, it returns the full image extent.

        Returns
        -------
        Tuple[float, float, float, float]
            (x, y, w, h) in pixels (floats).
        """
        if self.image is None:
            return (0.0, 0.0, 0.0, 0.0)

        H, W = self.image.shape
        if not self.use_roi:
            return (0.0, 0.0, float(W), float(H))

        pos = self.roi.pos(); size = self.roi.size()
        x = float(pos.x()); y = float(pos.y())
        w = float(size.x()); h = float(size.y())

        # Clamp to image bounds
        x_d = max(0.0, min(x, W))
        y_d = max(0.0, min(y, H))
        w_d = max(0.0, min(w, W - x_d))
        h_d = max(0.0, min(h, H - y_d))
        return (x_d, y_d, w_d, h_d)


    # ---- Server helpers ----
    def _show_server_info(self) -> None:
        """Show current server info (mode, URL, host, port)."""
        host, port = parse_host_port(self.server_conf.url)
        QtWidgets.QMessageBox.information(
            self, "Server info",
            f"Mode: {self.server_conf.mode} \nURL: {self.server_conf.url} \nAddress: {host} \nPort: {port}"
        )

    # ---- Models helpers ----
    def _load_models_from_local_file(self, path: Optional[str]) -> None:
        """Load models from a local XML path (if provided)."""
        if not path:
            self.log("LOCAL_FILE_MODELS not set — using default model")
            self._refresh_models_menu()
            return
        if not os.path.exists(path):
            self.log(f"Models file not found: {path}")
            self._refresh_models_menu()
            return
        loaded = xml_parse_models(path)
        if loaded:
            self.models = loaded
            self.log(f"Loaded {len(self.models)} models from {path}")
        else:
            self.log(f"No models parsed from {path} — using default model")
        self._refresh_models_menu()
        self._sync_combo_model_from_models()
        self._populate_combo_from_model()

    def _seed_default_model_if_needed(self) -> None:
        """Ensure there is at least one default model when none was loaded."""
        if not self.models:
            self.models = [ModelInfo('SDC1_Cornu_2024', quantizations=['FP32C_FP32A'], norms=['tanh'])]
        self._sync_combo_model_from_models()

    def _sync_combo_model_from_models(self) -> None:
        """Sync the model combo box with the current model list."""
        cur = self.combo_model.currentText()
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        for m in self.models:
            self.combo_model.addItem(m.name, userData=m)
        idx = self.combo_model.findText(cur) if cur else 0
        if idx < 0:
            idx = 0
        self.combo_model.setCurrentIndex(idx)
        self.combo_model.blockSignals(False)

    def _populate_combo_from_model(self) -> None:
        """Populate quantization/normalization comboboxes from the selected model,
        ensuring they are refreshed and handle comma-separated values.

        """
        m: Optional[ModelInfo] = self.combo_model.currentData()

        # Toujours vider avant de remplir
        self.combo_quant.clear()
        self.combo_norm.clear()

        if isinstance(m, ModelInfo):
            # --- Quantization ---
            quantizations: List[str] = []
            for q in (m.quantizations or []):
                quantizations.extend([v.strip() for v in q.split(",") if v.strip()])
            seen = set(); quantizations = [x for x in quantizations if not (x in seen or seen.add(x))]
            if not quantizations:
                quantizations = ["FP32C_FP32A"]
            self.combo_quant.addItems(quantizations)

            # --- Normalization ---
            norms: List[str] = []
            for n in (m.norms or []):
                norms.extend([v.strip() for v in n.split(",") if v.strip()])
            # remove duplicates while preserving order
            seen = set(); norms = [x for x in norms if not (x in seen or seen.add(x))]

            if not norms:
                norms = ["tanh"]
            self.combo_norm.addItems(norms)

        else:
            self.combo_quant.addItem("FP32C_FP32A")
            self.combo_norm.addItem("tanh")

        # Sélectionne le premier élément par défaut
        if self.combo_quant.count() > 0:
            self.combo_quant.setCurrentIndex(0)
        if self.combo_norm.count() > 0:
            self.combo_norm.setCurrentIndex(0)

    def _show_model_details(self, m: ModelInfo) -> None:
        """Show a rich details sheet for the selected model."""
        html = [f"<h2 style='margin-top:0'>{m.name}</h2>"]

        def row(label: str, value: Optional[str]) -> None:
            if value not in (None, ""):
                html.append(f"<p><b>{label}:</b> {value}</p>")

        row("ReleaseDate", m.release_date)
        row("BaseMemoryFootprint", m.base_memory_footprint)
        row("PerImageMemoryFootprint", m.per_image_memory_footprint)
        row("OriginalInputDim", m.original_input_dim)
        row("MinInputDim", m.min_input_dim)
        row("MaxInputDim", m.max_input_dim)
        row("YOLOGridElemDim", m.yolo_grid_elem_dim)
        row("YOLOBoxCount", m.yolo_box_count)
        row("YOLOParamCount", m.yolo_param_count)
        row("YOLOGridCount", m.yolo_grid_count)
        row("DataNormalization", m.data_normalization)
        row("DataNormalizationType", m.data_normalization_type)
        row("DataQuantization", m.data_quantization)
        row("ReceptiveField", m.receptive_field)
        row("TrainingQuantization", m.training_quantization)
        row("InferenceMode", m.inference_mode)
        row("InferenceQuantization", m.inference_quantization)
        row("InferencePatchShift", m.inference_patch_shift)
        row("InferenceOrigOffset", m.inference_orig_offset)
        if m.url_link:
            link = m.url_link if m.url_link.startswith('http') else ('https://' + m.url_link)
            html.append(f"<p><b>URL_Link:</b> <a href='{link}'>{m.url_link}</a></p>")
        if m.training_datasets:
            html.append("<h3>TrainingDatasets</h3>")
            for ds in m.training_datasets:
                html.append("<div style='margin-left:1em'>")
                row("Name", ds.name)
                row("Description", ds.description)
                if ds.location:
                    lk = ds.location if ds.location.startswith('http') else ds.location.strip('"')
                    html.append(f"<p><b>Location:</b> {lk}</p>")
                html.append("</div>")
        row("CheckpointPath", m.checkpoint_path)
        row("Comments", m.comments)
        if m.quantizations:
            row("Quantizations", ", ".join(m.quantizations))
        if m.norms:
            row("Normalizations", ", ".join(m.norms))
        if m.attributes:
            html.append("<h3>Other attributes</h3>")
            for k, v in m.attributes.items():
                row(k, v)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Model: {m.name}")
        dlg.resize(720, 600)
        lay = QtWidgets.QVBoxLayout(dlg)
        viewer = QtWidgets.QTextBrowser(); viewer.setOpenExternalLinks(True); viewer.setHtml("".join(html))
        lay.addWidget(viewer, 1)
        btn = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btn.rejected.connect(dlg.reject); btn.accepted.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.exec()

    def _download_models_xml_from_server(self) -> None:
        """Download and load `CIANNA_models.xml` from the server endpoint.

        This version FORCE-SAVES the file (no dialog), overwriting the destination.
        Destination:
            - If LOCAL_FILE_MODELS is set: write to that path.
            - Else: write to ./CIANNA_models.xml in the current working directory.
        """
        endpoint = self.server_conf.models_endpoint or "/model-files/CIANNA_models.xml"
        endpoint = endpoint if endpoint.startswith('/') else '/' + endpoint
        url = urljoin(self.server_conf.url.rstrip('/') + '/', endpoint.lstrip('/'))
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            self.log(f"Connection refused: {url}")
            QtWidgets.QMessageBox.critical(self, "Connection refused",
                                        f"Cannot connect to\n{url}\n\nCheck address/port (Server → Server info…).")
            return
        except Exception as e:
            self.log(f"Download failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Download failed", f"GET {url}\n\n{e}")
            return

        # FORCE save path (no dialog)
        dest = self.server_conf.local_models_path or os.path.join(os.getcwd(), "CIANNA_models.xml")
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        except Exception:
            # If dirname is empty (e.g. dest is just a filename), it's fine.
            pass

        try:
            with open(dest, 'wb') as f:
                f.write(r.content)
            loaded = xml_parse_models(dest)
            if loaded:
                self.models = loaded
                self.log(f"Downloaded and loaded {len(self.models)} model(s) to {dest}")
            else:
                self.log("Downloaded XML parsed to 0 model(s); keeping previous list")
            self._refresh_models_menu()
            self._sync_combo_model_from_models()
            self._populate_combo_from_model()
        except Exception as e:
            self.log(f"Save failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def _on_download_success(self, filepath):
        self.log(f"Result downloaded to {filepath}")
        QtWidgets.QMessageBox.information(
            self, "Identify sources",
            f"Job completed successfully.\nResult saved to:\n{filepath}"
        )

    def _on_download_failure(self, error_msg):
        self.log(f"Failed to download result: {error_msg}")
        QtWidgets.QMessageBox.warning(
            self, "Identify sources",
            f"Job completed but result download failed:\n{error_msg}"
        )


    def _on_identify_sources(self) -> None:
        """Build and send the identification request using existing endpoints.

        Uses the current selection (ROI if enabled, otherwise full image),
        converts the center to RA/Dec when a valid 2D celestial WCS is present,
        builds the XML payload with the selected model/quantization, then sends
        the job to the server, polls for completion, and downloads the result.
        """
        # --- Sanity checks
        if self.image is None or not self.image_path:
            QtWidgets.QMessageBox.warning(self, "Identify sources", "Please open a FITS image first.")
            return

        required = [
            self.fn_get_model_info, self.fn_create_xml_param,
            self.fn_send_xml_fits_to_server, self.fn_poll_for_completion,
            self.fn_download_result,
        ]
        if not all(required):
            QtWidgets.QMessageBox.critical(
                self, "Backend unavailable",
                "Required client helpers are not bound. Please ensure the client API module is importable."
            )
            return

        # --- Selection (ROI or full image)
        x, y, w, h = self.get_current_selection()
        if w <= 0 or h <= 0:
            QtWidgets.QMessageBox.warning(self, "Identify sources", "Selection area is empty.")
            return
        cx, cy = x + w / 2.0, y + h / 2.0

        # --- World coords if available
        if self.wcs is not None:
            try:
                world = self.wcs.pixel_to_world(cx, cy)
                ra_deg = float(world.ra.deg) if hasattr(world, "ra") else 0.0
                dec_deg = float(world.dec.deg) if hasattr(world, "dec") else 0.0
            except Exception:
                ra_deg, dec_deg = 0.0, 0.0
                self.log("No valid WCS for RA/Dec; sending 0.0, 0.0")
        else:
            ra_deg, dec_deg = 0.0, 0.0
            self.log("No WCS available; sending RA=0.0, Dec=0.0")

        # --- Model + params from UI
        model: ModelInfo = self.combo_model.currentData()
        yolo_model = model.name if isinstance(model, ModelInfo) else self.combo_model.currentText()
        quantization = self.combo_quant.currentText() or "FP32C_FP32A"
        normalization = self.combo_norm.currentText() or "tanh"  # currently for UI/log only

        # Look up model info from local XML (existing helper expects path + model name)
        config = {"LOCAL_FILE_MODELS": self.server_conf.local_models_path}
        model_info = self.fn_get_model_info(config.get("LOCAL_FILE_MODELS"), yolo_model)
        if model_info is None:
            QtWidgets.QMessageBox.critical(self, "Model not found",
                                        f"Model '{yolo_model}' not found in local models file.")
            return

        print(f"Using model info: {model_info}")

        # --- Build XML request (existing helper)
        user_id = 2443423 + self.req_counter
        self.req_counter += 1
        try:
            xml_data = self.fn_create_xml_param(
                user_id, ra_deg, dec_deg, int(h), int(w),
                self.image_path, yolo_model, quantization,
                model_info.get("ModelName") or model_info.get("Name") or yolo_model
            )
        except Exception as e:
            self.log(f"XML build failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Request error", f"Failed to build XML:\n{e}")
            return

        # --- Send → Poll → Download (existing helpers)
        server_url = self.server_conf.url
        self.log(
            f"Sending request: model={yolo_model}, quant={quantization}, norm={normalization}, "
            f"RA={ra_deg:.6f}, Dec={dec_deg:.6f}, w={int(w)}, h={int(h)}"
        )

        self.btn_identify.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            process_id = self.fn_send_xml_fits_to_server(server_url, xml_data)
            if process_id is None:
                raise RuntimeError("send_xml_fits_to_server returned None")

            self.log(f"Request sent, process id: {process_id}")
            self.status.showMessage(f"Polling job {process_id}...")

            if self.fn_poll_for_completion(server_url, process_id):
                self.log(f"Job {process_id} completed. Starting threaded download...")

                self.thread = QThread()
                self.worker = DownloadWorker(server_url, process_id, os.path.join(os.getcwd(), "results"))
                self.worker.moveToThread(self.thread)

                self.thread.started.connect(self.worker.run)
                self.worker.finished.connect(self._on_download_success)
                self.worker.failed.connect(self._on_download_failure)

                self.worker.finished.connect(self.thread.quit)
                self.worker.finished.connect(self.worker.deleteLater)
                self.thread.finished.connect(self.thread.deleteLater)

                self.thread.start()

        except requests.ConnectionError as e:
            self.log(f"Network error: {e}")
            QtWidgets.QMessageBox.critical(self, "Network error", str(e))
        except requests.Timeout as e:
            self.log(f"Timeout: {e}")
            QtWidgets.QMessageBox.critical(self, "Timeout", str(e))
        except Exception as e:
            self.log(f"Unexpected error: {e}")
            QtWidgets.QMessageBox.critical(self, "Unexpected error", str(e))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.btn_identify.setEnabled(True)
            self.status.showMessage("Ready.")

# ---
    def _bind_backend_helpers(self) -> None:
        """Bind backend helpers from the project's `src` package.

        It ensures that the parent directory containing `src/` is on sys.path,
        then imports the exact functions you listed:
        - from src.core.xml_utils import create_xml_param
        - from src.core.file_transfer import send_xml_fits_to_server
        - from src.services.server_comm import poll_for_completion, download_result
        - from src.utils.cianna_xml_updater import update_cianna_models, get_model_info
        - from src.utils.ssh_tunnel import create_ssh_tunnel
        - from src.utils.fits_utils import get_image_dim
        """
        # 1) Make sure the parent folder that contains "src" is importable
        candidates = [
            os.path.abspath(os.path.join(SCRIPT_DIR, "..")),              # e.g. CODE/client
            os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")),        # e.g. CODE
            os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")),  # repo root
        ]
        # Optional override via env var if your tree is atypical
        env_root = os.environ.get("CIANNA_SRC_ROOT")
        if env_root:
            candidates.insert(0, os.path.abspath(env_root))

        added_path = None
        for base in candidates:
            if os.path.isdir(os.path.join(base, "src")):
                if base not in sys.path:
                    sys.path.insert(0, base)
                    added_path = base
                break

        # 2) Try the direct imports from your package layout
        try:
            from src.core.xml_utils import create_xml_param
            from src.core.file_transfer import send_xml_fits_to_server
            from src.services.server_comm import poll_for_completion, download_result
            from src.utils.cianna_xml_updater import update_cianna_models, get_model_info
            from src.utils.ssh_tunnel import create_ssh_tunnel
            from src.utils.fits_utils import get_image_dim
        except Exception as e:
            if added_path:
                self.log(f"Backend import failed even after adding to sys.path: {added_path}")
            self.log(f"Import error for backend helpers: {e}")
            # Null out all function refs so the UI can disable features gracefully
            self.fn_get_model_info = None
            self.fn_get_image_dim = None
            self.fn_create_xml_param = None
            self.fn_send_xml_fits_to_server = None
            self.fn_poll_for_completion = None
            self.fn_download_result = None
            self.fn_create_ssh_tunnel = None
            self.fn_update_cianna_models = None
            return

        # 3) Bind to instance attributes used elsewhere in the GUI
        self.fn_get_model_info = get_model_info
        self.fn_get_image_dim = get_image_dim
        self.fn_create_xml_param = create_xml_param
        self.fn_send_xml_fits_to_server = send_xml_fits_to_server
        self.fn_poll_for_completion = poll_for_completion
        self.fn_download_result = download_result
        self.fn_create_ssh_tunnel = create_ssh_tunnel
        self.fn_update_cianna_models = update_cianna_models

        msg = "Backend helpers imported from 'src' package"
        if added_path:
            msg += f" (sys.path += {added_path})"
        self.log(msg)



# --------------------
# Entrypoint
# --------------------

def main() -> None:
    """Qt application entrypoint."""
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("CIANNA OTF")

    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
