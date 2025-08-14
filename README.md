# ARIana (Python/Tk)

*A cross-platform simulator for teaching air-enema reduction of pediatric intussusception.*

> **Training use only.** This simulator supplements—**does not replace**—supervised, real-patient experience under a trained pediatric radiologist.

---



## Table of Contents
- [What’s New](#whats-new)
- [Requirements](#requirements)
- [Quick Start (Run From Source)](#quick-start-run-from-source)
- [Folder Layout](#folder-layout)
- [Case Metadata JSON Schema](#case-metadata-json-schema)
- [Workflow](#workflow)
  - [1) Disclaimer](#1-disclaimer)
  - [2) Case Selection](#2-case-selection)
  - [3) Pre-Operation](#3-pre-operation)
  - [4) Simulation](#4-simulation)
  - [5) Results](#5-results)
- [Hardware Setup (Optional Manometer)](#hardware-setup-optional-manometer)
- [Keyboard & Interaction](#keyboard--interaction)
- [Safety Timers & Outcomes](#safety-timers--outcomes)
- [Case Engine Details](#case-engine-details)
- [Results & Plotting Details](#results--plotting-details)
- [Building From Source (macOS • Windows • Linux)](#building-from-source-macos--windows--linux)
  - [macOS — Nuitka `.app` + DMG](#macos--nuitka-app--dmg)
  - [Windows — Nuitka (or PyInstaller)](#windows--nuitka-or-pyinstaller)
  - [Linux — Run or Build](#linux--run-or-build)
- [Common Gotchas](#common-gotchas)
- [Troubleshooting](#troubleshooting)
- [Attribution & Provenance](#attribution--provenance)
- [License](#license)

---



## What’s New

- Cross-platform desktop app (macOS, Windows, Linux) using **Tkinter/ttk**.
- Optional **hardware manometer** via USB-serial (CP2102) with live pressure in **mmHg**.
- **Virtual Pressure** mode (slider + numeric entry) when hardware isn’t connected.
- Tabbed workflow: *Disclaimer → Case Selection → Pre-Operation → Simulation → Results*.
- **Configurable case engine** via JSON: stages, risk curves, safety limits.
- Results view with **pressure-vs-time plot**, stage progression, and post-procedure images.
- Built-in safety timing: **3-minute warning**, **5-minute hard stop**, and per-case **fluoroscopy cap**.

---



## Requirements

- **Python:** 3.9–3.12 recommended
- **GUI:** Tk/Tkinter (bundled with python.org installers on macOS & Windows; `python3-tk` on many Linux distros)
- **Packages (runtime):**
  - `numpy`, `scipy`, `matplotlib`, `Pillow`, `pyserial`
- **Packages (for building binaries):**
  - `nuitka` (recommended), `ordered-set`, `zstandard`
  - *(Windows alternative)* `pyinstaller`

> If you will interface with hardware, the manometer should enumerate as **Silicon Labs CP2102** (USB–UART).

---



## Quick Start (Run From Source)

To get started quickly, follow these steps to run ARIana from its source code:

```bash
# 1) Clone & enter the repo
git clone <your-repo-url>.git
cd <your-repo>

# 2) Create & activate a virtual environment
python3 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 3) Install dependencies
pip install -U pip
pip install numpy scipy matplotlib Pillow pyserial

# 4) Run the app
python ARIana.py
# If your entry file is different (e.g., intussusception_trainer.py), run that instead.
```

---



## Folder Layout

The project's folder structure is organized as follows:

```
repo/
├─ ARIana.py                      # Main application entry point (or intussusception_trainer.py)
├─ Patients/
│  └─ <CaseID>/
│     ├─ <CaseID>_metadata.json   # Case parameters (see schema below)
│     └─ Images/
│        ├─ Preprocedure/         # Pre-procedure images (e.g., preprocedure_1.png)
│        ├─ Simulation/           # Simulation images (e.g., simulation_1.png)
│        └─ Postprocedure/        # Post-procedure images (e.g., postprocedure_1.png)
├─ ARIana_logo.png                # Application icon/branding
└─ README.md
```

**Image Naming Rules:**

- Use the exact names shown above; numbering for images should be contiguous from 1 (e.g., `preprocedure_1.png`, `preprocedure_2.png`).
- Any reasonable clinical resolution is supported; the UI scales images to the window size.

---



## Case Metadata JSON Schema

Each case in ARIana includes a single JSON file named `<CaseID>_metadata.json`. This file defines the parameters for each simulation case.

```json
{
  "parameters": {
    "name": "Infant w/ ileocolic intussusception",
    "teaser": "Intermittent pain; currant jelly stool",
    "clinical_descrip": "Vitals ... brief history for pre-op dialog.",

    "num_stages": 5,
    "coeff": [100, 100, 100, 100, 100],

    "perf_data": [[0,0],[60,1],[90,3],[120,6],[150,10],[180,15]],
    "ret_data":  [[0,0],[60,2],[90,5],[120,10],[150,20],[180,30]],

    "max_fluoro_time": 120,
    "dontstart": 0
  }
}
```

### Field Reference

| Field Name | Type | Description |
|---|---|---|
| `name` | String | Shown in Case Selection & Pre-Op. |
| `teaser` | String | Shown in Case Selection & Pre-Op. |
| `clinical_descrip` | String | Vitals and brief history for pre-op dialog. |
| `num_stages` | Integer | Number of reduction stages (default: 5). |
| `coeff` | List[Integer] | Per-stage success multipliers (length = `num_stages`, default: 100s). |
| `perf_data` | List[[mmHg, %]] | Pressure-to-perforation probability curve. |
| `ret_data` | List[[mmHg, %]] | Pressure-to-retrogression probability curve. |
| `max_fluoro_time` | Integer | Fluoroscopy exposure cap in seconds (default: 120). |
| `dontstart` | Integer (0/1) | If 1, 'Start Intussusception' triggers a contraindication end. |

**Tuning Tips:**

- Keep `perf_data` monotonically increasing with pressure.
- Tune `ret_data` around mid-pressures to model “back-and-forth” movement.

---



## Workflow

ARIana guides users through a structured workflow, simulating the air-enema reduction procedure. The workflow is divided into several tabs:

### 1) Disclaimer

Users must acknowledge the training-only use of the simulator before proceeding.

### 2) Case Selection

The 'Refresh' button scans the `Patients/` folder for available cases. Select a case to view its teaser and clinical description, then click 'Select Case' to load it.

### 3) Pre-Operation

This tab prepares for the simulation. Key actions and information include:

- **Start Intussusception:** Proceeds to the 'Simulation' tab. If `dontstart=1` in the case metadata, this triggers a contraindication and ends the case with an explanation.
- **Call for Surgery:** Immediately ends the case, simulating a decision to send the patient to surgery.
- **Check Vitals and Medical History:** Displays the `clinical_descrip` from the case metadata in a dialog.

**Pre-Op Checklist (verbatim from app):**

- Pediatric surgery has been consulted and is aware of the patient.
- Abdominal radiographs (AP and cross-table lateral or decubitus) show no free air.
- No peritoneal signs are present.
- Patient is hemodynamically stable.
- IV access has been secured.
- Parents/guardians have consented to the procedure (preferable).
- Large-bore angiocath available at bedside.
- Provider in the room primarily responsible for the patient (nurse or doctor).
- Patient’s vital signs are being monitored.
- What catheter will be used?
- Will you sedate the patient?

### 4) Simulation

This is the core simulation phase where pressure is applied and monitored.

**Pressure Input:**

- **Hardware manometer (default):** Auto-detects CP2102 USB–UART devices. Incoming packets are converted to mmHg for live pressure readings.
- **Virtual Pressure:** Enable this mode to control pressure using a slider and numeric input field when hardware is not connected.

**Controls:**

- **Take Fluoroscopy Image (space bar):** Captures the next simulation frame and adds to the total fluoroscopy time. A red indicator is active while this is in use.
- **Call for Surgery:** Immediately ends the simulation.
- **End Simulation:** Ends the simulation and moves to the 'Results' tab.

**Live Status:**

The simulation displays real-time status indicators:

- **Stage:** Current reduction stage (e.g., 1…`num_stages`). Completion is marked as 'Complete'.
- **Sim Time:** Wall-clock time in seconds. Safety rules include a 3-minute warning at 180 seconds and a hard stop at 300 seconds.
- **Pressure:** Current pressure in mmHg (from hardware or virtual input).
- **Fluoro Time:** Cumulative fluoroscopy time in seconds. Exceeding `max_fluoro_time` ends the case.
- **Outcome:** The outcome of the last tick: `Success`, `Retrogress`, `Stuck`, `Perforated`, or `Complete`.
- **Visibility toggles:** Instructors can hide or show any status row (hidden rows display “Not Shown”).

### 5) Results

After the simulation, the 'Results' tab provides a summary and analysis:

- **Summary:** Displays the final Outcome, Total Simulation Time, and Total Fluoroscopy Time.
- **Plot:** A pressure (mmHg) over time (s) plot, with stage transitions overlaid.
- **Post-procedure images:** A scrollable viewer for post-procedure images. Click an image to open a zoomable window.

---



## Hardware Setup (Optional Manometer)

For an enhanced simulation experience, ARIana supports an optional hardware manometer. To use it:

1.  Connect the manometer via USB.
2.  The app will auto-detect the CP2102 USB-UART device.
    -   **macOS:** Prefers `/dev/cu.usbserial-0001`.
    -   **Windows:** Uses the first detected CP2102 COM port.
3.  The status will show `Connected` after a successful handshake. If connection fails, use the Virtual Pressure mode.

### CP2102 Protocol (Used by the App)

-   **VID/PID:** `0x10C4 / 0xEA60` (Product: “CP2102 USB to UART Bridge Controller”)
-   **Handshake:** The app writes `55 AA 01` and expects any packet with header `AA 56`.
-   **Packet Format (10 bytes):**
    `AA 56 | u | s | r r | d d d d d`
    -   `u`: unit code
    -   `s`: status byte (bit2 is sign)
    -   `r r`: reserved
    -   `d d d d d`: ASCII digits for value (no decimal point)

### Unit → mmHg Conversion

| Unit Code | Unit | Conversion Factor to mmHg |
|---|---|---|
| `0x00` | bar | `× 750.062` |
| `0x01` | oz/in² | `× 3.23218` |
| `0x02` | psi | `× 51.7149` |
| `0x03` | inHg | `× 25.4` |
| `0x04` | mbar | `× 0.750062` |
| `0x05` | mmHg | `× 1.0` |
| `0x06` | kPa | `× 7.50062` |
| `0x07` | kg/cm² | `× 735.559` |
| `0x08` | inH₂O | `× 1.86832` |
| `0x09` | ftH₂O | `× 22.4199` |
| `0x0A` | cmH₂O | `× 0.735559` |

### Sampling

-   The serial reader polls approximately every 10 ms.
-   The UI samples pressure approximately every 50 ms.

**Troubleshooting:** If the device doesn’t enumerate, install the CP210x driver (Windows) or run `modprobe cp210x` (Linux).

---



## Keyboard & Interaction

-   **Spacebar:** Triggers `Take Fluoroscopy Image` only when the Simulation tab is active. It is suppressed in other tabs.
-   **Clicking Images:** Click any case image to open a separate, zoomable viewer for detailed examination.

---



## Safety Timers & Outcomes

ARIana incorporates several safety timers and outcome triggers to simulate realistic scenarios:

-   **3-minute warning:** At 180 seconds of total simulation time, a red banner appears, advising the user to release/rest pressure.
-   **Perforation → vitals crash:** If a `Perforated` outcome occurs and 180 seconds elapse without resolution, the outcome becomes `Perforated Vitals Crashed`, and the case ends.
-   **5-minute hard stop:** At 300 seconds of total simulation time, the case automatically ends with a `Time Limit 5 Min` outcome.
-   **Fluoroscopy limit:** The cumulative fluoroscopy time is capped by `max_fluoro_time` (default 120 seconds). Exceeding this limit ends the case with a radiation over-exposure outcome.

---



## Case Engine Details

### Images

-   `simulation_N.png` corresponds to stage `N`.
-   If the current stage is greater than `num_stages`, the last available image is reused.

### Outcome Selection (per tick)

Outcomes are determined by a probabilistic model:

1.  Interpolate perforation and retrogression probabilities from `perf_data` and `ret_data`.
2.  Compute a pressure-scaled success term (heuristic) using the formula:

    ```
    success ∝ (pressure / 180)^2 × coeff[current_stage]
    ```

3.  A random number between 0 and 100 is drawn, and the first satisfied outcome is selected in this order:
    -   `Perforated`
    -   `Retrogress` (if `stage > 1`)
    -   `Success` (advances stage)
    -   `Stuck`

### Tuning Tips

-   Increase `coeff` for a specific stage to make it easier to clear.
-   Reduce `coeff` to make a stage more stubborn.
-   Raise mid-range values in `ret_data` to model “back-and-forth” movement.
-   Ensure `perf_data` is strictly increasing with pressure.

---



## Results & Plotting Details

-   **Plot:** The results display a plot of Pressure (mmHg) versus Time (s), with stage transitions overlaid for visual analysis.
-   Duplicate time samples are dropped to ensure data integrity.
-   Pressure values can be interpolated and Gaussian-smoothed for improved readability of the plot.

---



## Building From Source (macOS • Windows • Linux)

If you only need to run the app, refer to the [Quick Start](#quick-start-run-from-source) section. The following steps are for users who wish to produce redistributable binaries of ARIana.

### macOS — Nuitka `.app` + DMG

#### Prerequisites

-   **Xcode Command Line Tools:** Install by running `xcode-select --install` in your terminal.
-   **Python:** Version 3.9–3.12 (installers from python.org usually bundle a compatible Tk).
-   **Python Packages:** Install `nuitka`, `ordered-set`, and `zstandard`:
    ```bash
    pip install nuitka ordered-set zstandard
    ```

#### Build Command

Use the following command to build the `.app` bundle and a DMG installer:

```bash
python3 -m nuitka \
  --standalone \
  --macos-create-app-bundle \
  --macos-app-name="ARIana" \
  --include-data-dir=Patients=Patients \
  --include-data-file=ARIana_logo.png=ARIana_logo.png \
  --include-module=tkinter \
  --include-module=PIL \
  --include-module=matplotlib \
  --include-module=numpy \
  --include-module=scipy \
  --include-module=serial \
  --enable-plugin=tk-inter \
  --output-dir=build_macos \
  --macos-app-icon=ARIana_logo.png \
  --macos-sign-identity="-" \
  ARIana.py && \
hdiutil create -volname "ARIana" -srcfolder build_macos/ARIana.app -ov -format UDZO ARIana.dmg
```

#### Notes

-   Replace `ARIana.py` with your actual entry file if it's different.
-   `--macos-sign-identity="-"` performs ad-hoc signing, which is suitable for local testing. For wider distribution, use a Developer ID and notarization.

### Windows — Nuitka (or PyInstaller)

#### Prerequisites

-   **Python:** Version 3.10–3.12 (64-bit), added to your system's PATH.
-   **Visual Studio Build Tools:** C++ toolset is required for Nuitka.
-   **Python Packages:** Install `nuitka` (recommended) or `pyinstaller` (for the alternative):
    ```bash
    pip install nuitka
    # or
    pip install pyinstaller
    ```

#### Nuitka (Recommended)

Use the following PowerShell command to build with Nuitka:

```powershell
python -m nuitka `
  --standalone `
  --assume-yes-for-downloads `
  --windows-console-mode=disable `
  --windows-product-name="ARIana" `
  --windows-file-description="ARIana Application" `
  --include-data-dir=Patients=Patients `
  --include-data-file=ARIana_logo.png=ARIana_logo.png `
  --include-module=tkinter `
  --include-module=PIL `
  --include-module=matplotlib `
  --include-module=numpy `
  --include-module=scipy `
  --include-module=serial `
  --enable-plugin=tk-inter `
  --output-dir=build_windows `
  ARIana.py
```

**Result:** The executable will be located at `build_windows\ARIana.dist\ARIana.exe`. Zip the entire `*.dist` folder to share the application.

**Optional Icon:** Use `--windows-icon-from-ico=ARIana.ico` to include a custom icon (convert your PNG to ICO format first).

#### PyInstaller (Alternative)

If you prefer PyInstaller, use these commands:

```powershell
pip install pyinstaller
pyinstaller --noconfirm --clean --windowed `
  --name "ARIana" `
  --add-data "Patients;Patients" `
  --add-data "ARIana_logo.png;." `
  --hidden-import tkinter `
  --hidden-import PIL `
  --hidden-import matplotlib `
  --hidden-import numpy `
  --hidden-import scipy `
  --hidden-import serial `
  ARIana.py
```

**Result:** The executable will be at `dist\ARIana\ARIana.exe`.

**Note on `--add-data`:** On Windows, `--add-data` uses a semicolon (e.g., `src;dest`). On macOS/Linux, it uses a colon (e.g., `src:dest`).

**Driver Note:** For the hardware manometer, ensure the CP210x USB-UART driver is installed.

### Linux — Run or Build

#### Run From Source (Recommended)

For a quick setup on Debian/Ubuntu-based systems:

```bash
# Debian/Ubuntu example:
sudo apt-get update && sudo apt-get install -y python3-tk python3-venv build-essential
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip numpy scipy matplotlib Pillow pyserial
python ARIana.py
```

#### Optional: Nuitka Standalone Build

To create a standalone executable for Linux:

```bash
pip install nuitka ordered-set zstandard
python -m nuitka \
  --standalone \
  --include-data-dir=Patients=Patients \
  --include-data-file=ARIana_logo.png=ARIana_logo.png \
  --include-module=tkinter \
  --include-module=PIL \
  --include-module=matplotlib \
  --include-module=numpy \
  --include-module=scipy \
  --include-module=serial \
  --enable-plugin=tk-inter \
  --output-dir=build_linux \
  ARIana.py
```

#### Serial Access (Linux)

To grant your user access to serial ports (necessary for the hardware manometer):

```bash
sudo usermod -aG dialout "$USER"   # Log out/in after this command for changes to take effect.
```

---



## Common Gotchas

Here are some common issues you might encounter and their solutions:

-   **Missing cases/images in build:** Ensure that `Patients/` and `ARIana_logo.png` are correctly included in your build process using the appropriate `--include-data-*` (Nuitka) or `--add-data` (PyInstaller) flags for your operating system.
-   **Matplotlib backend issues:** ARIana uses `TkAgg`. Verify that Tk is available on your system by running `python -m tkinter`. This should open a test window.
-   **High-DPI scaling (Windows):** If the UI appears blurry, enable `System (Enhanced)` DPI scaling in the app compatibility settings, or adjust your Windows display scaling.
-   **Anti-virus false positives (Windows):** When distributing, prefer zipping the entire `*.dist` folder (from `--standalone` builds) and consider signing your binaries to avoid anti-virus flags.

---



## Troubleshooting

If you encounter problems, consult the following troubleshooting tips:

-   **No cases listed:** Verify that `Patients/<CaseID>/<CaseID>_metadata.json` files exist and that images use the required naming conventions within their respective folders.
-   **Spacebar doesn’t advance images:** Ensure that the `Simulation` tab is active, the application window has focus, and you have not exceeded the `max_fluoro_time` limit.
-   **Pressure stuck at 0 / “Disconnected”:**
    -   Check the USB cable connection to the manometer.
    -   Close any other serial applications (e.g., Arduino IDE, terminal emulators) that might be using the COM port.
    -   Confirm that the CP2102 driver is correctly installed.
    -   As a workaround, use the `Virtual Pressure` mode to continue the simulation.
-   **Immediate end after Preprocedure:** This usually indicates that the case metadata has `dontstart: 1`, which triggers a contraindication scenario.
-   **Unexpected time endings:** The 3-minute warning is informational. The 5-minute limit is a hard stop by design, and exceeding the fluoroscopy limit also ends the case as intended.

---



## Attribution & Provenance

-   **Original MATLAB manual:** `ARI manual 201409.pdf` (2014).
-   **Original authors:** S.K.Soosman, G.E. Roper, A.S. Wexler, J.C. Li, R. Stein-Wexier, M.S. Fleisher
-   **This release:** A Python/Tk re-implementation building upon the 2014 MATLAB ARI/ARIana simulator and manual.

The clinical workflow language (e.g., Pre-Op Checklist, safety timers, fluoroscopy guidance) and several UI conventions are adapted from the original manual. All original credit belongs to the original authors.

---



## License

Choose a license (e.g., MIT / Apache-2.0 / BSD-3-Clause) and add it as `LICENSE` in the root of your repository. Update the header of this README with the chosen license if desired.


