import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import random
import threading
import time
import queue # Added for thread-safe communication
import serial # Added for serial communication
import serial.tools.list_ports #for port scanning
from PIL import Image, ImageTk
import numpy as np
import textwrap
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


# Manometer Constants (extracted from read_hd700.py)
UNIT_CONVERSIONS_TO_MMHG = {
    0x00: 750.062,  # 1 bar = 750.062 mmHg
    0x01: 3.23218,  # 1 oz/in^2 = 3.23218 mmHg (Corrected)
    0x02: 51.7149,  # 1 psi = 51.7149 mmHg
    0x03: 25.4,     # 1 inHg = 25.4 mmHg
    0x04: 0.750062, # 1 mbar = 0.750062 mmHg
    0x05: 1.0,      # 1 mmHg = 1 mmHg (no conversion needed)
    0x06: 7.50062,  # 1 kPa = 7.50062 mmHg
    0x07: 735.559,  # 1 kg/cm^2 = 735.559 mmHg
    0x08: 1.86832,  # 1 inH2O = 1.86832 mmHg
    0x09: 22.4199,  # 1 ftH2O = 22.4199 mmHg
    0x0a: 0.735559, # 1 cmH2O = 0.735559 mmHg
}

UNIT_NAMES = {
    0x00: "bar",
    0x01: "oz/in²",
    0x02: "psi",
    0x03: "inHg",
    0x04: "mbar",
    0x05: "mmHg",
    0x06: "kPa",
    0x07: "kg/cm²",
    0x08: "inH₂O",
    0x09: "ftH₂O",
    0x0a: "cmH₂O",
}

HANDSHAKE_CMD = b'\x55\xaa\x01'

def parse_manometer_packet(packet):
    """Decode a 10-byte pressure packet and convert to mmHg without scaling factors."""
    if len(packet) != 10 or packet[0] != 0xAA or packet[1] != 0x56:
        return None
    
    unit_code = packet[2]
    status_byte = packet[3]
    sign_bit = (status_byte >> 2) & 0x01
    
    try:
        raw_value = float(packet[5:10].decode('ascii'))
        
        conversion_to_mmhg = UNIT_CONVERSIONS_TO_MMHG.get(unit_code, 1.0)
        converted_value = raw_value * conversion_to_mmhg
        
        if sign_bit:
            converted_value = -converted_value
            
    except (ValueError, UnicodeDecodeError):
        return None
        
    return unit_code, converted_value

class ManometerThread(threading.Thread):
    
    def __init__(self, data_queue, baudrate=9600):
        super().__init__(daemon=True)
        self.data_queue = data_queue
        self.baudrate = baudrate
        self.ser = None
        self._running = True

    def find_cp2102_port(self):
        target_vid = 0x10C4
        target_pid = 0xEA60
        target_product = "CP2102 USB to UART Bridge Controller"

        matching_ports = []

        for port in serial.tools.list_ports.comports():
            if (
                port.vid == target_vid and
                port.pid == target_pid and
                port.product == target_product
            ):
                matching_ports.append(port.device)

        # Prefer /dev/cu.usbserial-0001 if available
        for dev in matching_ports:
            if "usbserial-0001" in dev:
                return dev

        # Otherwise just return the first match
        return matching_ports[0] if matching_ports else None

    def run(self):
        while self._running:
            # Find port on each connection attempt
            port = self.find_cp2102_port()
            
            if not port:
                self.data_queue.put(("status", "Disconnected"))
                time.sleep(0.1)  # Very fast retry when no port found
                continue
                
            if self.ser is None or not self.ser.is_open:
                self.data_queue.put(("status", "Connecting..."))
                self.ser = self._connect_to_device(port)
                if self.ser:
                    self.data_queue.put(("status", "Connected"))
                    self._stream_data()
                else:
                    self.data_queue.put(("status", "Disconnected"))
                    time.sleep(0.1) # Very fast retry
            else:
                self.data_queue.put(("status", "Connected"))
                self._stream_data()

    def _connect_to_device(self, port):
        try:
            ser = serial.Serial(port, self.baudrate, timeout=0.1)  # Very short timeout
            time.sleep(0.05)  # Minimal settling time
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            # Quick handshake
            ser.write(HANDSHAKE_CMD)
            time.sleep(0.05)  # Minimal handshake wait
            response = ser.read(32)
            if b'\xAA\x56' in response:
                return ser
            else:
                ser.close()
                return None

        except (serial.SerialException, OSError):
            if ser and ser.is_open:
                ser.close()
            return None
        except Exception:
            if ser and ser.is_open:
                ser.close()
            return None

    def _stream_data(self):
        buffer = bytearray()
        try:
            while self._running and self.ser and self.ser.is_open:
                if self.ser.in_waiting > 0:
                    buffer.extend(self.ser.read(self.ser.in_waiting))
                    
                    while len(buffer) >= 10:
                        if buffer[0] != 0xAA or buffer[1] != 0x56:
                            buffer.pop(0)
                            continue
                        
                        packet = buffer[:10]
                        buffer = buffer[10:]
                        
                        result = parse_manometer_packet(packet)
                        if result:
                            unit_code, pressure_mmhg = result
                            original_unit_name = UNIT_NAMES.get(unit_code, "Unknown")
                            self.data_queue.put(("pressure", round(pressure_mmhg), original_unit_name, unit_code))
                time.sleep(0.01)
        except (serial.SerialException, OSError) as e:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = None
            self.data_queue.put(("status", "Disconnected"))
        except Exception as e:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = None
            self.data_queue.put(("status", "Disconnected"))

    def stop(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
class CaseLoader:
    """Handles loading case data from JSON files and managing images"""
    
    def __init__(self, base_path="Patients"):
        self.base_path = base_path
    
    def load_case_list(self):
        """Load the list of available cases from the directory structure"""
        cases = []
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)

        for item in os.listdir(self.base_path):
            if item.lower() == "placeholder":
                continue

            case_folder_path = os.path.join(self.base_path, item)
            if os.path.isdir(case_folder_path):
                metadata_file_path = None
                for filename in os.listdir(case_folder_path):
                    if filename.endswith("_metadata.json"):
                        metadata_file_path = os.path.join(case_folder_path, filename)
                        break
                
                if metadata_file_path:
                    try:
                        with open(metadata_file_path, "r") as f:
                            metadata = json.load(f)
                        cases.append({
                            "id": item,
                            "name": metadata.get("parameters", {}).get("name", item),
                            "description": metadata.get("parameters", {}).get("teaser", "Patient case"),
                            "clinical_descrip": metadata.get("parameters", {}).get("clinical_descrip", "No clinical history available.")
                        })
                    except Exception as e:
                        print(f"Error loading case {item}: {e}")
        
        # Alphabetize patient names
        cases.sort(key=lambda x: x["name"].lower())
        return cases
    
    def load_case(self, case_id):
        """Load a specific case by ID"""
        metadata_file = None
        case_folder_path = os.path.join(self.base_path, case_id)
        if os.path.exists(case_folder_path):
            for filename in os.listdir(case_folder_path):
                if filename.endswith("_metadata.json"):
                    metadata_file = os.path.join(case_folder_path, filename)
                    break

        if not metadata_file:
            raise FileNotFoundError(f"Case metadata file not found for case_id: {case_id}")
        
        with open(metadata_file, "r") as f:
            case_data = json.load(f)
        
        case_data["images"] = {
            "preprocedure": [], "simulation": [], "postprocedure": []
        }
        patient_image_base_path = os.path.join(self.base_path, case_id, "Images")

        for proc_type in ["Preprocedure", "Simulation", "Postprocedure"]:
            proc_path = os.path.join(patient_image_base_path, proc_type)
            if os.path.exists(proc_path):
                img_key = proc_type.lower()
                i = 1
                while True:
                    img_name = f"{proc_type.lower()}_{i}.png"
                    img_path = os.path.join(proc_path, img_name)
                    if os.path.exists(img_path):
                        case_data["images"][img_key].append(img_path)
                        i += 1
                    else:
                        break
        return case_data

class IntussusceptionSimulator:
    """Core simulation engine"""
    
    def __init__(self):
        self.case_data = None
        self.current_stage = 1
        self.pressure_history = []
        self.stage_history = []
        self.time_history = []
        self.sim_time = 0
        self.fluoro_time = 0
        self.last_fluoro_time = 0
        self.is_running = False

        self.simulation_thread = None
        self.callbacks = []
        self.last_outcome = "Ready"
        self.is_perforated = False
        self.perforation_time = None
        self.perforation_timer_active = False
        self.warned_at_3min = False # Initialize warned_at_3min here
    
    def load_case(self, case_data):
        self.case_data = case_data
        self.current_stage = 1
        self.pressure_history = [0]
        self.stage_history = [1]
        self.time_history = [0]
        self.sim_time = 0
        self.fluoro_time = 0
        self.last_fluoro_time = 0
        self.last_outcome = "Ready"
        self.is_perforated = False
        self.perforation_time = None
        self.perforation_timer_active = False
        self.warned_at_3min = False # Also reset when loading a new case
        return True
    
    def start_simulation(self, callback=None):
        if self.is_running: return
        self.is_running = True
        if callback and callback not in self.callbacks:
            self.callbacks.append(callback)
        
        def simulation_loop():
            while self.is_running:
                time.sleep(0.5)
                self.sim_time += 0.5
                state = {
                    "current_stage": self.current_stage, 
                    "sim_time": self.sim_time,
                    "fluoro_time": self.fluoro_time, 
                    "pressure": self.pressure_history[-1] if self.pressure_history else 0,
                    "outcome": self.last_outcome 
                }
                for cb in self.callbacks: cb(state)

                # Perforation timer logic
                if self.is_perforated and self.perforation_timer_active:
                    elapsed_perforation_time = self.sim_time - self.perforation_time
                    if elapsed_perforation_time >= 180: # 3 minutes
                        self.perforation_timer_active = False # Deactivate timer
                        # Trigger callback for vitals crashed scenario
                        for cb in self.callbacks: cb({
                            "current_stage": self.current_stage,
                            "sim_time": self.sim_time,
                            "fluoro_time": self.fluoro_time,
                            "pressure": self.pressure_history[-1] if self.pressure_history else 0,
                            "outcome": "Perforated Vitals Crashed"
                        })
                
                # Simulation time warnings and limits
                if self.sim_time >= 180 and self.sim_time < 300 and not self.warned_at_3min:
                    self.warned_at_3min = True
                    for cb in self.callbacks:
                        cb({"type": "warning", "message": "3_min_warning"})
                elif self.sim_time >= 300: # 5 minutes
                    for cb in self.callbacks: cb({
                        "current_stage": self.current_stage,
                        "sim_time": self.sim_time,
                        "fluoro_time": self.fluoro_time,
                        "pressure": self.pressure_history[-1] if self.pressure_history else 0,
                        "outcome": "Time Limit 5 Min"
                    })        
        self.simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
        self.simulation_thread.start()
    
    def stop_simulation(self):
        self.is_running = False
  
    def _interpolate_probability(self, data, pressure):
        """Helper function to interpolate probability from data table."""
        if not data or len(data) < 2: return 0
        
        x_coords, y_coords = zip(*data)
        
        if pressure <= x_coords[0]: return y_coords[0]
        if pressure >= x_coords[-1]: return y_coords[-1]
        
        return np.interp(pressure, x_coords, y_coords)

    def process_pressure_reading(self, pressure):
        #keep perforation permenant
        if self.is_perforated:
            self.pressure_history.append(pressure)
            self.time_history.append(self.sim_time)
            self.stage_history.append(self.current_stage)
            self.last_outcome = "Perforated"
            return {
                "current_stage": self.current_stage,
                "outcome": "Perforated",
                "pressure": pressure,
                "sim_time": self.sim_time,
                "fluoro_time": self.fluoro_time
            }

        # Always append pressure and time history for every reading
        self.pressure_history.append(pressure)
        self.time_history.append(self.sim_time)
        
        params = self.case_data.get("parameters", {})
        num_stages = params.get("num_stages", 5)
        perf_data = params.get("perf_data", [[0, 0], [180, 10]])
        ret_data = params.get("ret_data", [[0, 0], [180, 30]])
        coeff = params.get("coeff", [100] * num_stages)

        perf_prob = self._interpolate_probability(perf_data, pressure)
        ret_prob = self._interpolate_probability(ret_data, pressure)
        
        stage_coeff = coeff[self.current_stage - 1] if 0 < self.current_stage <= len(coeff) else 100
        success_prob = ((pressure / 180) ** 2) * stage_coeff

        random_value = random.random() * 100
        outcome = "Stuck"
        
        if random_value < perf_prob:
            outcome = "Perforated"
            self.is_perforated = True
            self.current_stage = num_stages + 1  # Set to a stage beyond normal range
            self.last_outcome = outcome
            if not self.perforation_timer_active:
                self.perforation_time = self.sim_time
                self.perforation_timer_active = True
            self.stage_history.append(self.current_stage)
            return {
                "current_stage": self.current_stage, "outcome": outcome, "pressure": pressure,
                "sim_time": self.sim_time, "fluoro_time": self.fluoro_time
            }
        elif not self.is_perforated:
            if random_value < (perf_prob + ret_prob):
                if self.current_stage > 1:
                    outcome = "Retrogress"
                    self.current_stage -= 1
            elif random_value < (perf_prob + ret_prob + success_prob):
                if self.current_stage < num_stages:
                    outcome = "Success"
                    self.current_stage += 1
        
        if self.current_stage == num_stages and not self.is_perforated:
            outcome = "Complete"
        
        self.last_outcome = outcome
        
        # THIS IS THE CRUCIAL CHANGE: Always append current_stage to stage_history after all logic, ensuring sync
        self.stage_history.append(self.current_stage)
        
        return {
            "current_stage": self.current_stage, "outcome": outcome, "pressure": pressure,
            "sim_time": self.sim_time, "fluoro_time": self.fluoro_time
        }
    
    def take_fluoro_image(self):
        if not self.is_running:
            return {"result": "not_running"}
        #default max fluoro time is 120s but that can be changed
        max_fluoro_time = self.case_data.get("parameters", {}).get("max_fluoro_time", 120)
        if self.fluoro_time >= max_fluoro_time:
            return {"result": "radiation_overdose"}

        # Increment fluoro time every second
        if self.sim_time-self.last_fluoro_time >=1:
            self.fluoro_time += 1
            self.last_fluoro_time = self.sim_time

        image_path = self.get_image_for_stage(self.current_stage)
        return {"result": "success", "image_path": image_path}

    def get_image_for_stage(self, stage, image_type="simulation"):
        if not self.case_data: return None
        images_list = self.case_data["images"].get(image_type, [])
        if not images_list: return None
        
        num_stages = self.case_data.get("parameters", {}).get("num_stages", 5)
        if image_type == "simulation" and stage > num_stages:
            return images_list[-1]
        if 1 <= stage <= len(images_list):
            return images_list[stage - 1]
        return None
    
    def get_performance_data(self):
        if not self.case_data: return None
        return {
            "pressure_history": self.pressure_history, "stage_history": self.stage_history,
            "time_history": self.time_history, "ending_stage": self.current_stage,
            "sim_time": self.sim_time, "fluoro_time": self.fluoro_time,
            "successful": self.current_stage == self.case_data.get("parameters", {}).get("num_stages", 5)
        }

class PreOpImageScroller(ttk.Frame):
    """Scrollable image viewer for pre-operation images with click-to-zoom."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.image_paths = []
        self.current_index = 0
        self.photo_image = None

        # Layout: row 0 = image (expands), row 1 = nav (fixed)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Image display
        self.image_label = ttk.Label(self, text="No Images Available", anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew", pady=5)

        # Click-to-zoom
        self.image_label.bind("<Button-1>", self.open_zoom_viewer)
        self.image_label.configure(cursor="")  # becomes "hand2" when an image is shown

        # Navigation bar (bottom)
        self.nav_frame = ttk.Frame(self)
        self.nav_frame.grid(row=1, column=0, sticky="ew", pady=5)

        self.nav_frame.grid_columnconfigure(0, weight=1)  # left spacer
        self.nav_frame.grid_columnconfigure(4, weight=1)  # right spacer

        self.prev_button = ttk.Button(self.nav_frame, text="◀", command=self.prev_image, width=3)
        self.prev_button.grid(row=0, column=1, padx=2)

        self.counter_label = ttk.Label(self.nav_frame, text="0/0", anchor="center", width=8)
        self.counter_label.grid(row=0, column=2, padx=5)

        self.next_button = ttk.Button(self.nav_frame, text="▶", command=self.next_image, width=3)
        self.next_button.grid(row=0, column=3, padx=2)

        # Initially hide nav if not needed
        self.update_navigation_visibility()


    def set_images(self, image_paths):
        self.image_paths = list(image_paths or [])
        self.current_index = 0
        self.update_navigation_visibility()
        if self.image_paths:
            self.display_image()
        else:
            self.image_label.config(image="", text="No Images Available", cursor="")

    #  Navigation 
    def prev_image(self):
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            self.display_image()

    def next_image(self):
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.display_image()

    #  Helpers 
    def update_navigation_visibility(self):
        """Show/hide navigation controls based on number of images."""
        if len(self.image_paths) <= 1:
            self.nav_frame.grid_remove()
        else:
            self.nav_frame.grid()
        self.update_counter()

    def update_counter(self):
        if self.image_paths:
            self.counter_label.config(text=f"{self.current_index + 1}/{len(self.image_paths)}")
        else:
            self.counter_label.config(text="0/0")

    def open_zoom_viewer(self, event=None):
        """Open the current image in the ZoomableImageViewer if available."""
        if not self.image_paths:
            return
        idx = max(0, min(self.current_index, len(self.image_paths) - 1))
        img_path = self.image_paths[idx]
        if os.path.exists(img_path):
            # Assumes your ZoomableImageViewer class is defined elsewhere in the file.
            ZoomableImageViewer(self, img_path)

    def display_image(self):
        if not self.image_paths:
            self.image_label.config(image="", text="No Images Available", cursor="")
            return

        path = self.image_paths[self.current_index]
        self.update_counter()

        if os.path.exists(path):
            try:
                img = Image.open(path)

                # Size to fit the label’s current size
                container = self.image_label
                container.update_idletasks()
                panel_w = container.winfo_width()
                panel_h = container.winfo_height()

                # Provide a sane default if not yet laid out
                if panel_w < 10 or panel_h < 10:
                    panel_w, panel_h = 800, 600

                img.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)

                self.photo_image = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.photo_image, text="", cursor="hand2")
            except Exception as e:
                print(f"Error displaying image {path}: {e}")
                self.image_label.config(image="", text=f"Image not found:\n{os.path.basename(path)}", cursor="")
        else:
            self.image_label.config(image="", text="No Image Available", cursor="")


class ImageScroller(ttk.Frame):
    """A frame with a label to show an image and buttons to scroll through a list of images."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.image_paths = []
        self.current_index = 0
        self.photo_image = None

        self.grid_rowconfigure(1, weight=1) # Row 1 (image) expands
        self.grid_columnconfigure(0, weight=1)

        # Navigation frame at the top (Row 0)
        self.nav_frame = ttk.Frame(self)
        self.nav_frame.grid(row=0, column=0, sticky="ew", pady=5)

        # Center the navigation buttons
        self.nav_frame.grid_columnconfigure(1, weight=1)

        # Previous button
        self.prev_button = ttk.Button(self.nav_frame, text="◀", command=self.prev_image, width=3)
        self.prev_button.grid(row=0, column=0, padx=5)

        # Counter label in the center
        self.counter_label = ttk.Label(self.nav_frame, text="0/0", anchor="center")
        self.counter_label.grid(row=0, column=1, sticky="ew")

        # Next button
        self.next_button = ttk.Button(self.nav_frame, text="▶", command=self.next_image, width=3)
        self.next_button.grid(row=0, column=2, padx=5)

        # Image display (Row 1)
        self.image_label = ttk.Label(self, text="No Images Available", anchor="center")
        self.image_label.grid(row=1, column=0, sticky="nsew", pady=5)
        self.image_label.bind("<Button-1>", self.open_zoom_viewer)

        # Initially hide navigation controls
        self.update_navigation_visibility()

    def set_images(self, image_paths):
        self.image_paths = image_paths
        self.current_index = 0
        self.update_navigation_visibility()
        if self.image_paths:
            self.display_image()
        else:
            self.image_label.config(image="", text="No Images Available")

    def open_zoom_viewer(self, event=None):
        if self.image_paths:
            img_path = self.image_paths[self.current_index]
            if os.path.exists(img_path):
                ZoomableImageViewer(self, img_path)

    def update_navigation_visibility(self):
        """Show/hide navigation controls based on number of images"""
        if len(self.image_paths) <= 1:
            self.nav_frame.grid_remove()
        else:
            self.nav_frame.grid()
        self.update_counter()

    def update_counter(self):
        """Update the counter label"""
        if self.image_paths:
            self.counter_label.config(text=f"{self.current_index + 1}/{len(self.image_paths)}")
        else:
            self.counter_label.config(text="0/0")

    def prev_image(self):
        """Navigate to previous image"""
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            self.display_image()

    def next_image(self):
        """Navigate to next image"""
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.display_image()

    def display_image(self):
        if not self.image_paths:
            self.image_label.config(image="", text="No Images Available")
            return

        path = self.image_paths[self.current_index]
        self.update_counter()

        if os.path.exists(path):
            try:
                img = Image.open(path)
                # Use the label for sizing, as its container (the grid cell) is now properly managed.
                container = self.image_label
                container.update_idletasks()
                
                panel_w = container.winfo_width()
                panel_h = container.winfo_height()

                if panel_w < 50 or panel_h < 50:
                    panel_w, panel_h = 800, 600 

                img.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)

                self.photo_image = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.photo_image, text="")
            except Exception as e:
                print(f"Error displaying image {path}: {e}")
                self.image_label.config(image="", text=f"Image not found:\n{os.path.basename(path)}")
        else:
            self.image_label.config(image="", text="No Image Available")


class ZoomableImageViewer(tk.Toplevel):
    def __init__(self, master, image_path=None):
        super().__init__(master)
        self.title("Image Viewer")
        self.canvas = tk.Canvas(self, background="black", cursor="fleur")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_path = image_path
        self.scale = 1.0
        self.img = None
        self.original_img = None
        self.tk_img = None
        self.image_on_canvas = None

        # Match size of the main window
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = int(screen_w * 0.75)
        win_h = int(screen_h * 0.75)
        self.geometry(f"{win_w}x{win_h}")

        self.update_idletasks()

        self.bind_events()
        self.after(100, self.init_display)

    def bind_events(self):
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)         # Windows / macOS
        self.canvas.bind("<Button-4>", self.on_mousewheel)           # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)           # Linux scroll down
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)

    def init_display(self):
        if self.image_path and os.path.exists(self.image_path):
            self.img = Image.open(self.image_path)
            self.original_img = self.img.copy()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            scale_w = canvas_w / self.img.width
            scale_h = canvas_h / self.img.height
            self.scale = min(scale_w, scale_h, 1.0)
            self.update_image() 
        else:
            # Show placeholder text if no image yet
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width()//2,
                self.canvas.winfo_height()//2,
                text="No image yet.\nTake a fluoroscopic image to display.",
                fill="white"
            )


    def on_mousewheel(self, event):
        factor = 1.1 if event.delta > 0 or getattr(event, 'num', 0) == 4 else 0.9
        self.scale *= factor
        self.update_image()

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def update_image(self):
        if not self.original_img:
            return
        new_size = (int(self.original_img.width * self.scale), int(self.original_img.height * self.scale))
        resized_img = self.original_img.resize(new_size, Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized_img)

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_w, img_h = resized_img.size
        x = max((canvas_w - img_w) // 2, 0)
        y = max((canvas_h - img_h) // 2, 0)

        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_img)
        else:
            self.canvas.coords(self.image_on_canvas, x, y)
            self.canvas.itemconfig(self.image_on_canvas, image=self.tk_img)

        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # Auto-recenter if image is smaller than canvas
        if img_w <= canvas_w:
            self.canvas.xview_moveto(0.0)
        if img_h <= canvas_h:
            self.canvas.yview_moveto(0.0)

class ARIanaApp:
    """Main application class using Tkinter GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.bind("<Configure>", self.on_window_resize)
        self.root.title("ARIana Intussusception Simulator")
        
        self.result_plot_images = []
        self.result_plot_index = 0

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.called_surgery_from_preop = False


        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        app_w = int(screen_w * 0.95)
        app_h = int(screen_h * 0.95)
        self.root.geometry(f"{app_w}x{app_h}")

        
        self.case_loader = CaseLoader()
        self.all_cases_data = []
        self.simulator = IntussusceptionSimulator()
        
        self.pressure_var = tk.DoubleVar(value=0)
        
        # Initialize status_labels here, before create_widgets()
        self.status_labels = {}
        self.status_label_last_values = {} # New: Store last known values
        self.warning_shown = False  # Add debouncing flag for 3-minute warning

        # Manometer integration
        self.manometer_queue = queue.Queue()
        self.manometer_thread = ManometerThread(self.manometer_queue)
        self.manometer_thread.start()
        self.manometer_pressure = 0 # Store latest manometer pressure

        # Call create_widgets first to ensure all frames are initialized
        self.create_widgets()
        self.show_disclaimer()

        # Start checking manometer queue periodically
        self.check_manometer_queue()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True,          padx=10, pady=10)
        
        self.disclaimer_frame = ttk.Frame(self.notebook)
        self.startup_frame = ttk.Frame(self.notebook)
        self.pre_operation_frame = ttk.Frame(self.notebook) # New frame
        self.simulation_frame = ttk.Frame(self.notebook)
        self.results_frame = ttk.Frame(self.notebook)
        
        # Pack the frames within the notebook before adding them
        self.disclaimer_frame.pack(fill=tk.BOTH, expand=True)
        self.startup_frame.pack(fill=tk.BOTH, expand=True)
        self.pre_operation_frame.pack(fill=tk.BOTH, expand=True) # Pack new frame
        self.simulation_frame.pack(fill=tk.BOTH, expand=True)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook.add(self.disclaimer_frame, text="Disclaimer")
        self.notebook.add(self.startup_frame, text="Case Selection")
        self.notebook.add(self.pre_operation_frame, text="Pre-Operation") # Add new tab
        self.notebook.add(self.simulation_frame, text="Simulation")
        self.notebook.add(self.results_frame, text="Results")
        
        # Tab-aware spacebar behavior: Simulation = capture; others = do nothing
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.create_disclaimer_screen()
        self.create_startup_screen()
        self.create_pre_operation_screen() # New call
        self.create_simulation_screen()
        self.create_results_screen()

        # Apply initial spacebar behavior for the initially selected tab
        self._on_tab_changed()

        # Prevent spacebar from "clicking" any buttons anywhere
        self.root.bind_class("TButton", "<space>", lambda e: "break")
        self.root.bind_class("TCheckbutton", "<space>", lambda e: "break")
        self.root.bind_class("Button", "<space>", lambda e: "break")  # if you use tk.Button anywhere

    def _on_tab_changed(self, event=None):
        """Spacebar behavior: only active in Simulation."""
        try:
            self.root.unbind_all("<space>")
        except Exception:
            pass
        current_tab = self.notebook.tab(self.notebook.select(), "text")
        if current_tab == "Simulation":
            self.root.bind_all("<space>", self.take_fluoro_image)
        else:
            # Swallow the spacebar everywhere else (e.g., Pre-Operation)
            self.root.bind_all("<space>", lambda e: "break")

    def create_pre_operation_screen(self):
        main_frame = ttk.Frame(self.pre_operation_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for buttons + checklist (checklist BELOW the buttons)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        ttk.Button(left_panel, text="Start Intussusception",
                   command=self.start_simulation_from_pre_op).pack(pady=10, fill="x")
        ttk.Button(left_panel, text="Call for Surgery",
                   command=self.call_for_surgery).pack(pady=10, fill="x")
        ttk.Button(left_panel, text="Check Vitals and Medical History",
                   command=self.show_clinical_history_pre_op).pack(pady=10, fill="x")

        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", pady=(8, 6))

        ttk.Label(left_panel, text="Pre-Procedure Checklist",
                  font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 4))

        # Legible checklist under buttons
        self.preop_checklist_text = tk.Text(left_panel, width=44, height=18, wrap="word")
        self.preop_checklist_text.configure(font=("Arial", 18))
        checklist = (
            "1. Pediatric surgery has been consulted and is aware of the patient.\n"
            "2. Abdominal radiographs (AP and Cross-table lateral or decubitus) demonstrate no free air.\n"
            "3. No peritoneal signs are present.\n"
            "4. Patient is hemodynamically stable.\n"
            "5. IV access has been secured.\n"
            "6. Parents or guardians have consented to the procedure (preferable).\n"
            "7. Large-bore angiocath available at bedside.\n"
            "8. Provider in the room who will be primarily responsible for the patient (nurse or doctor).\n"
            "9. Patient's vital signs are being monitored.\n"
            "10. What catheter will be used?\n"
            "11. Will you sedate the patient?\n"
        )
        self.preop_checklist_text.insert("1.0", checklist)
        self.preop_checklist_text.configure(state="disabled")
        self.preop_checklist_text.pack(fill="both", expand=True)

        # Right panel for images (full height)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        ttk.Label(right_panel, text="Pre-procedure Images", font=("Arial", 12, "bold")).pack()
        self.pre_op_image_scroller = PreOpImageScroller(right_panel)
        self.pre_op_image_scroller.pack(fill="both", expand=True)

    def on_closing(self):
        """Handle window close event and ensure proper cleanup"""
        print("Shutting down manometer connection...")
        self.manometer_thread.stop()
        # Give the thread a moment to clean up
        if self.manometer_thread.is_alive():
            self.manometer_thread.join(timeout=1.0)
        self.root.destroy()

    def show_pre_operation(self):
        self.notebook.select(self.pre_operation_frame)
        if self.current_case:
            pre_images = self.current_case.get("images", {}).get("preprocedure", [])
            self.pre_op_image_scroller.set_images(pre_images)

    def start_simulation_from_pre_op(self):
        #move check here

        dontstart = self.current_case.get("parameters", {}).get("dontstart", 0)
        if dontstart == 1:
            messagebox.showinfo("Contraindication", "There was a contraindication. Intussusception is not recommended. The patient should be sent into surgery")
            self.end_simulation(outcome_override="Contraindication Was Not Recognized")
            return # Prevent simulation from starting
        self.show_simulation()
    
    def show_clinical_history_pre_op(self):
        if not getattr(self, "current_case", None):
            messagebox.showwarning("No Case Loaded", "Please select a patient case first.")
            return
        clinical_desc = self.current_case.get("parameters", {}).get("clinical_descrip", "No clinical history available.")
        wrapped_text = textwrap.fill(clinical_desc, width=80)
        messagebox.showinfo("Vitals and Medical History", wrapped_text)


    def create_disclaimer_screen(self):
        import tkinter as tk
        from tkinter import ttk

        #  Centered group: title + body (centered as a unit)
        self._disc_center = ttk.Frame(self.disclaimer_frame)
        self._disc_center.pack(expand=True)  # centers vertically as one block

        self.disclaimer_title = ttk.Label(
            self._disc_center,
            text="ARIana Intussusception Simulator",
            font=("Arial", 32, "bold"),
            justify="center",
            anchor="center",
        )
        self.disclaimer_title.pack(pady=(0, 2))  # tiny gap under title

        disclaimer_text = (
            "This device is designed to help a trained pediatric radiologist teach a\n"
            "trainee the basics of reducing an intussusception with an air enema. It is\n"
            "intended to supplement rather than replace the experience of performing a\n"
            "supervised intussusception reduction on an actual patient.\n\n"
            "In other words, a trainee who has used this simulator a few times should\n"
            "not consider himself competent to perform an intussusception reduction in\n"
            "the absence of any further experience. There are nuances of the\n"
            "intussusception reduction procedure that are not within the scope of this\n"
            "device and which can only be gained through practical experience under the\n"
            "watchful eye of a trained pediatric radiologist.\n\n"
            "If you agree with the above disclaimer, click \"I Agree\" to use this\n"
            "program."
        )
        self.disclaimer_label = ttk.Label(
            self._disc_center,
            text=disclaimer_text,
            font=("Arial", 21),
            wraplength=900,
            justify="center",
            anchor="center",
        )
        # NOTE: no expand=True here -> prevents big vertical gap
        self.disclaimer_label.pack(padx=24, pady=(0, 0), fill=tk.X)

        #  Buttons pinned to bottom 
        self.disclaimer_button_frame = ttk.Frame(self.disclaimer_frame)
        self.disclaimer_button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        ttk.Button(self.disclaimer_button_frame, text="I Agree", command=self.show_startup).pack(side=tk.LEFT, padx=10)
        ttk.Button(self.disclaimer_button_frame, text="Decline", command=self.root.quit).pack(side=tk.LEFT, padx=10)

        # Bind resize (add='+' so we don't clobber other Configure handlers) and apply once
        self.disclaimer_frame.bind("<Configure>", self._resize_disclaimer, add="+")
        self._resize_disclaimer()  # initial sizing


    def _resize_disclaimer(self, event=None):
        """Responsive sizing for disclaimer without introducing vertical gaps."""
        # Current size
        w = (event.width if event else self.disclaimer_frame.winfo_width()) or 900
        h = (event.height if event else self.disclaimer_frame.winfo_height()) or 700

        # Scale factor (lower denominator -> larger overall text)
        a=1.2
        scale = min(w / (1000.0*a), h / (650.0*a))  # was 1200,800 before

        # Increased base sizes
        BASE_TITLE, MIN_TITLE, MAX_TITLE = 34, 20, 54  # was 26 base
        BASE_BODY,  MIN_BODY,  MAX_BODY  = 22, 14, 38  # was 16 base

        title_size = max(MIN_TITLE, min(MAX_TITLE, int(BASE_TITLE * scale)))
        body_size  = max(MIN_BODY,  min(MAX_BODY,  int(BASE_BODY  * scale)))

        # Apply fonts
        self.disclaimer_title.config(font=("Arial", title_size, "bold"))
        self.disclaimer_label.config(font=("Arial", body_size))

        # Increased wrap clamp so lines stay longer before wrapping
        wrap = max(500, min(int(w * 0.95), 1800))  # was 420–1400
        self.disclaimer_label.config(wraplength=wrap)

        # Tiny fixed gap
        self.disclaimer_title.pack_configure(pady=(0, 2))

        # Scaled side padding
        pad_x = max(20, int(w * 0.05))
        self.disclaimer_label.pack_configure(padx=pad_x, pady=(0, 0), fill="x")

        # Keep centered group & pinned buttons
        self._disc_center.pack_configure(expand=True)
        self.disclaimer_button_frame.pack_configure(side="bottom", fill="x", pady=10)




    def create_startup_screen(self):
        main_frame = ttk.Frame(self.startup_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        selection_frame = ttk.Frame(main_frame)
        selection_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        title_frame = ttk.Frame(selection_frame)
        title_frame.pack(fill="x", pady=(0,10))
        ttk.Label(title_frame, text="Select a Patient Case", font=("Arial", 16, "bold")).pack(side=tk.LEFT)
        ttk.Button(title_frame, text="\u21BB", command=self.load_cases_into_tree).pack(side=tk.RIGHT)
        
        button_frame = ttk.Frame(selection_frame)
        button_frame.pack(side=tk.BOTTOM, pady=(10, 0), fill="x")
        
        ttk.Button(button_frame, text="Select Case", command=self.start_selected_case).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=10)        
        tree_frame = ttk.Frame(selection_frame)
        tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        style = ttk.Style()
        style.configure("Treeview", rowheight=60)

        self.case_list_tree = ttk.Treeview(tree_frame, columns=("Name", "Description"), show="headings", style="Treeview", yscrollcommand=scrollbar.set)
        self.case_list_tree.heading("Name", text="Patient Name")
        self.case_list_tree.heading("Description", text="Description")
        self.case_list_tree.column("Name", width=150, stretch=tk.NO)
        self.case_list_tree.column("Description", width=400)
        
        self.case_list_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.case_list_tree.yview)
        
        self.case_list_tree.bind("<<TreeviewSelect>>", self.on_case_select_display)

        # Removed preview_frame and its contents

    def on_case_select_display(self, event=None):
        selected_items = self.case_list_tree.selection()
        if not selected_items: return
        case_id = selected_items[0]
        try:
            case_data = self.case_loader.load_case(case_id)
        except Exception as e:
            print(f"Could not load preview for {case_id}: {e}")

    def load_cases_into_tree(self):
        for i in self.case_list_tree.get_children():
            self.case_list_tree.delete(i)
        
        self.all_cases_data = self.case_loader.load_case_list()
        
        for case in self.all_cases_data:
            wrapped_desc = textwrap.fill(case["description"], width=50)
            self.case_list_tree.insert("", tk.END, iid=case["id"], values=(case["name"], wrapped_desc))

    def start_selected_case(self):
        selected_items = self.case_list_tree.selection()
        if not selected_items:
            messagebox.showwarning("No Case Selected", "Please select a patient case to start the simulation.")
            return
        case_id = selected_items[0]
        try:
            self.current_case = self.case_loader.load_case(case_id)

            if self.current_case:
                self.simulator.load_case(self.current_case)
                self.show_pre_operation() # Navigate to Pre-Operation screen
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred while loading the case: {e}")
    def start_pressure_sampling(self):
        def sample_loop():
            if not self.simulator.is_running:
                return
            
            pressure = 0
            if self.virtual_slider_var.get():
                pressure = round(self.pressure_var.get())
            else:
                pressure = self.manometer_pressure # Use manometer pressure when virtual slider is off

            result = self.simulator.process_pressure_reading(pressure)
            self.update_simulation_status(result)
            self.root.after(50, sample_loop)  # Adjust interval as needed (ms)

        self.root.after(50, sample_loop)

    def create_simulation_screen(self):
        main_sim_frame = ttk.Frame(self.simulation_frame)
        main_sim_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_sim_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_panel.pack_propagate(False)

        right_panel = ttk.Frame(main_sim_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        controls_frame = ttk.LabelFrame(left_panel, text="Controls")
        controls_frame.pack(pady=10, fill="x")

        self.virtual_slider_var = tk.BooleanVar(value=False) # Default to unchecked
        slider_switch = ttk.Checkbutton(controls_frame, text="Virtual Pressure Slider", variable=self.virtual_slider_var, command=self.toggle_pressure_input)
        slider_switch.pack(anchor="w", padx=5, pady=(5,0))

        # Use a regular Frame for the pressure input area to remove the dark box
        self.pressure_input_frame = ttk.Frame(controls_frame) # Changed from ttk.Frame to tk.Frame if dark box is from ttk.Frame style
        self.pressure_input_frame.pack(pady=5, padx=5, fill="x")

        self.pressure_frame = ttk.Frame(self.pressure_input_frame)
        ttk.Label(self.pressure_frame, text="Pressure (mmHg):").pack(side=tk.LEFT)
        self.pressure_display_label = ttk.Label(self.pressure_frame, text="0", width=4)
        self.pressure_display_label.pack(side=tk.RIGHT)
        self.pressure_scale = ttk.Scale(self.pressure_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=self.pressure_var, command=lambda v: self.on_pressure_change(v, self.pressure_display_label))
        self.pressure_scale.pack(side=tk.RIGHT, fill="x", expand=True)

        # Manometer Status and Pressure Display

        self.manometer_status_label = ttk.Label(self.pressure_input_frame, text="Disconnected", foreground="red")

        self.manometer_status_label.pack(fill="x", expand=True)

        self.toggle_pressure_input()

        ttk.Button(controls_frame, text="Take Fluoroscopy Image", command=self.take_fluoro_image).pack(pady=5, fill="x")
        ttk.Button(controls_frame, text="Call for Surgery", command=self.call_for_surgery).pack(pady=5, fill="x")
        ttk.Button(left_panel, text="End Simulation", command=self.end_simulation).pack(side=tk.BOTTOM, pady=10, fill="x")

        status_frame = ttk.LabelFrame(left_panel, text="Live Status")
        status_frame.pack(pady=10, fill="x")
        
        status_frame.grid_columnconfigure(1, weight=1)
        for i, status_name in enumerate(["Stage", "Sim Time", "Pressure", "Fluoro Time", "Outcome"]):
            row_frame = ttk.Frame(status_frame) # Use a frame for each row
            row_frame.grid(row=i, column=0, columnspan=2, sticky="ew")
            row_frame.grid_columnconfigure(1, weight=1)
            
            label = ttk.Label(row_frame, text=f"{status_name}:", width=12, anchor="w")
            label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
            value_label = ttk.Label(row_frame, text="N/A", anchor="w")
            value_label.grid(row=0, column=1, sticky="w", padx=5, pady=2)
            
            self.status_labels[status_name] = value_label

        # Red dot indicator for fluoroscopy
        self.fluoro_dot_canvas = tk.Canvas(status_frame, width=10, height=10, highlightthickness=0)
        self.fluoro_dot_canvas.create_oval(0, 0, 10, 10, fill="red", outline="")
        self.fluoro_dot_canvas.grid(row=0, column=2, padx=2, pady=2, sticky="ne") # Position in top-right
        self.fluoro_dot_canvas.grid_remove() # Initially hide the dot

        visibility_frame = ttk.LabelFrame(left_panel, text="Visibility")
        visibility_frame.pack(pady=10, fill="x")
        self.visibility_vars = {}
        for status_name in self.status_labels.keys():
            var = tk.BooleanVar(value=True)
            self.visibility_vars[status_name] = var
            chk = ttk.Checkbutton(visibility_frame, text=status_name, variable=var, command=lambda name=status_name: self.toggle_visibility(name))
            chk.pack(anchor="w", padx=5)
        
        # Add red warning text underneath visibility settings
        self.warning_label = tk.Label(left_panel, text="", fg="red", font=("Arial", 16, "bold"), wraplength=300, justify="left", relief="flat", bd=0)
        # Don't pack it yet — wait until text is shown
     
        self.image_label = ttk.Label(right_panel, anchor="center")
        
        # Pop-out button for Simulation fluoroscopy
        self.image_label.pack(pady=10, fill=tk.BOTH, expand=True)

    def create_results_screen(self):
        main_frame = ttk.Frame(self.results_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Use a 3-row grid layout: summary, content, and buttons
        main_frame.grid_rowconfigure(1, weight=1)  # Let content_frame expand
        main_frame.grid_columnconfigure(0, weight=1)

        # Row 0: Summary
        self.results_summary = ttk.Label(main_frame, text="", font=("Arial", 12), justify="center", anchor="center")
        self.results_summary.grid(row=0, column=0, pady=10, sticky="ew")

        # Row 1: Content Frame
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, sticky="nsew")
        
        content_frame.grid_columnconfigure(0, weight=1) # Plot column
        content_frame.grid_columnconfigure(1, weight=1) # Image column
        content_frame.grid_rowconfigure(0, weight=1)    # Ensure the row can expand vertically

        # Left content: Plot
        self.plot_canvas_frame = ttk.Frame(content_frame)
        self.plot_canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.plot_image_label = ttk.Label(self.plot_canvas_frame)
        self.plot_image_label.pack(fill="both", expand=True)
        self.plot_image_label.configure(anchor="center", justify="center", wraplength=600)

        # Right content: Image scroller
        image_frame = ttk.Frame(content_frame)
        image_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        ttk.Label(image_frame, text="Post-procedure Images", font=("Arial", 12, "bold")).pack()
        self.results_image_scroller = ImageScroller(image_frame)
        self.results_image_scroller.pack(fill="both", expand=True)

        # Row 2: Buttons (FIXED HEIGHT)
        bottom_button_frame = ttk.Frame(main_frame, height=60)
        bottom_button_frame.grid(row=2, column=0, sticky="ew", pady=5)
        bottom_button_frame.grid_propagate(False)  # Prevent collapse

        ttk.Button(bottom_button_frame, text="Back to Case Selection", command=self.show_startup).pack(side=tk.LEFT, padx=20, ipadx=10, ipady=5)
        ttk.Button(bottom_button_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=20, ipadx=10, ipady=5)


    def toggle_pressure_input(self):
        """Shows/hides the virtual slider or the serial status label."""
        if self.virtual_slider_var.get():
            self.manometer_status_label.pack_forget()
            self.pressure_frame.pack(fill="x", expand=True)
        else:
            self.pressure_frame.pack_forget()
            self.manometer_status_label.pack(fill="x", expand=True)
        # Ensure focus is returned to the root window after toggling input
        self.root.focus_set()

    def on_pressure_change(self, value, label):
        rounded_value = round(float(value))
        label.config(text=str(rounded_value))
        result = self.simulator.process_pressure_reading(rounded_value)
        self.update_simulation_status(result)

    def call_for_surgery(self):

        # Was the button pressed before starting the simulation?
        try:
            pressed_in_preop = (self.simulator.sim_time == 0 and self.simulator.current_stage == 1)
        except Exception:
            pressed_in_preop = True  # safest default if simulator isn't initialized yet

        self.called_surgery_from_preop = pressed_in_preop

        messagebox.showinfo("Surgery Called", "The patient has been sent to surgery.")
        self.end_simulation(outcome_override="Patient Sent to Surgery")


    def toggle_visibility(self, name):
        """Hides or shows a status label row by changing its text and color."""
        value_label = self.status_labels[name]
        if self.visibility_vars[name].get():
            # When checked, restore the last known value and default color
            value_label.config(text=self.status_label_last_values.get(name, "N/A"), foreground="") # Set to default text color
        else:
            # When unchecked, display "Not Shown", and grey out text
            value_label.config(text="Not Shown", foreground="grey") # Simplified text, set color to grey
        # After toggling visibility, ensure focus is returned to the root window
        self.root.focus_set()

    def show_red_dot(self):
        self.fluoro_dot_canvas.grid()

    def hide_red_dot(self):
        self.fluoro_dot_canvas.grid_remove()

    def take_fluoro_image(self, event=None):
        # Show the red dot
        self.show_red_dot()
        # Schedule hiding the red dot after 300ms
        self.root.after(300, self.hide_red_dot)

        # Call the simulator\"s take_fluoro_image method directly
        result = self.simulator.take_fluoro_image()
        
        if result["result"] == "not_running":
            return "break"
        elif result["result"] == "radiation_overdose":
            messagebox.showerror("Radiation Overdose", "Radiation limit exceeded! Simulation ended.")
            self.end_simulation()
        else:
            self.display_image(result["image_path"])
        self.root.focus_set()
        return "break" # Crucial: Stop event propagation
    def show_plot_image(self):
        if not self.result_plot_images:
            self.plot_image_label.config(image="", text="No plot images")
            return
        img = self.result_plot_images[self.result_plot_index]
        self.plot_image_label.config(image=img, text="")
        self.plot_image_label.image = img  # prevent garbage collection

    def update_simulation_status(self, result):
        if not result:
            return

        # Handle warning separately
        if result.get("type") == "warning" and result.get("message") == "3_min_warning":
            if not self.warning_shown:
                self.warning_label.config(
                    text="WARNING: The patient has been under pressure for 3 minutes. "
                        "This is the maximum recommended amount of time. It is advised to complete or end the surgery promptly."
                )
                self.warning_label.pack(pady=(5, 10), fill="x")
                self.warning_shown = True
            return  # Skip rest of method for warnings

        # Expect regular outcome dictionary
        try:
            self.status_label_last_values["Stage"] = str(result["current_stage"])
            self.status_label_last_values["Sim Time"] = f'{result["sim_time"]:.1f}s'
            self.status_label_last_values["Pressure"] = "{:.0f} mmHg".format(result["pressure"])
            self.status_label_last_values["Fluoro Time"] = f'{result["fluoro_time"]:.0f}s'
            self.status_label_last_values["Outcome"] = result["outcome"]

            for name in self.status_labels.keys():
                if self.visibility_vars[name].get():
                    self.status_labels[name].config(text=self.status_label_last_values[name], foreground="")
                else:
                    self.status_labels[name].config(text="Not Shown", foreground="grey")

            if result["outcome"] == "Perforated Vitals Crashed":
                messagebox.showerror("Vitals Crashed!", "The patient perforated 3 minutes ago and was left untreated. The patient's vitals have now crashed.")
                self.end_simulation(outcome_override="Patient Perforated and Vitals Crashed")

            elif result["outcome"] == "Time Limit 5 Min":
                messagebox.showerror("Time Limit Exceeded", "The patient has undergone insufflation for 5 minutes. This is past the maximum recommended time, the simulation will now end.")
                self.end_simulation(outcome_override="Excessive Insufflation Occurred")

        except KeyError as e:
            print(f"Malformed result received in update_simulation_status: missing {e}")

    # In the ARIanaApp class...

    def display_image(self, image_path): 
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                
                # Use the main image_label for sizing
                container = self.image_label
                container.update_idletasks()
                panel_w = container.winfo_width()
                panel_h = container.winfo_height()

                if panel_w > 1 and panel_h > 1:
                    img.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
                else:
                    # Fallback for when the window is not yet rendered
                    img.thumbnail((600, 450), Image.Resampling.LANCZOS)

                # Keep a reference to the PhotoImage to prevent garbage collection
                self.photo_image = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.photo_image, text="")
            except Exception as e:
                print(f"Error displaying image {image_path}: {e}")
                self.image_label.config(image="", text=f"Image not found:\n{os.path.basename(image_path)}")
        else:
            # This handles the case where there's no image to display
            self.image_label.config(image="", text="No Image Available")


    
    def on_window_resize(self, event):
        # Redraw current images
        if hasattr(self, 'display_current_image'):
            self.display_current_image()
        if hasattr(self, 'display_result_image'):
            self.display_result_image()
    
    def render_figure_to_photoimage(self, fig):
        """Render a matplotlib figure to a Tkinter-compatible PhotoImage."""
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()  # modern API
        w, h = canvas.get_width_height()

        # Create a Pillow image from the RGBA buffer without copying
        img = Image.frombuffer("RGBA", (w, h), buf, "raw", "RGBA", 0, 1)
        # If you want RGB (no transparency), convert here
        img = img.convert("RGB")

        photo = ImageTk.PhotoImage(img)
        plt.close(fig)
        return photo


    def plot_performance_data(self, data):
        self.result_plot_images = []
        self.result_plot_index = 0

        if not data or not data.get("time_history"):
            print("No performance data to plot.")
            return

        times = np.array(data["time_history"])
        pressures = np.array(data["pressure_history"])
        stages = np.array(data["stage_history"])
        showplot = True
        if len(times) < 2:
            print("Not enough data points to generate plot.")        
            showplot=False
            return

        # Remove duplicates
        unique_times, unique_indices = np.unique(times, return_index=True)
        times = unique_times
        pressures = pressures[unique_indices]
        stages = stages[unique_indices]

        # Interpolate
        dense_time = np.linspace(times[0], times[-1], 200)
        pressure_interp = interp1d(times, pressures, kind='linear', fill_value="extrapolate")
        stage_interp = interp1d(times, stages, kind='previous', fill_value="extrapolate")

        smooth_pressure = gaussian_filter1d(pressure_interp(dense_time), sigma=2)
        smooth_stage = gaussian_filter1d(stage_interp(dense_time), sigma=1.5)

        # First plot: smoothed
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Pressure (mmHg)", color="red")
        ax1.plot(dense_time, smooth_pressure, color="red", label="Smoothed Pressure")
        ax1.tick_params(axis="y", labelcolor="red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Stage", color="blue")
        ax2.plot(dense_time, smooth_stage, color="blue", label="Smoothed Stage")
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2.set_yticks(range(1, max(stages) + 2))

        fig.tight_layout()
        self.result_plot_images.append(self.render_figure_to_photoimage(fig))

        # Show first plot
        if showplot != False:
            self.show_plot_image()
  
    def show_disclaimer(self):
        self.notebook.select(self.disclaimer_frame)
    
    def show_startup(self):
        self.load_cases_into_tree()
        self.notebook.select(self.startup_frame)
    
    def show_simulation(self):
        self.pressure_var.set(0.0)
        if hasattr(self, "pressure_display_label"):
            self.pressure_display_label.config(text="0")

        # Reset warning flag when starting new simulation
        self.warning_shown = False
        self.warning_label.config(text="")

        self.notebook.select(self.simulation_frame)
        
        # Reset all visibility checkboxes to checked and update labels
        for name in self.visibility_vars.keys():
            self.visibility_vars[name].set(True)

        # Start simulator and begin UI updates
        self.simulator.start_simulation(callback=self.update_simulation_status)
        
        # Begin periodic sampling of pressure input (either virtual or manometer)
        self.start_pressure_sampling()

        # Initial image display: No image until first fluoro
        self.image_label.config(image="", text="Take first fluoroscopic image to begin simulation.")

        # Initial update of UI with baseline values
        self.update_simulation_status({
            "current_stage": 1,
            "sim_time": 0,
            "fluoro_time": 0,
            "pressure": 0,
            "outcome": "Ready"
        })
        # Ensure root has focus to capture keyboard events
        self.root.focus_set()

    def end_simulation(self, outcome_override=None):
        """Stop the simulator and render the Results tab based on how the run ended."""
        # 1) Stop the engine
        self.simulator.stop_simulation()

        # 2) Collect data once
        performance_data = self.simulator.get_performance_data()
        if not performance_data:
            # Nothing to show; still navigate to Results with a minimal summary
            self.results_summary.config(text=f"Simulation ended.\nOutcome: {outcome_override or self.simulator.last_outcome}")
            self.plot_image_label.config(image="", text="No performance data for this case.")
            self.plot_image_label.image = None
            post_images = self.current_case.get("images", {}).get("postprocedure", [])
            self.results_image_scroller.set_images(post_images)
            self.show_results()
            self.called_surgery_from_preop = False
            return

        # 3) Decide if we should suppress plotting (Pre-Op surgery or explicit contraindication start)
        is_contra_case = (
            self.current_case and
            self.current_case.get("parameters", {}).get("contraindication_start", 0) == 1
        )
        called_surgery_in_preop = (
            outcome_override == "Patient Sent to Surgery"
            and getattr(self, "called_surgery_from_preop", False)
        )

        if is_contra_case or called_surgery_in_preop:
            # No plot; centered message
            if hasattr(self, "plot_image_label"):
                self.plot_image_label.config(image="", text="No performance data for this case.")
                self.plot_image_label.image = None  # clear any stale image

            # Summary + post images
            summary_text = "Simulation ended.\n"
            summary_text += f"Outcome: {outcome_override or 'Contraindication was not recognized'}"
            self.results_summary.config(text=summary_text)

            post_images = self.current_case.get("images", {}).get("postprocedure", [])
            self.results_image_scroller.set_images(post_images)

            self.show_results()
            # Reset the flag for future runs
            self.called_surgery_from_preop = False
            return

        # 4) Normal path (plot should be shown)
        self.called_surgery_from_preop = False  # ensure it doesn’t leak into next run

        # Summary
        summary_text = "Simulation ended.\n"
        summary_text += f"Outcome: {outcome_override or self.simulator.last_outcome}"
        self.results_summary.config(text=summary_text)

        # Post-procedure images
        post_images = self.current_case.get("images", {}).get("postprocedure", [])
        self.results_image_scroller.set_images(post_images)

        # Plot performance (fallback text if plotting can’t produce an image)
        self.plot_image_label.config(image="", text="")   # clear any prior text
        self.result_plot_images = []                      # reset plot cache
        self.plot_performance_data(performance_data)      # populates self.result_plot_images (or not)

        if not self.result_plot_images:
            # Not enough points or some guard tripped inside plotter
            self.plot_image_label.config(image="", text="No performance data available to plot.")
            self.plot_image_label.image = None

        # 5) Navigate to Results
        self.show_results()



    def show_results(self):
        self.notebook.select(self.results_frame)

    def run(self):
        try:
            self.root.mainloop()
        finally:
            # Ensure cleanup happens even if mainloop exits unexpectedly
            print("Cleaning up manometer connection...")
            self.manometer_thread.stop()
            if self.manometer_thread.is_alive():
                self.manometer_thread.join(timeout=1.0)

    def check_manometer_queue(self):
        try:
            while True:
                msg_type, *values = self.manometer_queue.get_nowait()
                if msg_type == "status":
                    status_text = values[0]
                    # Combine "Connecting..." and "Handshaking..." into a single "Connecting..." status, quick fix
                    if status_text == "Handshaking...":
                        status_text = "Connecting..."

                    self.manometer_status_label.config(text=status_text)
                    if status_text == "Connecting...":
                        self.manometer_status_label.config(foreground="yellow")
                    elif status_text == "Connected":
                        self.manometer_status_label.config(foreground="green") # Changed to green
                    else: # Disconnected
                        self.manometer_status_label.config(foreground="red")
                        self.manometer_pressure = 0 # Reset pressure on disconnect
                elif msg_type == "pressure":
                    pressure_mmhg = values[0]
                    original_unit_name = values[1]
                    unit_code = values[2]

                    self.manometer_pressure = pressure_mmhg
                    # Update pressure display if manometer is active
                    if not self.virtual_slider_var.get():
                        display_text = f"Connected: {self.manometer_pressure} mmHg"
                        if unit_code != 0x05: # If not already mmHg
                            self.manometer_status_label.config(text=f"Connected: {self.manometer_pressure} mmHg (from {original_unit_name})", foreground="green") # Still green
                        else:
                            self.manometer_status_label.config(text=display_text, foreground="green") # Still green

        except queue.Empty:
            pass # No messages in queue
        finally:
            self.root.after(100, self.check_manometer_queue) # Check again in 100ms

if __name__ == "__main__":
    app = ARIanaApp()
    app.run()