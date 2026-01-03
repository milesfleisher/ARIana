## ARIana (Python/Tk)
[*https://tinyurl.com/ARIana-download*]
(https://tinyurl.com/ARIana-download)

## A cross-platform simulator for teaching air-enema reduction of pediatric intussusception.

### Disclaimer

This device is designed to help a trained pediatric radiologist teach a trainee the basics of reducing an intussusception with an air enema. It is intended to supplement rather than replace the experience of performing a supervised intussusception reduction on an actual patient. In other words, a trainee who has used this simulator a few times should not consider himself competent to perform an intussusception reduction in the absence of any further experience. There are nuances of the intussusception reduction procedure that are not within the scope of this device and which can only be gained through practical experience under the watchful eye of a trained pediatric radiologist.

### History

ARIana was originally a closed source app coded in MATLAB. The original app was a product of Lucy LLC created by S.K. Soosman, G.E. Roper, A.S. Wexler, J.C. Li, R. Stein-Wexler. The original MATLAB version was discontinued, but the demand for a computer-based intussusception simulator remained. This is a new version of ARIana that has used the original MATLAB file as inspiration. Miles Fleisher created this new, open source, Python/Tkinter version with help from the original creators. In addition to porting from MATLAB to Python/Tkinter, some additional functionality has been added, as well as more cross-platform support.

### Downloading:
Operating System Versions Supported:  
ARIana currently supports Windows 7 or later and macOS 10.9 or later. There is also a Linux version available but it is currently in testing. If you are using an unsupported operating system, you are also welcome to compile for yourself using the Python files in the Source Code folder. Niutika is the recommended compiler. There are .yml files that include Niutika commands for Windows, Mac, and Linux, which can be used to compile your own version. 

Windows:  
Download [*ARIana-Windows-Installer.zip*](https://github.com/milesfleisher/ARIana/raw/refs/heads/main/Source%20and%20Resources/App%20Downloads/Windows/ARIana-Windows-Installer.zip?download=). Place the folder in a read/write folder and extract the zip file. **Do not remove the ARIana.exe file from the folder.** If you would like to put the app somewhere else, instead, create a shortcut by right clicking and pressing “Create a shortcut” and drag the created shortcut to your preferred location. Double click this shortcut to launch ARIana.exe. The first time you launch ARIana, it may say “Windows Protected Your PC. Microsoft Defender SmartScreen prevented an unrecognized app from starting. Running this app might put your PC at risk.” This can be safely ignored–the error occurs because, since this app is open source and doesn’t generate profit, it wasn’t signed under a paid Microsoft license. All the code was written by Miles Fleisher and can be viewed in the Source Code folder and all dependencies have been thoroughly vetted.  This issue can be easily fixed by clicking   
the small “more info” link and then clicking “Run anyway.” After doing this once, the warning should not show up again. 

On Windows, it is important to make sure that you have the correct drivers and that they’re up to date. This app is designed to work with an Extech HD 750 Differential Manometer, which uses a CP2102 USB to UART Bridge. The drivers for this bridge are available from [Silicon Labs](https://www.silabs.com/software-and-tools/usb-to-uart-bridge-vcp-drivers?tab=downloads). If you are on Windows 10 (build 1803\) or older (like Windows 9, 8, and 7), it’s recommended to install the Windows VCP driver. If you’re on Windows 10 (build 1803\) or higher, it’s recommended to install the Windows Universal Driver. If you are on Windows 10 but don’t know if you’re before or after version 1803, you can type *winver* into the Command Prompt and it will take you to a page listing the build number. If the build number is greater than or equal to 17134, then you are on a new version of Windows and should install the Universal driver [here](https://www.silabs.com/documents/public/software/CP210x_Universal_Windows_Driver.zip). If the build number is smaller than 17134, you are on an older build of Windows and should install the VCP driver [here](https://www.silabs.com/documents/public/software/CP210x_VCP_Windows.zip). To check if your driver is up to date, go to the *Device Manager* app, click view, verify “show hidden” is enabled, and expand “Ports (COM & LPT).” The driver should be called “Silicon Labs CP210x USB to UART Bridge (COMx).” Right click on the driver, click “Properties,” and go into the Drivers tab. Note the Driver Version and Driver Date and compare them to the latest available on Silicon Labs’ website. As long as your driver is in the same major version as the ones listed below (v11 for the universal driver or v6.7 for the VCP driver), it should probably be fine but, if you are having trouble connecting to the manometer, it might be worth updating. As of 9/3/25, the latest drivers are as shown below:

- *v11.4.0 → Current Universal driver (for Win10 1803+ / Win11).*  
- *v6.7 (2020) → Latest legacy VCP driver (for Win7/8/early 10).*

The first time you run the app, you may experience a longer loading time than usual. This is because the app is generating multiple setting files. The app may also give you some alerts that say that the “checklist file could not be found.” As long as the alert doesn’t say that the “file couldn’t be created,” this is not an issue and can safely be ignored. On future startups, the app should load much faster. 

Mac:  
Download [*ARIana-Mac-Installer.dmg*](https://github.com/milesfleisher/ARIana/raw/refs/heads/main/Source%20and%20Resources/App%20Downloads/Mac/ARIana-macOS.zip?download=). Open the .dmg file and drag the ARIana app to your Applications folder. You may then safely eject the ARIana.dmg. 

The first time you run the app, you may experience a longer loading time than usual. This is because the app is generating multiple setting files. The app may also give you some alerts that say that the “checklist file could not be found.” As long as the alert doesn’t say that the “file couldn’t be created,” this is not an issue and can safely be ignored. On future startups, the app should load much faster. 

When launching the app for the first time, you may receive a warning that “the app cannot be scanned for malware.” This is not an issue and it can be safely ignored. This warning is because the app was not signed with the Apple Developer License. This can be safely ignored–the error occurs because, since this app is open source and doesn’t generate profit, it wasn’t signed under a paid Microsoft license. All the code was written by Miles Fleisher and can be viewed in the Source Code folder and all dependencies have been thoroughly vetted. The warning can be bypassed by first going to *System Preferences\>Security*, scrolling down to the bottom. Next, find the warning that says “ARIana was blocked to protect your Mac.” Click *Open Anyway*, and give your admin password. After you’ve done this once, the app should run as expected in the future. 

### Folder Layout:

The files in this GitHub are laid out as follows. \<CaseID\> is a placeholder for the names of different cases/patients. 

Folder Layout  
The project's folder structure is organized as follows:  
Plain Text  
repo/  
├─ ARIana.py \# Main application entry point for logic file(intussusception\_trainer.py)  
├─ Patients/  
│ └─ \<CaseID\>/  
│ ├─ \<CaseID\>\_metadata.json \# Case parameters (see schema below)  
│ └─ Images/  
│       ├─ Preprocedure/ \# Pre-procedure images (e.g.,preprocedure\_1.png)  
│       ├─ Simulation/ \# Simulation images (e.g.,simulation\_1.png)  
│       └─ Postprocedure/ \# Post-procedure images (e.g.,postprocedure\_1.png)  
├─ ARIana\_logo.png \# Application icon/branding  
└─ README.md

### Adding your own preoperation checklist

Windows:  
Go into the folder holding ARIana.exe, find *preop\_checklist.txt, and* open this file in your text editor of choice, make changes to it, and save it. Make sure that you save the file as a .txt and don’t duplicate the original. Note that some special characters may not be supported and can cause issues with the graphical interface. Reloading the app may be required for changes to take effect. 

Mac:  
Right click on the ARIana app and click “Show Package Contents.” Navigate to *preop\_checklist.txt* and open the file in your text editor of choice, make changes to it, and save it. Make sure that you save the file as a .txt and don’t duplicate the original. Note that some special characters may not be supported and can cause issues with the graphical interface. Reloading the app may be required for changes to take effect. 

### Adding you own cases:

Windows:   
Go into the folder holding ARIana.exe, find the folder named “Placeholder.” Don’t include spaces in the name of your folder. Copy this folder and rename it to the name of your patient/case. This is an extremely important step as the app will ignore folders named “Placeholder.” Enter this folder and rename the placeholder\_metadata.json to \<patient name\>\_metadata.json(the brackets should not be included; they are only there to emphasize the part of the name that should be replaced with the name of the case). Make sure that the name of the .json file and the name of the folder containing it match. Go into the *Images* folder and replace the images in each of the three folders (*Preoperation, Simulation,* and *Postoperation).* Make sure that you use the same naming convention as the placeholder images: \<Foldername\>\_\<imagenumber\>.png. The image number dictates the order that images will be shown and, in the simulation folder, the order stage that an image will be associated with. If you have a perforation image, add it to the Images folder as the final image. You can also use any of the images from the included cases to create your new cases with new probabilities or features. Please make sure that all images are formatted as .png. 

Now that you have renamed the folder, .json, and added your own images, it’s time to edit the .json file. Open the .json file in any text editor. Make sure that the “*num stages”* value is set to the number of images in the Simulation folder, otherwise, some of the images may not display. The following is a list of values in the .json file and what they control. 

| Value  | Control |
| :---- | :---- |
| *name* | The name displayed in the case selection screen |
| *teaser* | The short case description displayed in the case selection screen. |
| *clinical\_descrip* | The longer description displayed when the “Take Vitals and History” is pressed in the preoperation screen |
| *perf\_data* | This is the probability at each pressure listed that the patient will perforate. Pairs are listed as *\[pressure, probability\]* and are linearly interpolated for values in between the listed pressure/probability pairs listed |
| *ret\_data* | This is the probability for retrogression (moving from the current stage to a previous stage). This uses the same linear interpolation logic as perf\_data |
| *coeff* | This coefficient value effectively is the maximum probability of success when the pressure is at the maximum (180 mmHg). This value scales the success probabilities for all other pressures. |
| *num\_stages* | The number of stages in the simulation. This should be set to the number of images in the *Simulation* folder |
| *donstart* | This value is *0* by default. If the patient has a contraindication and surgery is not advised, set it to *1\.* Please note that this will cause the app to skip the simulation step entirely and display a warning if the user tries to start air enema reduction.  |

### Hardware Setup:

While the software can be used with a virtual pressure slider and no external hardware, the full setup tends to be more immersive. The total materials required to build the setup are as follows:

- Computer running the ARIana software  
- USB mini B cable that can plug into your computer  
- Extech HD750 Differential Manometer  
- ⅛” inner diameter flexible tubing  
- ¼” inner diameter flexible tubing  
- Insufflator with luer-lock output  
- Luer lock to ¼” inner diameter flexible tubing adapter (can use a 3 way stopcock with one of the branches turned off and a luer lock to ¼” tubing adapter   
- Bulb with tubing barb attachments for insufflator (¼” inner diameter) and manometer (⅛” inner diameter)   
  - Optionally put this inside a modified doll  
- (optional) bleed valve with male ¼” inner diameter tubing on both ends to simulate slight leaks and inefficiencies  
  - If you are using a bleed valve, make sure to connect the manometer onto or near the bulb instead of connecting it between the insufflator and bleed valve. 

### Usage:

Open the ARIana app. Read and agree to the disclaimer. Select a patient and press “Select Patient.” You are now in the “preoperation screen.” The resident can go through the preprocedure checklist, get more information about the patient using the “Take Vitals and History” button, and zoom into the images by clicking and scrolling. The zoom feature can also be used to pop out images into a separate window for later reference, as they will not be cleared by the app. If there are multiple preprocedure images, the user can scroll through them with the arrow buttons at the bottom. 

After reading about the case, if the user finds a contraindication they should press *Call for Surgery.* 

Simulation:

In the simulation screen there is an option to use a virtual slider. This allows the app to be used without any external hardware. To access this, simply click the “Virtual Pressure Slider” button. There is also information about *Stage, Simulation Time, Pressure, Fluoroscopic Image Time, and Outcome.* These can be hidden to change the experience for the resident. 

To connect the manometer, make sure that the “Virtual Pressure Slider” button is left unchecked. Now, use a usb type A to mini usb type B cable (one is included with the Extech HD750) and plug it into the computer. It may take a few seconds to connect and slightly longer the first time. You will be able to see the connection status at the top left of the screen. The program will convert all pressure values into mmHg no matter what unit the manometer is in. However, for the most accurate results, it is best to put the manometer into mmHg units. 

The user can use the space bar to “take a fluoroscopic image.” This will increase the counter for fluoroscopic time.” When this counter reaches 3 minutes, the simulation will stop and a warning will be displayed saying that the patient experienced radiation poisoning. 

While the simulation is live, the program uses probabilistic logic to decide whether the patient should progress to the next stage (and show the next image) or retrogress to the previous stage. The formula for the probability of progression is as shown below, where stage\_coeff is the coefficient of the current stage listed in the .json.   
**success\_prob \= ((pressure / 180\) \*\* 2\) \* stage\_coeff**

The probabilities of retrogression and perforation are slightly more complicated. They are created by linearly interpolating between the *ret\_data* and the *perf\_data* respectively–these are both defined in the .json file–to find the correct probability for the current pressure and stage. 

These are the possible outcomes of the case:

| Outcome | Cause |
| :---- | :---- |
| Success | The intussusception was successfully reduced |
| Stuck | The intussusception was not successfully reduced |
| Perforation | The simulation ended and the patient perforated |
| Patient Sent To Surgery | The patient was sent to surgery either before the simulation or after the simulation ended |
| Contraindication Was Not Recognized | The patient had a contraindication that was not recognized in the preoperation. The user pressed “Start Air Enema Reduction” |
| Radiation Limit Exceeded\! | The fluoro time counter reached 5 minutes (the patient was under fluoroscopic imaging for too long and received radiation portioning). Every time the fluoro button is pressed(unless the button was pressed within the last 2 seconds), 2 seconds is added to the fluoro time counter. |
| Excessive Insufflation Occurred | The simulation lasted for longer than 5 minutes |
| Patient Perforated and Vitals Crashed | The patient perforated and the user did not end the simulation within 3 minutes of perforation |
| Perforation Occurred and Was Recognized | The patient perforated and the user ended the simulation within 3 minutes of perforation |

### Troubleshooting:

If you would like more information, you can read the [*original documentation*](https://github.com/milesfleisher/ARIana/blob/main/Source%20and%20Resources/Original%20Documentation/Original_ARIana_Manual.pdf) for the closed source version of ARIana. While this app was slightly different and based on MATLAB, a lot of the core logic and information is very similar. Note that the support email given is no longer monitored. Instead, please post in the *Discussion* section of this Github. 

### Why Is It Not Starting\!? – Troubleshooting:

First of all, make sure that all tubes and wires are plugged in appropriately.

Potential problems:

1. Pressing the space bar does not change the image.  
   1. Make sure that the mouse cursor on the screen is over the fluoroscopic image and click once on the image. This error may occur if a display setting is checked or unchecked during the procedure or if the mouse is clicked outside of the simulator window, deselecting the window.  
2. The Program displays a sensor error message upon starting the simulator from the pre-procedure window.  
   1. Make sure the pressure sensor is powered on and plugged into the computer.

   3\. 	A case from the startup screen will not load (an error message appears)

    

      a. 	This issue will only occur if the selected case file is incomplete or corrupted. Reinstalling the case file from the source will fix this problem. You can find them in ARIana/Source Code/Patients/\<patientname\>/Images

   4\. 	The pressure on the insufflator is not the same as the pressure that the manometer is measuring. What gives?

   a. 	In a closed system, the pressures would be identical. However, since the instructor creates a small air leak in the system with the control valve, there is a small difference in pressure. The program accounts for this with a “fudge factor.”

 **If you encounter an issue that is not described above and prevents the simulator from working properly, please contact us in the [GitHub Discussion forums](https://github.com/milesfleisher/ARIana/discussions).**
