# **A tutorial of multiple-class particle picking on SHREC2021 dataset**
## **Preprocessing**

- Data preparation
  
	Before launching the graphical user interface, we recommend creating a single folder to save inputs and outputs of DeepETpicker. Inside this base folder you should make a subfolder to store raw data. This raw_data folder should contain: 
  - tomograms(with extension .mrc or .rec)
  - coordinates file with the same name as tomograms except for extension. (with extension *.csv, *.coords or *.txt. Generally, *.coords is recoommand.).  

	<br>

  The sample dataset of SHREC_2021 can be download in one of two ways:
  - Baidu Netdisk Link: [https://pan.baidu.com/s/1aijM4IgGSRMwBvBk5XbBmw](https://pan.baidu.com/s/1aijM4IgGSRMwBvBk5XbBmw ); verification code: cbmi
  - Microsoft onedrive Link: [https://1drv.ms/u/s!AmcdnIXL3Vo4hWf05lhsQWZWWSV3?e=dCWvew](https://1drv.ms/u/s!AmcdnIXL3Vo4hWf05lhsQWZWWSV3?e=dCWvew); verification code: cbmi

- Data structure
  
  The data should be organized as follows:
	├── /base/path
	│   ├── raw_data
	│   │   ├── IS002_291013_005_iconmask2_norm_rot_cutZ.coords
	│   │   └── IS002_291013_005_iconmask2_norm_rot_cutZ.mrc
	│   │   └── IS002_291013_006_iconmask2_norm_rot_cutZ.mrc
	│   │   └── IS002_291013_007_iconmask2_norm_rot_cutZ.mrc
	│   │   └── IS002_291013_008_iconmask2_norm_rot_cutZ.mrc
	│   │   └── IS002_291013_009_iconmask2_norm_rot_cutZ.mrc
	│   │   └── IS002_291013_010_iconmask2_norm_rot_cutZ.mrc
	│   │   └── IS002_291013_011_iconmask2_norm_rot_cutZ.mrc
	``` 			

- Input & Output

	<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/images/EMPIAR_10045_GIF/Preprocessing.gif" width="80%"> 
	</div>

	Launch the graphical user interface of DeepETPicker. On the `Preprocessing` page, please set some key parameters as follows:
	- `input`
		- Dataset name: e.g. SHREC_2021_preprocess
		- Base path: path to base folder
		- Coords path:  path to raw_data folder 
		- Coords format: .csv, .coords or .txt
		- Tomogram path: path to raw_data folder
		- Tomogram format: .mrc or .rec
		- Number of classes:  multiple classes of macromolecules also can be localized separately
	- `Output`
		- Label diameter(in voxels):  smaller than the average diameter of the targets
		- Ocp diameter(in voxels): this value can be the average diameter of the targets. For particles of multi-classes, their diameters should be separated with a comma.
		- Configs: if you click 'Save configs', it would be the path to the file which contains all the parameters filled in this page  


### **Training of DeepETPicker**
	
<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/images/EMPIAR_10045_GIF/Training.gif" width="80%"> 
</div>

In practice, default parameters can give you a good enough result.

*Training parameter description:*

- Dataset name: e.g. SHREC_2021_train
- Train dataset ids: datasets used for training. You can click `Dataset list` to obain the dataset ids firstly.
- Val dataset ids: datasets used for validation. You can click `Dataset list` to obain the dataset ids firstly.
- Number of classes:  particle classes you want to pick
- Batch size: a number of samples processed before the model is updated. It is determined by your GPU memory, reducing this parameter might be helpful if you encounter out of memory error.
- Learning rate:  the step size at each iteration while moving toward a minimum of a loss function.
- Max epoch: total training epochs. The default value 60 is usually sufficient.
- GPU id: the GPUs used for training, e.g. 0,1,2,3 denotes using GPUs of 0-4. You can run the following command to get the information of available GPUs: nvidia-smi.
- Save Configs: save the configs listed above. The saved configurations contains all the parameters filled in this page, which can be directly loaded via *`Load configs`* next time instead of filling them again.

### **Inference of DeepETPicker**
	
<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/images/EMPIAR_10045_GIF/Inference.gif" width="80%"> 
</div>
	
In practice, default parameters can give you a good enough result.

*Inference parameter description:*

- Train Configs: path to the configuration file which has been saved in the `Training` step
- Networks weights: path to the model which has be generated in the `Training` step
- Patch size & Pad_size: tomogram is scanned with a specific stride S and a patch size of N in this stage, where S = N - 2Pad_size.
- GPU id: the GPUs used for inference, e.g. 0,1,2,3 denotes using GPUs of 0-4. You can run the following command to get the information of available GPUs: nvidia-smi.

### **Particle visualization and mantual picking**

<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/images/Visualization.gif" width="90%"> 
</div>

- *Showing Tomogram*

	You can click the `Load tomogram` button on this page to load the tomogram.

- *Showing Labels*
	
	After loading the coordinates file by clicking `Load labels`, you can click `Show result` to visualize the label. The label's diameter and width also can be tuned on the GUI.

- *Parameter Adjustment*

	In order to increase the visualization of particles, Gaussian filtering and histogram equalization are provided:
	- Filter: when choosing Gaussian, a Gaussian filter can be applied to the displayed tomogram, kernel_size and sigma(standard deviation) can be tuned to adjust the visual effects
	- Contrast: when choosing hist-equ, histogram equalization can be performed 
	

- *Position Slider*
	
	You can scan through the volume in x, y and z directions by changing their values. For z-axis scanning, shortcut keys of Up/Down arrow can be used.
	

