# MEGA for Video Object Detection - Comparative Implementation

This repository is an adaptation of the official implementation found at [Scalsol/mega.pytorch](https://github.com/Scalsol/mega.pytorch) for the research paper ["Memory Enhanced Global-Local Aggregation for Video Object Detection"](https://arxiv.org/abs/2003.12063). The primary goal of this repository is for educational purposes and serves as the foundation for analyzing the performance of the MEGA approach against a single-frame baseline (BASE).


## Installation

To install the code, download and execute the [INSTALL.sh](INSTALL.sh) script in a folder where you want the necessary code and data to be downloaded. The bash file requires conda and Python to be installed. This implementation is designed for use with the following specifications:
- Linux OS (Ubuntu)
- CUDA 10.1

To run the bash file, execute the following commands:
1. `chmod +x /path/to/your/script/install.sh`
2. `/path/to/your/script/install.sh`

Additionally, apart from the code, you'll need the checkpoints for both the MEGA and BASE models, which you can download from [here for MEGA](https://drive.google.com/file/d/1ZnAdFafF1vW9Lnpw-RPF1AD_csw61lBY/view) and [here for BASE](https://drive.google.com/file/d/1W17f9GC60rHU47lUeOEfU--Ra-LTw3Tq/view). Save these checkpoint files inside the `mega.pytorch` folder after running `install.sh`.


## Data Preparation

To execute the comparison implementation, simply store your files for comparison in a folder of your choice (e.g., `input_folder`) inside the `mega.pytorch` directory. Additionally, create an output folder (`output_folder`) within `mega.pytorch` where you'd like the results to be stored. The input data can be either a video or an image. For videos, files in ".avi" or ".mp4" formats have been tested, while for images, only ".JPEG" format has been specifically tested; however, other formats should work as well.
We have included in the repo the data we have used to test and compare the two methods. The images of thedog and cats sequence can be found in the image_folder directory and the videso used in the videos_folder one.

### Inference Comparison between MEGA and BASE

The command line for conducting inference and testing both MEGA and BASE on input files offers two options. One is tailored for using images as input (first example), and the other is for video input (second example):

```
python demo/mega_vs_base.py \
    --suffix ".JPEG" \
    --visualize-path input_folder \
    --output-folder output_folder \
    [--output-video]
```

```
python demo/mega_vs_base.py \
    --video \
    --visualize-path input_folder/panama_canal.mp4 \
    --output-folder output_folder \
    [--output-video]
```

Please note the following:
1. The `output-video` parameter is optional. If specified, it denotes the output as an mp4 file; otherwise, it generates a sequence of images.
2. Terminal output includes the duration of each method and a brief time comparison analysis.
3. For video input, specify the `video` parameter along with the file path, not just the folder path.
4. Prior to running these commands, activate the conda environment MEGA using:
```
source activate MEGA
```
5. These commands are configured to run the code from the terminal within the `mega.pytorch` folder as root.
6. The output video is consistently named `comparison.mp4`. Rename it if you intend to preserve previous comparison outputs. Similarly, use a new `output_folder` to store comparisons separately for new inputs.