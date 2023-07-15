# <u>Intel Unnati Project - Tech Busters</u>

Social Distance Violation Detection Project created as a part of Intel Unnati Industrial Training 
<br><p>Our aim was to develop a model that could identify person in the frame 
and track them and find the distance between adjacent persons to detect the social
violation.<br>
This project uses MobileNetV2 with 2 SSD heads for person detection, OmniScaleNet(OSNet) for person reidentification and Deep Sort algorithm for person tracking

<p> Procedure to train the model are as follows - 

1. Clone | fork this project.

2. Create Virtual Environment
   ```python
   conda create -n <venv-name> python=3.10
   conda activate <venv-name>
   ```

3. Install requirements
    ```python
   pip install -r requirements.txt
   pip install openvino-dev
   ```
   You can run the file in devcloud to skip this step

4. Run the file <br>
    for optimized model:
   ```python
   python social_distancing_openvino.py
   ```
   Output of this file will be stored in demo_videos folder
   <br>For unoptimized model:
   ```python
   python social_distancing_yolov3-tiny.py
   ```