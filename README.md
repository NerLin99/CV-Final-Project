Only the modified file was upload

Original Repo can be visited via https://github.com/HengyiWang/spann3r?tab=readme-ov-file

The dataset can be visited via https://drive.google.com/drive/folders/1ws7vTH_suWDYconTHfezGzSLop5U7F5C?usp=share_link

Also, the setup of the code should follow the original repository. After that, please put model_modified.py under 
./spann3r/spann3r and replace the demo.py since original one has some bugs. And choose model when import. Please put the 
frame dataset under ./spann3r/examples and put the .ply file under ./pointcloud.

To predict the cloud point, should run demo.py by python demo.py --demo_path ./path/to/file --kf_every 10

For visualize the 3D cloud point, after fill the file path, run python viewer.py