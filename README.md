# CS301-101-team7_SemanticSegmentation
# Team Members: Peter Akdemir, Joshua Quizon

## Milestone 1 (Environment Preparation)
**This list describes the steps that were taken to setup the project environment and complete milestone 1**
- created a Github repo with a README.md
- added directory 288 contents to this repo
- created a branch named milestone-1, committed changes and pushed to remote
- created a Colab notebook and pushed the copy to the milestone-1 branch
  - we cloned our repository for this project from github using the git clone command
  - the py files in dir 288 were analyzed and the following dependencies were installed using pip install:
    - sklearn
    - numpy
    - matplotlib
    - patchify
    - Pillow
    - segmentation_models
    - keras
    - opencv-python
    - pyngrok
  - it is important to note that Colab notebooks come pre-installed with PyTorch and TensorFlow, both of which are going to be used for this project
  - NNI was also installed, a ngrok web server was set up so that the localhost server for the NNI UI could be accessed publicly
  - steps that were followed were in the project webpage--we specifically used this [link](https://nni.readthedocs.io/en/stable/sharings/nni_colab_support.html)
    - pip install nni installs nni
    - wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip installs the ngrok zip, which then gets unzipped using unzip ngrok-stable-linux-amd64.zip
    - the Microsoft official NNI Github repo needs to be cloned, as it allows for configurating and serving NNI--repo cloned using git clone https://github.com/microsoft/nni.git nni_repo/nni
    - we then set the authentication token from our ngrok account using ngrok authtoken
    - NNI and ngrok sessions are started on the same port using nnictl create and get_ipython().system_raw
    - curl checks for the public ngrok url
  - the result set of steps leads to a creation of a local NNI session hosted on a public ngrok url: ![image](https://user-images.githubusercontent.com/98120760/198899088-86520d40-4173-4505-bd5d-60715131da17.png)
- the link to the milestone-1 branch will be submitted on Canvas
- the TA will be added as a collaborator and the Colab notebook will also be shared with him
