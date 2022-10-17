# Audio-Super-Resolution

KAIST GCLab Sanghyun Jung

-------------
## Environment
* Python 3.8
* PyTorch 1.10.1
* VCTK Dataset 0.92

-------------
## How to Run
1. Clone this repository.
```
git clone https://github.com/J-SangHyun/Audio-Super-Resolution.git
```

2. Install ```Python 3.8``` or higher from the [Python Webpage](https://www.python.org/).

3. Download ```VCTK Dataset 0.92``` from the [Dataset Link](https://datashare.ed.ac.uk/handle/10283/3443).

4. Unzip dataset and place it in ```dataset``` folder.
```
Audio-Super-Resolution/dataset/VCTK-Corpus-0.92/wav48_silence_trimmed/...
```

5. Install PyTorch + Cuda by following instruction at the [PyTorch Webpage](https://pytorch.org/).

6. Install dependencies.
```
pip install -r requirements.txt
```

7. If you want to train the model, run ```train.py```.

8. If you want to test the model, run ```test.py```.

9. You can check your test result in ```index.html``` after running test script.
