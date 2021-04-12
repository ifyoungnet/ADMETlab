* Conda environment
```
conda create --name admetlab python=2.7 --yes
conda activate admetlab
pip install scikit-learn==0.17.1
conda install pandas --yes
conda install scipy --yes
conda install -c rdkit rdkit --yes
```

* Example instructions for linux
* Download eigen2
* https://gitlab.com/libeigen/eigen/-/releases/2.0.17
* Untar 
* Download openbabel binary (example for linux)
* https://sourceforge.net/projects/openbabel/files/openbabel/2.3.1/openbabel-2.3.1.tar.gz/download
* Untar
* Navigate to cmake folder
```
cmake ../ -DPYTHON_BINDINGS=ON -DEIGEN2_INCLUDE_DIR=/home/bdw2292/eigen-2.0.17 -DCMAKE_INSTALL_PREFIX:PATH=/home/bdw2292/openbabel-2.3.1/
make -j 8
make install
```

* Install these packages below with conda envioronment activated (follow their instructions)
* https://github.com/gadsbyfly/PyBioMed
* https://github.com/cosylabiiit/chemopy.git


* Make a file called .admetlab.bashrc, add the following lines. Source before running admetlab scripts
```
conda activate admetlab
export PYTHONPATH=/home/bdw2292/openbabel-2.3.1/lib:$PYTHONPATH
export PYTHONPATH=/home/bdw2292/openbabel-2.3.1/lib/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/home/bdw2292/openbabel-2.3.1/lib:$LD_LIBRARY_PATH
```

* May need to change any statements such as from openbabel import pybel to "import pybel", (there will be an error)
