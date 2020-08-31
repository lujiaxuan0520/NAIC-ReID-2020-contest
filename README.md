
0. Install dependencies by `pip install -r requirements.txt`.
1. To accelerate evaluation (10x faster), you can use cython-based evaluation code (developed by [luzai](https://github.com/luzai)). First `cd` to `eval_lib`, then do `make` or `python setup.py build_ext -i`. After that, run `python test_cython_eval.py` to test if the package is successfully installed.




2. change run.sh. For example, when the folder is '/home/msn/road/imagesnpy/train', you should change it to '--root /home/msn/road/imagesnpy'

3. bash run.sh



