# Dynamic Shapley Value Computation

Code for implementation of ["Dynamic Shapley Value Computation"](https://github.com/ZJU-DIVER/DynamicShapley).

**Please cite the following work if you use this benchmark or the provided tools or implementations:**

```
@inproceedings{DBLP:conf/icde/zhang2023dynamic,
  author    = {Jiayao Zhang and
               Haocheng Xia and
               Qiheng Sun and
               Jinfei Liu and
               Li Xiong and
               Jian Pei and
               Kui Ren},
  title     = {Dynamic Shapley Value Computation},
  booktitle = {39th {IEEE} International Conference on Data Engineering, {ICDE} 2023,
               Anaheim, California, USA, April 3–7, 2023},
  publisher = {{IEEE}},
  year      = {2023}
}
```


### Prerequisites

- Python, NumPy, Scikit-learn, Tqdm

### Experiments in the Paper

They can be found in folder `paper_exps`.

### Basic Usage

To divide value fairly between individual train data points/sources which are dynamic, given the learning algorithm and a measure of performance for the trained model (test accuracy, etc.).

#### Run Example Experiments

```
$ python3 examples.py
```

If you have browser env, jupyter notebook is recommended.

```
$ jupyter_notebook examples.ipynb
```

### Documents

More detailed usages and code implementation can refer to the documents.

```
$ make docs
```

(\* Documents are powered by [Sphinx](https://github.com/sphinx-doc/sphinx).)

### License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
