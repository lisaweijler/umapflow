# umap-hdbscan
Automatic MRD detection in pediatric AML patients with UMAP and HDBSCAN. 

Official implementation of our work: *"UMAP based anomaly detection for minimal residual disease quantification within acute myeloid leukemia"*
by Lisa Weijler, Florian Kowarsch, Matthias Wödlinger, Michael Reiter, Margarita Maurer-Granofszky, Angela Schumich and Michael Dworzak. 


## Usage
### Create virtual env and setup project
```shell
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install cython
$ pip install umap-learn
$ pip install hdbscan
$ python setup.py install
```

### Config file 

A template for the config file is given by  ``config/example-experiment.json``; it specifies where to load the data from as well as the hyperparameters used for UMAP and HDBSCAN. The *data_loader* config specifies path to *.txt files, which hold the paths to raw flow cytometry files. Those file are preloaded and saved using pickle into a directory specified in "fast_preload_dir". If the fast preload directories already exist, the preloaded files are used directly. 

Here a few preloaded example files are given to test the repository. To load raw flowcytometry file we recommend using the package ``flowme``. 

### To run 

```shell
$ python main.py --config config/example_experiment.json
```

## Citation

If you use this project please consider citing our work

```
@article{weijler2022umap,
  title={UMAP based anomaly detection for minimal residual disease quantification within acute myeloid leukemia},
  author={Weijler, Lisa and Kowarsch, Florian and W{\"o}dlinger, Matthias and Reiter, Michael and Maurer-Granofszky, Margarita and Schumich, Angela and Dworzak, Michael N},
  journal={Cancers},
  volume={14},
  number={4},
  pages={898},
  year={2022},
  publisher={MDPI}
}
```








