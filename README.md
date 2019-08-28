# DS-Unit-2-Regression-Classification

You can work on Colab, or locally.

### Local instructions

1. Download and install [Anaconda Distribution](https://www.anaconda.com/distribution/). 

If you're on Windows, during installation, you'll be asked whether to "Add Anaconda to my PATH environment variable." Yes, check this box.

2. If you're on Windows, download and install [Git for Windows](https://gitforwindows.org/).

3. Open a command line terminal. (On Windows, Anaconda Prompt or Git Bash. On Mac, Terminal.)

4. [Clone](https://help.github.com/en/articles/cloning-a-repository) your fork of this repository, to create a local copy on your computer.

5. Navigate into the directory on your local computer where you cloned the repository.

6. Enter this command in your terminal:  

```
conda env create --name unit2 --file environment.yml --force
```

This creates a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments) for unit2, and installs the dependencies listed in the `environment.yml` file.

7. Activate the environment:

```
conda activate unit2
```

8. Launch Jupyter Notebook:

```
jupyter notebook
```
