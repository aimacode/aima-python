

# `AIMA Code Installation Instructions` [![Build Status](https://travis-ci.org/aimacode/aima-python.svg?branch=master)](https://travis-ci.org/aimacode/aima-python) [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/aimacode/aima-python)

Implementations of all algorithms presented in the lecture in several programming languages are available online at [https://github.com/aimaTUM/aima-python](https://github.com/aimaTUM/aima-python). For most of the examples from the lecture we provide _Jupyter Notebooks_ that implement the example in Moodle. This should encourage you to debug the code for the examples step by step in order to develop a better understanding of the involved algorithms. The following two steps are required to set up a programming environment that allows you to execute the _Jupyter Notebooks_:

1. Installation of _Anaconda_
2. Download of the AIMA python code

# 1. Installation of Anaconda

To execute the _Jupyter Notebooks_ it is required to first install _Python_, _Jupyter_, and several standard python libraries. We recommend to use the _Anaconda_ environment, which installs all the above mentioned programs at once including the package manager conda. Conda is also used to create a virtual environment. _If you already use conda or want to use the python environment `venv`, or simply your home python distribution, feel free to do so and jump directly to point 2._

## 1.1 Installation on Linux

1. Download the Python 3 (currently 3.7) installer from: 

   https://www.anaconda.com/download/#linux
2. Go to the download folder your terminal and run: 

   `bash Anaconda-latest-Linux-x86_64.sh`
3. Follow the prompts on the installer screens. If you are unsure about any setting, accept the defaults. You can change them later. One of them is the auto `conda init`. It will initialize the conda base environment each time you start your terminal.
4. To make the changes take effect, close and then re-open your terminal.
5. To test your installation, in your terminal or Anaconda prompt, run the following command to list all installed packages: 

   `conda list`.

## 1.2 Installation on Windows

1. Download the Python 3 (currently 3.7) installer from: 

   https://www.anaconda.com/download/#windows
2. Double-click on the _.exe_ file.
3. Follow the instructions on the screen. If you are unsure about any setting, accept the defaults. You can change them later.
4. When installation is finished, form the _start_ menu, open the _Anaconda prompt_.
5. To test your installation, in your anaconda prompt, run the following command to list all installed packages: 

   `conda list`.

## 1.3 Installation on macOS

1. Download the Python 3 (currently 3.7) installer from: 

   https://www.anaconda.com/download/#macos
2. Double-click the _.pkg_ file.
3. Follow the prompts on the installer screens. If you are unsure about any setting, accept the defaults. You can change them later.
4. To make the changes take effect, close and then re-open your terminal.
5. To test your installation, in your terminal or anaconda prompt, run the following command to list all installed packages: 

   `conda list`.

## 1.4 How to use Anaconda
Anaconda distribution comes with more than 1,500 packages as well as the conda package and virtual environment manager. It also includes a GUI, Anaconda Navigator, as a graphical alternative to the command line interface (CLI). First time users might find helpful information in the [anaconda docs](https://docs.anaconda.com/anaconda/navigator/). As you will see in the following section we will use an Anaconda environment for package managing. An introduction to how to use Anaconda within the command line can be found [here](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html). The most important commands are:

1. Creating a new Anaconda environment: 

   `conda create --name <env_name>`
2. List all existing environments: 

   `conda info --envs`
3. Activate specific environment: 

   `conda activate <env_name>`
4. Install package: 

   `conda install <package_name>`
5. List all packages of current environment: 

   `conda list`

# 2.Download of the AIMA python code

Python implementations for the algorithms from the lecture are available on the repository at https://github.com/aimaTUM/aima-python. For installation, the following steps are required:

1. Create a new Anaconda environment. Here it is assumed that the environment is called AI_AIMA. 

   `conda create --name AI_AIMA python=3.7` 
  
   This step is not required. You can also work within the base environment, however in that case you __have to use Python 3.7, and not a newer version__. Additionally, using environments makes it easier to distribute your projects later on.
		
   If not yet activated, activate your environment. This step is needed each time you want to work within the environment. The current environment is indicated left to your computers name in the terminal.

   `conda activate AI_AIMA`

2. If git is not yet installed on your machine or in your _Anaconda_ environment, install it with the following command

   `conda install -c anaconda git`

3. Download the repository

   `git clone https://github.com/aimaTUM/aima-python`

4. Install pip within your conda environment:

   `conda install pip git`

5. Go inside the project folder and install the project requirements:

   `cd aima-python`

   `pip install -r requirements.txt --use-feature=2020-resolver`

   This will fetch all python packages needed. Unfortunately _conda_ has some issues installing _opencv_ so we used _pip_ in this case. Usually it is easier to just use `conda install` to install the needed packages.

   The `--use-feature=2020-resolver` may be necessary if your pip installation is not the latest version.

   The download may take a while. It case it fails to download some of the packages (for example due to connection problems), try to run the command again.

   If you see that one package cannot be installed, even after repeated tries through pip, you can also try to install it through conda, using one of the following commands:

   `conda install <NAME_OF_PACKAGE>`

   `conda install -c conda-forge <NAME_OF_PACKAGE>`

   where `<NAME_OF_PACKAGE>` is the name of the package whose installation fails.

6. Check if the packages are installed:

   `conda list`

7. Fetch the corresponding dataset from the _aima-data_ repository:

   `git submodule init`

   `git submodule update`

   The download of the set may take a while.

8. Run the test suite:

   `py.test`

   If all tests were successful, you are now ready to start!

   If not, look below in the known bugs/FAQ section.

# 3. Executing the Jupyter notebooks

For most of the examples from the lecture we provide _Jupyter Notebooks_ on Moodle. <span style="color:red"> To avoid issues with the relative file path we recommend to place these notebooks in the root folder of the _AIMA_ repository you downloaded in the previous step. </span> To start the _Jupyter_ web-interface simply type the following command into your terminal / anaconda prompt:

`jupyter notebook`
   
From the web-interface you can then easily open, modify, and execute the _Jupyter Notebooks_. Depending on your environment, it is possible that you have to install some additional python libraries. This can be done with the command:

(Note: Make sure you have activated your __project environment__ _`AI\_AIMA`_.)

`pip install <library name>`

or 

`conda install <library name>`

# 4. Known Bugs/FAQ

## 4.1 In Step 5 of installing the AIMA package, the `pip install` command returns an error, when installing `qpsolvers` or `quadprog`.

### For Windows
You may need the latest version of the C++ compiler provided by Visual Studio. You can download it [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

During the installation, you will be asked which options you want to install alongside Visual Studio. Make sure that you select the option "Desktop development with C++". If you missed this option and installed Visual Studio without it, you can still install it afterwards through the Visual Studio app (Start $\rightarrow$ Visual Studio Installer): In the box "Visual Studio Build Tools" you can select the option "Modify", and select there the module "Desktop development with C++".

After installation, reboot your system. This should fix the problem.

If this does not solve your problem, you may also try to install `qpsolvers` (which contains `quadprog`) through conda:

`conda install -c conda-forge quadprog`

The installation using pip will continue to fail if you try to do that again, but as long as `py.test` runs without problems, you do not need to worry about it.

### For iOS/macOS

This is a known bug related to an outdated version of Xcode. You can update Xcode by running the following command:

`xcode-select --install`

You can find more information [here](https://stackoverflow.com/questions/58364832/problems-installing-qpsolvers-on-mac)

## 4.2 When running `py.test` on Windows, two tests fail: `test_learning.py` and `test_learning4e.py`

This may be caused by an incorrect installation of `cvxopt` through pip. Instead, you can install it through conda:

`conda install -c conda-forge cvxopt`

Then, `py.test` should run without errors.

## 4.3 When running `py.test` on iOS/macOS, several `AttributeError` are returned by some file `plugin.py`,with the description `'Function' object has no attribute 'get_marker'`.

This is a known bug originating from an incompatibility with newer versions of `pytest`. It can be solved by installing an older version:

`pip install pytest==3.10.1`

You can find more information [here](https://stackoverflow.com/questions/54254337/pytest-attributeerror-function-object-has-no-attribute-get-marker)

## 4.4 On iOS/macOS, I have another error appearing, that was not listed here.

Often, mac-related issues may arise because you have not installed the latest update. Simply update your OS and/or your applications, and continue with the AIMA installation instructions.

If your problem is still not solved, don't hesitate to notify us of the problem on the Moodle forums.
