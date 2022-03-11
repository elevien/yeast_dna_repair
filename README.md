# Code for **Single-cell microfluidic analysis unravels individual cellular fates during Double-Strand Break Repair**

## Using this repository


### ``code``

This is where all the code to reproduce the Bayesian inference and simulations is contained. To reproduce the results, only files from this folder need to be run.

* ``data_processing``
The ``data_processing`` scripts will convert the output from the Matlab scripts into a cvs file which are used throughout the analysis. Running this notebook is unnecessary unless you have manipulated or generated new raw data.

* ``analysis`` The various functions used to perform inference and run stochastic simulations are contained here. In particular:  
    * ``functions.jl`` contained some useful Julia functions used throughout the analysis. You should not need to work with this file directly.
    * ``inference_pipeline.ipynb`` runs the Bayesian inference on molecular data and reproduces the posterior predictions
    * ``inference_test.ipynb`` tests the Bayesian inference on synthetic data
    * ``variability.ipynb`` generates the stochastic simulations which are compared to the histogram from microfluidic all_experiments





###  ``experimental_data``

The experimental data is contained in this folder 


### ``figures``

All figures produced by the analysis pipeline are saved here.






## LICENSE


```
Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
