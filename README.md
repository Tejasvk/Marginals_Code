# MarginalsCode (Under Construction)

Experiments from the SIGMOD 18 paper 
> [Marginal Release Under Local Differential Privacy](http://dimacs.rutgers.edu/~graham/pubs/papers/sigmod18.pdf)

 Dependency = Pandas, Numpy, Scipy & xxHash.
 To install dependencies, execute 
> pip install -r requirements.txt


You might have to install pip first. Then run 
> python compute_marginals.py

If you don't work with Python regularly then the easiest way to install all the dependencies is to use [Anaconda](https://www.anaconda.com/download/#download).
Choose Python 2.7 
 
Run compute_marginals.py with either "vary_d()" or "vary_k()". Note that the code may run for a while.  

To generate plots, run 
> python plotting_script.py

I have added some sample datafiles and plots in the directory. In case you don't want to download the large input data file "data/nyc_taxi_bin_sample.pkl", you can work with the synthetic data by uncommenting the relevant line in the function "driver".
 The code has been tested on a Linux machine with Python 2.7. I haven't tested thoroughly, but it should also work with Python 3. 
 
 Method description.
 
 * vary_d() -- Vary 'd', the number of questions users are required to answer keeping k, N, epsilon fixed. 
 * vary_k() -- Vary 'k', the number of questions aggregator is interested in keeping d, N, epsilon fixed.
 * driver_vary_all() -- Vary d, k, N, epsilon, N. 
 * driver() -- The main driver program that runs all algorithms. We are simulating clients and the aggregator part.
 * compute_marg() --  Receives corrected estimates from all algorithms and compute marginals. This method simulates the aggregator.

Please include a citation to the paper if you use this code.

>@inproceedings{CKS18,
  title = {Marginal Release Under Local Differential Privacy},
  author = {Graham Cormode and Tejas Kulkarni and Divesh Srivastava},  
  booktitle = {{ACM} {SIGMOD} International Conference on Management of Data ({SIGMOD})},
  year = {2018},
}

Contact: [Tejas Kulkarni](https://warwick.ac.uk/fac/sci/dcs/people/research/u1554597/) at abc@gmail.com, where a=tejas, b=vijay, c=kulkarni 

