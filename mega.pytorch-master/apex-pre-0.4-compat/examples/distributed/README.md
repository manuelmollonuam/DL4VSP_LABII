# Basic Multirpocess Example based on the MNIST example

This version of this examples requires APEx which can be installed from https://www.github.com/nvidia/apex. This example demonstrates how to modify a network to use a basic but effective distributed data parallel module. This parallel method is designed to easily run multi-gpu runs on a single node. It was created as current parallel methods integraded into pytorch can induce significant overhead due to python GIL lock. This method will reduce the influence of those overheads and potentially provide a benefit in performance, especially for networks with a significant number of fast running operations.

## Getting started
Prior to running please run
```pip install -r requirements.txt```

and start a single process run to allow the dataset to be downloaded (This will not work properly in multi-gpu. You can stop this job as soon as it starts iterating.).
```python main.py```

You can now the code multi-gpu with
```python -m apex.parallel.multiproc main.py ...```
adding any normal option you'd like.

## Converting your own model
To understand how to convert your own model to use the distributed module included, please see all sections of main.py within ```#=====START: ADDED FOR DISTRIBUTED======``` and ```#=====END:   ADDED FOR DISTRIBUTED======``` flags.

## Requirements
Pytorch master branch built from source. This requirement is to use NCCL as a distributed backend.
APEx installed from https://www.github.com/nvidia/apex
