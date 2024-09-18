# usefull-utils
maybe some usefull tools for production

# INFO


## llama-factory-plot

plot code whose input is `trainer_log*.jsonl` located in `llama-factory-plot` folder

## delta_rank

delta effective rank generator and plot code for qbadam inner method is in `delta_rank` folder

## 

to calculate the singlular ratio through formula:

$$ \eta = \frac{\sum^k_{i=0} s }  {\sum^n_{i=0} s }  $$

where s is the singular value calculated by using SVD, and ordered in descending. $ k $ means the k-th top singular value. n is the max rank of the matrix.