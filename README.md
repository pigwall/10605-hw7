# 10605-hw7

run the code with following command line in autolab:

spark-submit dsgd_mf.py <num_factors> <num_workers> <num_iterations> \
<beta_value> <lambda_value> \
<inputV_filepath> <outputW_filepath> <outputH_filepath>

for example:

/afs/cs.cmu.edu/project/bigML/spark-1.3.0-bin-hadoop2.4/bin/spark-submit dsgd_mf.py 20 6 30 0.7 0.1 nf_subsample.csv w.csv h.csv

to run on AWS:(in the master node)

./spark/bin/spark-submit --master spark://<internal id>:7077 dsgd_mf.py 20 6 30 0.6 0.1 /data/nf_subsample.csv w.csv h.csv
