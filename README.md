# PIGE-DTI

To Train PIGE_DTI, run

    - python src/main.py 

if you want to use the data provided in the paper directly

    - python src/main.py --dataset <path_to_your_data>
    - <path_to_your_data>="bindingdb" or "biosnap" or "DrugBank" or "human"

if you want to set postive and negative samples ratio 1:1 or 1:10,
you can use data_proc.py by:

    while len(negative_pair_d) < len(positive_pair):
        i_d = random.choice(ind_d)
        i_p = random.choice(ind_p)
        if (i_d, i_p) not in positive_pair:
            negative_pair_d.append(i_d)
            negative_pair_p.append(i_p)

    while len(negative_pair_d) < len(positive_pair)*10:
        i_d = random.choice(ind_d)
        i_p = random.choice(ind_p)
        if (i_d, i_p) not in positive_pair:
            negative_pair_d.append(i_d)
            negative_pair_p.append(i_p)


### Requirements
PIGE_DTI is tested to work under Python 3.6.2  
The required dependencies for NASNet_DTI are Keras, PyTorch, TensorFlow, numpy, pandas, scipy, and scikit-learn.


