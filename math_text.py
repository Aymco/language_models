
import random
import numpy as np


from datasets import load_dataset
import json
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("OpenLLM-France/Claire-Dialogue-French-0.1")

# Print the first 5 examples of the training set
for el in ds['train'][:5]:
    print(el)
    for k, v in el.items():
        print(f"{k}: {v}")

    print("\n\n")






# Generate math expressoins
iterations = 0
def generate_math_expression(n_digits=1, n_out_digits=2, n_decimals=0, n_operation=5,  op_list=['*', '/','+', '-']):
    """
    Generate a math expression with n_terms terms, n_ops operators, and n_digits digits.
    """
    global iterations
    max_output = 10**(n_out_digits) - 1
    
    get_random_number = lambda: random.randint(0, 10**(n_digits + n_decimals) - 1) / 10**n_decimals
    if n_decimals == 0: get_random_number = lambda: random.randint(0, 10**(n_digits) - 1)

    # generate the first term
    value = None    
    while value is None or not (0 <= value <= max_output) or value != np.round(value, n_decimals):
        
        ops = [random.choice(op_list) for _ in range(n_operation)]
        # if 2 consecutive ** : continue
        if [ops[i] == ops[i+1] == '**' for i in range(len(ops) - 1)].count(True) > 0:
            continue

        terms = [get_random_number() for _ in range(n_operation + 1)]
        # avoid division by zero
        terms = [terms[0]] + [1 if terms[i+1] == 0 and op in ['/',"%"]  else terms[i+1] for i, op in enumerate(ops)]
        
        exp = ''.join([str(terms[0])] + [op + str(t) for op, t in zip(ops, terms[1:])])
        
        
        try:
            value = eval(exp)
        except Exception as e:
            print(f"Error: {exp} : {e}")
            value = None
        iterations += 1
    
        
    value = np.round(value, n_decimals)
    if n_decimals == 0:
        value = int(value)
    
    if value != eval(exp):
        print(f"Error: {exp} != {value} but = {eval(exp)}")
        exit()
    return exp + '=' + str(value)



for i in range(1):
    #out = generate_math_expression(n_operation=random.randint(1, 5), n_out_digits=3, op_list=['*', '/','+', '-', '**', '%'])
    #out = generate_math_expression(n_operation=random.randint(1, 5), n_out_digits=3, op_list=['*', '/','+', '-', '**'])
    out = generate_math_expression(n_operation=random.randint(1, 5), n_out_digits=2, op_list=['+'])
    print(out)

print(f"Total iterations: {iterations} : {iterations/5000}")