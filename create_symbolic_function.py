import openturns as ot
import numpy as np

"""Symbolic function"""


def symboli_func(m, sig,  xvalues):
    # CREATING THE LIST OF VARIABLES
    XX = [f"x{i}" for i in range(1, m+1)]
    terms = []
    xvalues = xvalues.tolist()  # converting to list from numpy array
    xvalues.insert(0, sig)
    xvalues.insert(0, m)
    for each in XX:
        terms.append(each)

    # Create symbolic function with string expression
    expression = f" {m} + (3 * {sig} * ({m}^0.5)) - "

    for i in range(1, m+1):
        if i != m:
            expression = expression + f"x{i} - "
        if i == m:
            expression = expression + f"x{i}"

    symbolic_function = ot.SymbolicFunction(terms, [expression])
    return symbolic_function


'''use below only to check the above function'''

#values = np.ones(40)
#print(symboli_func(40, 0.2, values))
