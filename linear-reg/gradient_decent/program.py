

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

def predict_using_sklean():
    df = pd.read_csv("data.csv")
    r = LinearRegression()
    r.fit(df[['math']],df.cs) # math is your feature and y is your target
    return r.coef_, r.intercept_
# when we use that fit method from LinearRegression from sklearn
# the code below is the internal implementation of it 
def gradient_descent(x,y):
    # u should image in the curve as 3d shaped bowel having a abs minima
#    at some point u wanna get that point
#   u assume m and c value as 0 initially
# y = mx+b and md-> partial derivative of d wrt x same with b
# learning rate is as u go near and near to abs minima
# the steps which u take will get smaller and smaller 
    m_curr = 0
    b_curr = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002

    cost_previous = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n)*sum([value**2 for value in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print ("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost, i))

    return m_curr, b_curr

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x,y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = predict_using_sklean()
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))
