import pandas as pd
from io import StringIO

# Provided data
data = """
zscore,scaling_method,imputation_strategy,model,accuracy,precision,recall,f1
mean,random_forest,0.9491666666666667,0.23447401774397972,0.9685863874345549,0.37755102040816324
mean,logistic_regression,0.9104166666666667,0.11022927689594356,0.6544502617801047,0.18867924528301888
median,random_forest,0.9419166666666666,0.20161290322580644,0.9776536312849162,0.3342884431709646
median,logistic_regression,0.9001666666666667,0.09595559080095163,0.6759776536312849,0.16805555555555557
most_frequent,random_forest,0.9465833333333333,0.2491103202846975,0.963302752293578,0.3958529688972667
most_frequent,logistic_regression,0.8775833333333334,0.09144350097975179,0.6422018348623854,0.16009148084619781
zero,random_forest,0.9430833333333334,0.20823529411764705,0.946524064171123,0.3413693346190935
zero,logistic_regression,0.7908333333333334,0.045401174168297455,0.6203208556149733,0.08460977388767323
random,random_forest,0.9408333333333333,0.19129438717067582,0.9766081871345029,0.31992337164750956
random,logistic_regression,0.8951666666666667,0.08416220351951033,0.6432748538011696,0.14884979702300405
mean,minmax,random_forest,0.94625,0.21170610211706103,0.9340659340659341,0.3451776649746193
mean,minmax,logistic_regression,0.9905833333333334,0.6650717703349283,0.7637362637362637,0.710997442455243
median,minmax,random_forest,0.951,0.2558139534883721,0.9428571428571428,0.4024390243902439
median,minmax,logistic_regression,0.99,0.6956521739130435,0.7619047619047619,0.7272727272727272
most_frequent,minmax,random_forest,0.9483333333333334,0.2222222222222222,0.9405405405405406,0.3595041322314049
most_frequent,minmax,logistic_regression,0.9898333333333333,0.6425339366515838,0.7675675675675676,0.6995073891625616
zero,minmax,random_forest,0.947,0.24311377245508983,0.9806763285024155,0.3896353166986564
zero,minmax,logistic_regression,0.98925,0.6681034482758621,0.748792270531401,0.7061503416856493
random,minmax,random_forest,0.9391666666666667,0.21172638436482086,0.9798994974874372,0.34821428571428575
random,minmax,logistic_regression,0.9895833333333334,0.6594827586206896,0.7688442211055276,0.7099767981438515
"""

# Convert the data string to a DataFrame
df = pd.read_csv(StringIO(data))

# Save the DataFrame to a CSV file
df.to_csv('output.csv', index=False)
