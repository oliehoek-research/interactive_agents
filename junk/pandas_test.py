import numpy as np
import pandas

if __name__ == "__main__":
    max_range = 20
    stats = {
        "alpha": 1,
        "beta":  2,
        "gamma": 3,
    }

    series = {}
    for key, step in stats.items():
        indices = np.array(list(range(0, max_range, step)))
        values = np.arange(len(indices))
        series[key] = pandas.Series(values, indices)
    
    data = pandas.DataFrame(series)
    print("\n\nResulting Data Frame:\n")
    print(data)
