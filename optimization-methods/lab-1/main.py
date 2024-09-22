# LSP=s2016020; a=2; b=2
import optimization
import alternate
import plots

def objectiveFunction(x:float):
    return ((x**2 - 2)**2 / 2) - 1

# interval_minimum = optimization.intervalMethod(objectiveFunction, 0, 10)
# print(interval_minimum)

# golden_minimum = optimization.goldenRatioSearchMethod(objectiveFunction, 0, 10)
# print(golden_minimum)

newtons_minimum = optimization.newtonsMethod(objectiveFunction)
print(newtons_minimum)


# interval = alternate.intervalMethod(objectiveFunction, 0, 10)
# golden = alternate.goldenRatioSearchMethod(objectiveFunction, 0, 10)
# newtons = alternate.newtonsMethod(objectiveFunction)
# plots.visualize(objectiveFunction, interval, golden, newtons)