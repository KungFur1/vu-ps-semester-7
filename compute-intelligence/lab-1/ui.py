def getActivationFunction() -> str:
    activation_func:str = input("Select activation neuron activation function: ('heapsviside', 'sigmoid'): ")
    while (activation_func != "heapsviside" and activation_func != "sigmoid"):
        activation_func = input("Invalid input, must enter: 'heapsviside' or 'sigmoid': ")

    return activation_func

def getSearchType() -> str:
    search_type:str = input("Select search type: ('random', 'iterative'): ")
    while (search_type != "random" and search_type != "iterative"):
        search_type = input("Invalid input, must enter: 'random' or 'iterative': ")

    return search_type