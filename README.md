To run the optimization problems, simply execute the main.py file from the command line in the same directory. A set of options will display:

      Choose problem:
                    FlipFlop: f,
                    Knapsack: k,
                    Queens: q:
         
After selection, another prompt will display:

      Choose algorithm:
                    Hill climb: R,
                    Annealing: A,
                    Genetic: G,
                    MIMIC: M: 
                 
The program will run and save the appropriate figures and graphs to a 'figures' subdirectory. It will also print a few benchmarking metrics.

To run the neural network weight optimization, run nn.py from the command line. A set of options will display:
      
       Select weight opt algorithm:
                    Hill climb: r,
                    Genetic: g,
                    Annealing: a: 
                    
The program will run and save the appropriate figures and graphs to a 'figures' subdirectory. It will also print a few benchmarking metrics. Dataset preprocessing is handled automatically. 

Requirements include: 
        pandas
        numpy
        scikit-learn
        mlrose-hiive
        matplotlib
