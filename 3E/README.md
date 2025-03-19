## NOT A WORKING SOLUTION
## IT IS NOT ESSENTIAL TO SOLVE - NO REGULAR SOLUTION FOR THE TASK
## HIGHLY SPECIFIC AND COULD BE EXCLUSIVE

Directed Acyclic Workflow

Create a workflow framework, that reads and executes workflow in a framework described in a file in the following way (sample):
0 --> 1
0 --> 2
1 --> 2
2 --> 3
1 --> 3
Each process has some work to do, which depends on results of previous processes:
#1 depends on the work of #0. #2 depends on #0 and #1. #3 depends on #1 and #2. Processes should print a message when doing work, and send a token around with MPI to emulate data transfer. Using these tokens, the system should detect, if there is a cyclic dependency in the graph, so execution should be terminated with an informative error message even with a problematic input file. Data file should be read by process #0, and information distributed accordingly to the other processes.
The program should be tested on a larger and a cyclic dependency graph.
