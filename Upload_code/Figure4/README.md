# README

Figure 4. displays the the propsed observable shift method can estimate high order information more accurately than doing nothing.

This numerical experiment based on "PaddleQuantum" platform. Sepcifically, we require the following packages:

- python=3.10.12
- paddle-quantum=2.4.0
- scipy=1.10.1
- openfermion=1.5.1
- matlabengine=9.14.3
- matplotlib=3.7.1

Please execute the plot.py directly.

The file contains the following contents:

- QNN_QPS_EXPTOOL.py: The tools are going to be used in HubbardGenerator.py
- HubbardGenerator.py: Generate the ground state of Hubbard model, and store the state in "ground_state.npy"
- ground_state.npy: The ground state we are going to investigate.
- tensor.m: A function to realize tensor.
- LShiftMatrix.m: Generate a matrix that left-shifts subsystems
- JDGenerator.m: The tools are going to used in plot.py
- plot.py: Execute the experiment and plot the figure. The figure is saved as Hubbar_Simulation.png