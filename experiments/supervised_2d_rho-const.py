import laboratory as lab
import funcs
import numpy as np


m = 50
rho_values = np.linspace(start = 0, stop = 0.3, num = 50, endpoint = False)
x_values = np.sqrt(1 / (rho_values * m + 1))

y_values = np.linspace(start = 0, stop = 0.5, num = 50)

kwargs = {'x_arg': 'r',
          'y_arg': 'lmb',
          'x_values': x_values,
          'y_values': y_values,
          'neurons': 3000,
          'layers': 3,
          'k': 3,
          'm': m,
          'beta': np.inf,
          'h_norm': 0,
          'max_it': 100,
          'split': False,
          'supervised': True,
          'error': 0,
          'av_counter': 1,
          'dynamic': 'sequential',
          }

experiment = lab.Experiment(directory = 'Data', func = funcs.disentanglement_2d, **kwargs)
experiment.create()
for sample in experiment.samples_missing(50):
    experiment.run(sample, disable = False)

