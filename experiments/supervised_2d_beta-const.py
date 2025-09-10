import laboratory as lab
import funcs
import numpy as np

split = True
m_per_layer = 50
rho_values = np.linspace(start = 0, stop = 0.3, num = 50, endpoint = False)

layers = 3
if split:
    m = layers * m_per_layer
else:
    m = m_per_layer
x_values = np.sqrt(1 / (rho_values * m_per_layer + 1))

y_values = np.linspace(start = 0, stop = 0.5, num = 50)

kwargs = {'x_arg': 'r',
          'x_values': x_values,
          'lmb': y_values,
          'neurons': 3000,
          'layers': 3,
          'k': 3,
          'm': m,
          'beta': np.inf,
          'h_norm': 0,
          'max_it': 100,
          'split': split,
          'supervised': True,
          'error': 0,
          'av_counter': 1,
          'dynamic': 'sequential',
          }

experiment = lab.Experiment(directory = 'Data', func = funcs.disentanglement_lmb_r, **kwargs)
experiment.create()
for sample in experiment.samples_missing(50):
    experiment.run(sample, disable = True)


