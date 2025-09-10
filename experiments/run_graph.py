import laboratory as lab
import funcs
import numpy as np
from matplotlib import pyplot as plt
from auxiliary import gridvec_toplot

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
          'split': False,
          'supervised': True,
          'error': 0,
          'av_counter': 1,
          'dynamic': 'sequential',
          }

experiment = lab.Experiment(directory = 'Data', func = funcs.disentanglement_lmb_r, **kwargs)
experiment.create()
missing =  experiment.samples_missing(50)

m, n, its = experiment.read()


cutoff = 0.8

fig, ax = plt.subplots(1)
c = gridvec_toplot(ax, 'dis', m, limx0 = x_values[0], limx1 = x_values[-1],
                   limy0 = y_values[0], limy1 = y_values[-1], cutoff = cutoff, beta = kwargs['beta'])


ax.set_xlabel(rf'${kwargs['x_arg']}$')
# ax.set_ylabel(rf'$\{y_arg}$')

beta_title = rf'\infty' if np.isinf(kwargs['beta']) else kwargs['beta']

ax.set_title(rf'$\beta = {beta_title},\quad$split')
plt.show()

fig, ax = plt.subplots(1)
vec_for_imshow = np.transpose(np.flip(np.max(its, axis=0), axis=-1))

ax.imshow(vec_for_imshow, cmap='Reds', vmin=0, vmax=kwargs['max_it'], aspect='auto', interpolation='nearest',
                      extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]])
ax.set_xlim(x_values[0], x_values[-1])
ax.set_ylim(y_values[0], y_values[-1])
plt.show()