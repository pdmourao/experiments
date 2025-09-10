import numpy as np
import os
from functools import reduce
from scipy.interpolate import make_interp_spline

def mathToPython(file, directory = None):
    if directory is None:
        fname = file
    else:
        fname = os.path.join(directory, file)
    with open(fname, 'rb') as f:
        depth = np.fromfile(f, dtype=np.dtype('int32'), count=1)[0]
        dims = np.fromfile(f, dtype=np.dtype('int32'), count=depth)
        data = np.transpose(np.reshape(np.fromfile(f, dtype=np.dtype('float64'),
                                                   count=reduce(lambda x, y: x * y, dims)), dims))
    return data

def mags_id(state, m, cutoff):
    if state == 'dis':
        pats = np.argmax(np.abs(m), axis=1)
        pats_mags = np.array([np.abs(m)[idx, pats[idx]] for idx in range(len(m))])
        if len(set(pats)) == len(pats) and np.all(pats_mags > cutoff):
            return True
        else:
            return False
    elif state == 'mix':
        if np.all(m > cutoff):
            return True
        else:
            return False
    else:
        return False

def gridvec_toplot(ax, state, m_array, limx0, limx1, limy0, limy1, cutoff, aspect='auto',
                   interpolate='x', linewidth = 1, **kwargs):
    all_samples, len_x, len_y, *rest = np.shape(m_array)
    success_array = np.zeros((all_samples, len_x, len_y))

    for idx_s in range(all_samples):
        for idx_x in range(len_x):
            for idx_y in range(len_y):
                if mags_id(state, m_array[idx_s, idx_x, idx_y], cutoff):
                    success_array[idx_s, idx_x, idx_y] = 1

    success_av = np.average(success_array, axis=0)

    vec_for_imshow = np.transpose(np.flip(success_av, axis=-1))

    input_str = '_'.join(
        [f'{key}{int(1000 * value)}' if not np.isinf(value) else f'{key}inf' for key, value in kwargs.items()])
    disname = f'discurve_{input_str}'
    mixname = f'mixcurve_{input_str}'

    cutoffname = f'magdiscurve_{input_str}_c{int(1000 * cutoff)}'
    filesfromM = [disname, mixname, cutoffname]

    colorsfromM = ['red', 'blue', 'black']
    stylesfromM = ['solid', 'solid', 'dashed']
    interp_funcs = []
    tr_lines = []

    for idx_f, file in enumerate(filesfromM):
        try:
            data = mathToPython(file, 'Transitions')
            if interpolate == 'x':
                interpolator = make_interp_spline(*data)
                interp_funcs.append(interpolator)
                x_values_smooth = np.linspace(start=data[0, 0], stop=data[0, -1], num=500, endpoint=True)
                if idx_f == 2:
                    x_values_smooth = np.array([x for x in x_values_smooth if
                                                interp_funcs[1](x) < interpolator(x) < interp_funcs[0](x)])
                    try:
                        idx_zero = np.where(interpolator(x_values_smooth) <= 0)[0][0]
                        x_values_smooth = x_values_smooth[:idx_zero]
                    except IndexError:
                        pass
                tr_lines.append([x_values_smooth, interpolator(x_values_smooth)])
            elif interpolate == 'y':
                interpolator = make_interp_spline(*(data[::-1]))
                interp_funcs.append(interpolator)
                y_values_smooth = np.linspace(start=data[1, 0], stop=data[1, -1], num=500, endpoint=True)
                if idx_f == 2:
                    y_values_smooth = [y for y in y_values_smooth if
                                       interp_funcs[1](y) < interpolator(y) < interp_funcs[0](y)]
                tr_lines.append([interpolator(y_values_smooth), y_values_smooth])

            else:
                tr_lines.append(data)
        except FileNotFoundError:
            tr_lines.append([[], []])

    c = ax.imshow(vec_for_imshow, cmap='Greens', vmin=0, vmax=1, aspect=aspect, interpolation='nearest',
                  extent=[limx0, limx1, limy0, limy1])

    ax.set_xlim(limx0, limx1)
    ax.set_ylim(limy0, limy1)

    for idx_line, line in enumerate(tr_lines):
        if interpolate:
            ax.plot(*line, color=colorsfromM[idx_line], linestyle=stylesfromM[idx_line], linewidth=linewidth)
        else:
            ax.scatter(*line, color=colorsfromM[idx_line])

    return c