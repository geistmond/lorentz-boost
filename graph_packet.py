from datetime import datetime

# Custom stuff
import packet_another_attempt as pkt 

import matplotlib
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np

from functools import lru_cache

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import evolutionary_algorithm as eva

save_animation, save_as_gif = False, True

current_frame = 0

x_min, x_max, num_points = -10, 10, 8000

#grid = np.linspace(x_min, x_max, num_points)

xo, ko, sigma = -5, 1, 1

fps, dur_of_video, dt = 10, 3, 0.15
total_frames = fps * dur_of_video

calculate_psi = pkt.calculate_psi
calculate_psi_list = pkt.calculate_psi_marseille

t = 0
axs = []


def evolve_psi_list(list_x: list[float], list_k: list[float], list_s: list[float]) -> list[float]:
    psi_lists = calculate_psi_list(list_x, list_k, list_s)
    """
    Something to flip the matrix around so that the lists of Psi values are 2D. 
    The value evolves as changes in dx/dt are rate-checked against 1D Lorenz boost.
    Rate-checking returns a new 1D array of the same type already, that's the other code.
    Final value in the array would be after it evolves toward that, I think.
    If it needs an optimizer, the file evolutionary_algorithm.py has a few like Ackley's.
    """
    return None





def plot_function(r_min, r_max)->None:
    x_axis = np.arange(r_min, r_max, 0.01)
    y_axis = np.arange(r_min, r_max, 0.01)
    x, y = np.meshgrid(x_axis, y_axis)
    results = eva.ackley_result(x, y) # how optimal is the function already with Lorenz boosts added?
    x2 = pkt.psi_x(x)
    y2 = pkt.psi_x(y)
    results2 = np.ndarray([x2, y2])
    results3 = eva.ackley_result(x2, y2)
    figure = pyplot.figure()
    axis = figure.add_subplot(projection='3d')
    axis.plot_surface(x2, y2, results, cmap='jet')
    pyplot.show()
    return None

# Run 3D plot
plot_function(1, 10)












@lru_cache
def get_animation(x: float, k: float, s: float) -> list:
    global pylot
    plt = pyplot
    global t, axs
    xo = x
    sigma = s
    ko = k
    w_max = np.power(2*np.pi*sigma**2, -0.25)
    not_moving_psi = calculate_psi(t, 0, 0, sigma)
    moving_psi = calculate_psi(t, xo, ko, sigma)

    matplotlib.rc('animation', html='html5') 
    plt.style.use('dark_background')

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.1, wspace=0.025, hspace=0)
    fig.suptitle("1-dim Gaussian Wave Packets", fontsize=20, color='w', x=0.55, y=0.99)

    axs[0].set_ylabel(r"|$\psi(x, t={:.1f})$|".format(t), fontdict={"fontsize": 25})
    axs[0].set_title(r"$k_o=0$", fontsize=20, x=0.1, y=0.935)
    axs[1].set_title(r"$k_o={}$".format(ko), fontsize=20, x=0.1, y=0.935)

    for ax in axs:
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(np.linspace(x_min, x_max, 4, endpoint=False))
        ax.set_ylim(-1.1*w_max, 1.1*w_max)
        ax.set_yticks([])

    not_moving_plot, = axs[0].plot(grid, np.abs(not_moving_psi), '-w', linewidth=3, label=r"|$\psi|$", zorder=3)
    not_moving_plot_r, = axs[0].plot(grid, np.real(not_moving_psi), '-r', linewidth=1, label=r"$\Re(\psi)$", zorder=2)
    not_moving_plot_i, = axs[0].plot(grid, np.imag(not_moving_psi), '-b', linewidth=1, label=r"$\Im(\psi)$", zorder=1)

    moving_plot, = axs[1].plot(grid, np.abs(moving_psi), '-w', linewidth=3, zorder=3)
    moving_plot_r, = axs[1].plot(grid, np.real(moving_psi), '-r', linewidth=1, zorder=2)
    moving_plot_i, = axs[1].plot(grid, np.imag(moving_psi), '-b', linewidth=1, zorder=1)

    for ax in axs:
        for side in ["top", "left", "right", "bottom"]:
            ax.spines[side].set_linewidth(1)
        ax.set_xlabel(r"$X$", fontdict={"fontsize": 15})
        
    axs[0].legend(loc=3)
    
    def update(frame):
        global t, axs, current_frame, save_animation, save_as_gif

        if current_frame % fps == 0:
            print(f"@ {current_frame//fps} second ...")

        not_moving_psi = calculate_psi(t, 0, 0, sigma)
        moving_psi = calculate_psi(t, xo, ko, sigma)

        not_moving_plot.set_ydata(np.abs(not_moving_psi))
        not_moving_plot_r.set_ydata(np.real(not_moving_psi))
        not_moving_plot_i.set_ydata(np.imag(not_moving_psi))

        moving_plot.set_ydata(np.abs(moving_psi))
        moving_plot_r.set_ydata(np.real(moving_psi))
        moving_plot_i.set_ydata(np.imag(moving_psi))

        axs[0].set_ylabel(r"|$\psi(x, t={:.1f}\sigma^2)$|".format(t), fontdict={"fontsize": 20})

        t += dt
        current_frame += 1

        return [not_moving_plot, not_moving_plot_r, not_moving_plot_i,
                moving_plot, moving_plot_r, moving_plot_i]
    anim = FuncAnimation(fig, update, frames=total_frames, blit=False, repeat=True, interval=1000/fps)
    if save_animation:
        anim.save("./1dimGaussian.mp4", writer="ffmpeg", fps=fps, dpi=160, bitrate=-1,
                metadata={
                    "title": "1 dimensional Gaussian Wave Packets (with Lorenz boost and conservation rules)",
                    "artist": "武満世阿弥 TAKEMITSU, Zeami",
                    "subject": "Quantum Wave Packets, Quantum Gravity, Marseille quantum interpretation"
                })
    elif save_as_gif:
        writer = PillowWriter(fps=fps, metadata={
            "title": "1 dimensional Gaussian Wave Packets (with Lorenz boost and conservation rules)",
            "artist": "武満世阿弥 TAKEMITSU, Zeami",
            "subject": "Quantum Mechanics, Quantum Gravity, Marseille quantum intepretation"})
        anim.save('./1dimGaussian.gif', dpi=80, writer=writer)
    else:
        plt.show()

#get_animation(xo, ko, sigma)