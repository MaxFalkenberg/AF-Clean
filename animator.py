import matplotlib.pyplot as plt
import matplotlib.animation as animation
import functools
import pickle
import numpy as np


class Visual:
    def __init__(self, file_name=None, shape=(200, 200)):
        """
        shape is assumed to be 200 by 200. Will probably need a way to record this in
        main.py and then read it here.

        :param file_name:
        :param shape:
        """

        if not file_name:
            raise ValueError("Need a file to gather data from.")
        self.shape = shape
        self.file_data = pickle.load(open("%s.p" % file_name, 'rb'))
        self.animation_data = []
        self.frames = len(self.file_data)

        """
        animation figures
        """
        self.__animation_fig = plt.figure()
        self.__iteration_text = self.__animation_fig.text(0, 0, "Time Step: 1")
        self.__animation_grid = np.zeros(shape, dtype=np.int8)
        self.__im = plt.imshow(self.__animation_grid, cmap="gray", interpolation="none", vmin=0, vmax=50)

    def unravel(self, data):
        """
        Function that unravels the list of lists into tuple data type:
        [list] --> (array,array)
        :param data:
        :return:
        """
        return np.unravel_index(data, self.shape)

    def convert(self):
        """
        Converts all the file data into arrays that can be animated.
        :return:
        """
        for individual_data in self.file_data:
            self.__animation_grid[(self.__animation_grid > 0) & (self.__animation_grid <= 50)] -= 1
            indices = Visual.unravel(self, individual_data)
            for i in range(len(indices[0])):
                self.__animation_grid[indices[0][i]][indices[1][i]] = 50
            current_state = self.__animation_grid.copy()
            self.animation_data.append(current_state)

    def init(self):
        """
        Initialises the animation. Used in show_animation function.
        :return:
        """
        self.__im.set_array(np.zeros(self.shape, dtype=np.int8))
        return self.__im,

    def animate(self, t):
        """
        Function that updates the animation figure. Used in show_animation.
        :param t:
        :return:
        """
        self.__im.set_array(self.animation_data[t])
        self.__iteration_text.set_text("Time Step: {0}".format(t))
        return self.__im,

    def show_animation(self):  # only plays once. Need to work out why.
        """
        Plays the animation (only plays it once).
        :return:
        """
        _ = animation.FuncAnimation(self.__animation_fig, functools.partial(Visual.animate, self),
                                    init_func=functools.partial(Visual.init, self), frames=self.frames, interval=1,
                                    repeat=False)
        plt.show()

    def save_animation(self, name_of_file):
        """
        Saves the animation as a mp4. opens in VLC.
        :param name_of_file:
        :return:
        """

        ani = animation.FuncAnimation(self.__animation_fig, functools.partial(Visual.animate, self),
                                      init_func=functools.partial(Visual.init, self), frames=self.frames, interval=1,
                                      repeat=False)
        file_name = '%s.mp4' % name_of_file
        ani.save(str(file_name), fps=30, extra_args=['-vcodec', 'libx264'])

    def show_frame(self, desired_frame):
        """
        Can view a specific frame in the animation.
        :param desired_frame:
        :return:
        """

        plt.imshow(self.animation_data[desired_frame], cmap="gray", interpolation='nearest', vmin=0, vmax=50,
                   origin='lower')
        plt.annotate("Time Step: %s" % desired_frame, xy=(1, 0), xycoords='axes fraction', fontsize=16,
                     xytext=(100, -20), textcoords='offset points', ha='right', va='top')
        c_bar = plt.colorbar()
        c_bar.ax.tick_params(labelsize=14)
        c_bar.set_label(r'$S(I,J)}$', fontsize=16, rotation=0, labelpad=25)
        plt.xlabel(r'$J$', fontsize=16, labelpad=12)
        plt.ylabel(r'$I$', fontsize=16, rotation=0, labelpad=15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
