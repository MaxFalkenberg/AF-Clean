import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import functools
import numpy as np


class Visual:
    def __init__(self, file_name=None):
        """
        shape is assumed to be 200 by 200. Will probably need a way to record this in
        main.py and then read it here.
        :param file_name:
        """

        if not file_name:
            raise ValueError("Need a file to gather data from.")
        origin = np.load("%s.npy" % file_name)
        self.file_data = origin[0]
        self.shape = origin[1]
        self.rp = origin[2]
        self.nu = origin[3]
        self.delta = origin[4]
        self.epsilon = origin[5]
        self.state_history = origin[6]
        self.destroyed = origin[9]
        self.starting_t = origin[10]
        self.pulse_rate = origin[11]
        self.animation_data = []
        self.frame_range = []
        self.starting_frame_t = None
        # Initial refractory data for applying to animations.
        self.refractory_data = []
        self.raw_refractory_data = []

        self.animation_grid = np.zeros(self.shape, dtype=np.int8)

        Visual.range(self)

    def unravel(self, data):
        """
        Function that unravels the list of lists into tuple data type:
        [list] --> (array,array)
        :param data:
        :return:
        """
        return np.unravel_index(data, self.shape)

    def init_grid(self):
        """

        :return:
        """
        self.animation_grid = np.zeros(self.shape, dtype=np.int8)
        count = self.rp
        for refractory_data in self.raw_refractory_data[::-1]:
            if refractory_data == []:
                pass
            else:
                indices = Visual.unravel(self, refractory_data)
                for i in range(len(indices[0])):
                    self.__animation_grid[indices[0][i]][indices[1][i]] = count
            count -= 1

    def convert(self, data, output_list):
        """
        Converts all the file data into arrays that can be animated.
        :return:
        """
        count = self.starting_t
        for individual_data in data:
            self.__animation_grid[(self.__animation_grid > 0) & (self.__animation_grid <= self.rp)] -= 1
            if individual_data == []:            # could use <if not individual_data.any():> but this is more readable.
                current_state = self.__animation_grid.copy()
                output_list.append(current_state)
            else:
                indices = Visual.unravel(self, individual_data)
                for i in range(len(indices[0])):
                    self.__animation_grid[indices[0][i]][indices[1][i]] = self.rp
                current_state = self.__animation_grid.copy()
                output_list.append(current_state)

            if count in self.destroyed:
                destroyed_set = self.destroyed[count]
                for destroyed_index in range(len(destroyed_set)):
                    indices = Visual.unravel(self, destroyed_set[destroyed_index])
                    for i in range(len(indices[0])):
                        self.__animation_grid[indices[0][i]][indices[1][i]] = self.rp + 10
            count += 1

    def range(self):
        """

        :return:
        """
        self.animation_data = []
        self.refractory_data = []
        self.__animation_grid = np.zeros(self.shape, dtype=np.int8)
        count = 0
        state = 'NORMAL'
        entry = [0, -1]
        self.AF_states = list()

        for exc_list in self.file_data:

            if state == 'NORMAL' and len(exc_list) > (1.1 * self.shape[0]):
                state = 'AF'
                entry[0] = count

            if state == 'AF' and len(exc_list) <= (1.1 * self.shape[0]):
                state = 'TEST'
                test_count = count

            if state == 'TEST' and len(exc_list) > (1.1 * self.shape[0]):
                state = 'AF'

            if state == 'TEST' and (count - test_count == 2 * self.pulse_rate):
                state = 'NORMAL'
                entry[1] = count
                self.AF_states.append(entry)
                entry = [0] * 2

            if (state == 'TEST' or state == 'AF') and (count + 1) == len(self.file_data):
                entry[1] = count + 1
                self.AF_states.append(entry)
            count += 1

        print '\n'
        print "Simulation Length: %s" % len(self.file_data)
        print '\n'
        print "RANGE OPTIONS"
        print "------------------------------------------------"
        print "Basic ranges:"
        print "all"
        print "custom"
        print "\n"
        print "Specific AF ranges:"
        for i in self.AF_states:
            print i
        print "------------------------------------------------"
        print '\n'
        answer = raw_input("Please select a range to animate (a/c): ")

        #  Whole animation data
        if answer == "a":
            self.data_range = self.file_data
            self.raw_refractory_data = []
            self.frames = len(self.data_range)
            self.starting_frame_t = 0

        #  Custom Range
        elif answer == "c":
            start = int(raw_input("Start: "))
            end = int(raw_input("End: "))
            self.data_range = self.file_data[start:end]
            refractory_start = start - self.rp
            refractory_end = start
            if refractory_start < 0:
                refractory_start = 0
            self.raw_refractory_data = self.file_data[refractory_start:refractory_end]
            self.frames = len(self.data_range)
            self.starting_frame_t = start

        if self.raw_refractory_data:
            Visual.init_grid(self)

        Visual.convert(self, self.data_range, self.animation_data)

    def figure_init(self):
        """

        :return:
        """
        self.__animation_fig = plt.figure()
        self.__iteration_text = self.__animation_fig.text(0.02, 0.02, "Time Step: 1")
        self.__nu_text = self.__animation_fig.text(0.84, 0.84, r'$\nu$ = $%s$' % self.nu, fontsize=14)
        self.__delta_text = self.__animation_fig.text(0.84, 0.78, r'$\delta$ = $%s$' % self.delta, fontsize=14)
        self.__epsilon_text = self.__animation_fig.text(0.84, 0.72, r'$\epsilon$ = $%s$' % self.epsilon, fontsize=14)

    def init(self):
        """
        Initialises the animation. Used in show_animation function.
        :return:
        """
        c_map_custom = matplotlib.cm.gray
        c_map_custom.set_over('r')
        self.__im = plt.imshow(np.zeros(self.shape, dtype=np.int8), cmap=c_map_custom, interpolation="nearest", vmin=0,
                               vmax=self.rp,
                               origin="lower", animated=True)
        return self.__im,

    def animate(self, t):
        """
        Function that updates the animation figure. Used in show_animation.
        :param t: Time step.
        :return:
        """
        self.__im.set_array(self.animation_data[t])
        self.__iteration_text.set_text("Time Step: {0}".format(t+self.starting_frame_t))
        return self.__im,

    def show_animation(self, fps=30):
        """
        Shows the animation.
        :param fps: frames per second for the playback.
        :return:
        """
        Visual.figure_init(self)
        _ = animation.FuncAnimation(self.__animation_fig, functools.partial(Visual.animate, self),
                                    init_func=functools.partial(Visual.init, self),
                                    frames=self.frames,
                                    interval=1000/fps,
                                    repeat=False)
        plt.show()
        plt.close("all")

    def save_animation(self, name_of_file):
        """
        Saves the animation as a mp4. opens in VLC.
        :param name_of_file:
        :return:
        """
        Visual.figure_init(self)
        ani = animation.FuncAnimation(self.__animation_fig, functools.partial(Visual.animate, self),
                                      init_func=functools.partial(Visual.init, self), frames=self.frames, interval=1,
                                      repeat=False)
        file_name = '%s.mp4' % name_of_file
        ani.save(str(file_name), fps=60, extra_args=['-vcodec', 'libx264'])
        plt.close("all")

    def show_frame(self, desired_frame):
        """
        Can view a specific frame in the animation.
        :param desired_frame:
        :return:
        """

        c_map_custom = matplotlib.cm.gray
        c_map_custom.set_over('r')
        plt.imshow(self.animation_data[desired_frame], cmap=c_map_custom, interpolation='nearest', vmin=0, vmax=self.rp,
                   origin='lower')
        plt.annotate("Time Step: %s" % desired_frame, xy=(1, 0), xycoords='axes fraction', fontsize=16,
                     xytext=(100, -20), textcoords='offset points', ha='right', va='top')
        c_bar = plt.colorbar()
        c_bar.ax.tick_params(labelsize=14)
        c_bar.set_label(r'$S(I,J)$', fontsize=16, rotation=0, labelpad=25)
        plt.xlabel(r'$J$', fontsize=16, labelpad=12)
        plt.ylabel(r'$I$', fontsize=16, rotation=0, labelpad=15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
