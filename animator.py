import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import functools
import inquirer
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
        self.animation_data = []
        self.frame_range = []

        """
        animation figures
        """
        self.__animation_grid = np.zeros(self.shape, dtype=np.int8)
        Visual.range(self)
        Visual.convert(self)

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
        for individual_data in self.data_range:
            if self.starting_t in self.destroyed:
                indices = Visual.unravel(self, self.destroyed[self.starting_t])
                for i in range(len(indices[0])):
                    self.__animation_grid[indices[0][i]][indices[1][i]] = self.rp + 1
            self.starting_t += 1
            self.__animation_grid[(self.__animation_grid > 0) & (self.__animation_grid <= self.rp)] -= 1
            if individual_data == []:            # could use <if not individual_data.any():> but this is more readable.
                current_state = self.__animation_grid.copy()
                self.animation_data.append(current_state)
            else:
                indices = Visual.unravel(self, individual_data)
                for i in range(len(indices[0])):
                    self.__animation_grid[indices[0][i]][indices[1][i]] = self.rp
                current_state = self.__animation_grid.copy()
                self.animation_data.append(current_state)

    def range(self):
        """

        :return:
        """
        self.animation_data = []
        self.__animation_grid = np.zeros(self.shape, dtype=np.int8)
        count = 0
        state = 'NORMAL'
        entry = [0] * 2
        self.AF_states = ["All", "Custom"]

        for exc_list in self.file_data:
            if state == 'NORMAL' and len(exc_list) > (1.1 * self.shape[0]):
                    state = 'AF'
                    entry[0] = count
            if state == 'AF' and len(exc_list) <= (1.1 * self.shape[0]):
                    state = 'NORMAL'
                    entry[1] = count
                    self.AF_states.append(entry)
                    entry = [0] * 2
            count += 1

        range_inquire = [inquirer.List('Range', message="Please select a range to animate", choices=self.AF_states,),]
        answer = inquirer.prompt(range_inquire)

        if answer['Range'] == "All":
            self.data_range = self.file_data
            self.frames = len(self.data_range)
        elif answer['Range'] == "Custom":
            custom_range = [inquirer.Text('Start', message="Starting index"),
                            inquirer.Text('End', message="Ending index")]
            custom_answer = inquirer.prompt(custom_range)
            self.data_range = self.file_data[int(custom_answer['Start']):int(custom_answer['End'])]
            self.frames = len(self.data_range)
        else:
            start = int(answer['Range'][0])
            finish = int(answer['Range'][1])
            self.data_range = self.file_data[start:finish]
            self.frames = len(self.data_range)

        print(len(self.data_range))
        Visual.convert(self)

    def figure_init(self):
        """

        :return:
        """
        self.__animation_fig = plt.figure()
        self.__iteration_text = self.__animation_fig.text(0.84, 0.03, "Time Step: 1")
        self.__nu_text = self.__animation_fig.text(0.84, 0.09, r'$\nu$ = $%s$' % self.nu, fontsize=14)
        self.__delta_text = self.__animation_fig.text(0.84, 0.15, r'$\delta$ = $%s$' % self.delta, fontsize=14)
        self.__epsilon_text = self.__animation_fig.text(0.84, 0.21, r'$\epsilon$ = $%s$' % self.epsilon, fontsize=14)

    def init(self):
        """
        Initialises the animation. Used in show_animation function.
        :return:
        """
        self.__im = plt.imshow(np.zeros(self.shape, dtype=np.int8), cmap="gray", interpolation="nearest", vmin=0,
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
        self.__iteration_text.set_text("Time Step: {0}".format(t))
        return self.__im,

    def show_animation(self, fps=60):
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

        ani = animation.FuncAnimation(self.__animation_fig, functools.partial(Visual.animate, self),
                                      init_func=functools.partial(Visual.init, self), frames=self.frames+1, interval=1,
                                      repeat=False)
        file_name = '%s.mp4' % name_of_file
        ani.save(str(file_name), fps=60, extra_args=['-vcodec', 'libx264'])

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
