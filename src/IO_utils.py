

from datetime import datetime
import os
def save(plot_list):
    folder_name = "plots/plots_on_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    os.mkdir(folder_name)
    for plot_object in plot_list:
        plot_object.savefig(folder_name)
    print("plots are saved")