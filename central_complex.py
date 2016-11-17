import numpy as np
from scipy.special import expit
import memory_cell_model_class as mc

def gen_tb_tb_weights(weight=1.):
    """Weight matrix to map inhibitory connections from TB1 to other neurons"""
    W = np.zeros([n_tb1, n_tb1])
    sinusoid = -(np.cos(np.linspace(0, 2*np.pi, n_tb1, endpoint=False)) - 1)/2

    for i in range(n_tb1):
        values = np.roll(sinusoid, i)
        W[i, :] = values

    return weight * W


def noisy_sigmoid(v, slope, bias, noise=0.01):
    """Takes a vector v as input, puts through sigmoid and
    adds Gaussian noise. Results are clipped to return rate
    between 0 and 1"""
    sig = expit(v * slope - bias)

    if noise > 0:
        sig += np.random.normal(scale=noise, size=len(v))

    return np.clip(sig, 0, 1)


# PARAMETERS
n_tl2 = 16
n_cl1 = 16
n_tb1 = 8
n_tl2 = 16
n_cpu4 = 16
n_cpu1 = 16

tl2_prefs_default = np.tile(np.linspace(0, 2*np.pi, n_tb1, endpoint=False), 2)
noise_default = 0.1  # 10% additive noise


# TUNED PARAMETERS:
cpu4_mem_gain_default = 0.005
cpu4_mem_loss_default = 0.0026  # This is tuned to keep memory constant...

tl2_slope_tuned = 6.8
tl2_bias_tuned = 3.0

cl1_slope_tuned = 3.0
cl1_bias_tuned = -0.5

tb1_slope_tuned = 5.0
tb1_bias_tuned = 0

cpu4_slope_tuned = 5.0
cpu4_bias_tuned = 2.5

cpu1_slope_tuned = 6.0
cpu1_bias_tuned = 2.0


class CX:
    """Class to keep a set of parameters for a model together.
    No state is held in the class currently."""
    def __init__(self,
                 noise=noise_default,
                 tl2_slope=tl2_slope_tuned,
                 tl2_bias=tl2_bias_tuned,
                 tl2_prefs=tl2_prefs_default,
                 cl1_slope=cl1_slope_tuned,
                 cl1_bias=cl1_bias_tuned,
                 tb1_slope=tb1_slope_tuned,
                 tb1_bias=tb1_bias_tuned,
                 cpu4_slope=cpu4_slope_tuned,
                 cpu4_bias=cpu4_bias_tuned,
                 cpu4_mem=np.zeros(n_cpu4),
                 cpu4_mem_gain=cpu4_mem_gain_default,
                 cpu4_mem_loss=cpu4_mem_loss_default,
                 cpu1_slope=cpu1_slope_tuned,
                 cpu1_bias=cpu1_bias_tuned):

        # Default noise used by the model for all layers
        self.noise = noise

        # Weight matrices based on anatomy. These are not changeable!)
        self.W_CL1_TB1 = np.tile(np.eye(n_tb1), 2)
        self.W_TB1_TB1 = gen_tb_tb_weights()
        self.W_TB1_CPU1 = np.tile(np.eye(n_tb1), (2, 1))
        self.W_TB1_CPU4 = np.tile(np.eye(n_tb1), (2, 1))
        self.W_CPU4_CPU1 = np.array([
                [0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
                [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
                [0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0]
            ]).T
        self.W_CPU1_motor = np.array([-1, -1, -1, -1, -1, -1, -1, -1,
                                      1, 1, 1, 1, 1, 1, 1, 1])

        # The cell properties (for sigmoid function)
        self.tl2_slope = tl2_slope
        self.tl2_bias = tl2_bias
        self.tl2_prefs = tl2_prefs

        self.cl1_bias = cl1_bias
        self.cl1_slope = cl1_slope

        self.tb1_slope = tb1_slope
        self.tb1_bias = tb1_bias

        self.cpu4_slope = cpu4_slope
        self.cpu4_bias = cpu4_bias

        self.cpu4_mem_gain = cpu4_mem_gain_default
        self.cpu4_mem_loss = cpu4_mem_loss_default

        self.cpu1_slope = cpu1_slope
        self.cpu1_bias = cpu1_bias

        # Manually selected membrane conductance for the CPU4 memory neuron model.
        # The original 2-neuron memory model in the paper is using 1.0 but to create 
        # a small leakage I increased the conductance a bit. The range of acceptable 
        # conductance values is at least 1.0 to 1.0001.
        G = 1.00001
        # The 2-neuron memory model instances are stored in a list with 16 items.
        self.cpu4_vector = [mc.MemCell(Gm=G), mc.MemCell(Gm=G), mc.MemCell(Gm=G), mc.MemCell(Gm=G), 
                            mc.MemCell(Gm=G), mc.MemCell(Gm=G), mc.MemCell(Gm=G), mc.MemCell(Gm=G),
                            mc.MemCell(Gm=G), mc.MemCell(Gm=G), mc.MemCell(Gm=G), mc.MemCell(Gm=G), 
                            mc.MemCell(Gm=G), mc.MemCell(Gm=G), mc.MemCell(Gm=G), mc.MemCell(Gm=G)]
        
    def tl2_output(self, theta):
        """Just a dot product with preferred angle and current heading"""
        output = np.cos(theta - self.tl2_prefs)
        return noisy_sigmoid(output, self.tl2_slope, self.tl2_bias, self.noise)

    def cl1_output(self, tl2):
        """Takes input from the TL2 neurons and gives output."""
        return noisy_sigmoid(-tl2, self.cl1_slope, self.cl1_bias, self.noise)

    def tb1_output(self, cl1, tb1):
        """Ring attractor state on the protocerebral bridge."""
        prop_cl1 = 0.667
        prop_tb1 = 1.0 - prop_cl1
        output = (prop_cl1 * np.dot(self.W_CL1_TB1, cl1) -
                  prop_tb1 * np.dot(self.W_TB1_TB1, tb1))
        return noisy_sigmoid(output, self.tb1_slope, self.tb1_bias, self.noise)

    def cpu4_update(self, cpu4_mem, tb1, speed=1.0):
        """Memory neurons update."""
        cpu4_mem -= speed * self.cpu4_mem_loss

        # Each of the 2-neuron memory model instances is stored in a list of 16 items,
        # thus for updating it I do some silly trickery. Sorry...
        i = 0
        for value in cpu4_mem:
            self.cpu4_vector[i].set_value(value)
            i += 1
        
        # I kept your formula as it was.
        delta_values = (speed * self.cpu4_mem_gain *
                           np.dot(self.W_TB1_CPU4, 1.0 - tb1))
        # I use here a temporary cpu4 values list for this function to return.
        cpu4_mem_out = np.zeros(n_cpu4)
        # I update the proper cpu4 memory cells as well as copy the values to the temporary list.
        i = 0
        for value in delta_values:
            self.cpu4_vector[i].set_value(self.cpu4_vector[i].get_value() + value)
            cpu4_mem_out[i] = self.cpu4_vector[i].get_value()
            i += 1
        # I return the temporary list with the copies of the updated cpu4 memory values.
        return cpu4_mem_out

    def cpu4_output(self, cpu4_mem):
        """The output from memory neuron, based on current calcium levels."""
        # I use here a temporary cpu4 values list for this function to return.
        cpu4_mem_out = np.zeros(n_cpu4)
        # I read the cpu4 memory values and copy them to the temporary list to be returned.
        i = 0
        for cpu4_vector_element in self.cpu4_vector:
            cpu4_mem_out[i] = cpu4_vector_element.get_value()
            i += 1
        # Return the values after passing through the sigmoid. (I kept this as it was).
        return noisy_sigmoid(cpu4_mem_out, self.cpu4_slope,
                             self.cpu4_bias, self.noise)

    def cpu1_output(self, tb1, cpu4):
        """The memory and direction used together to get population code for
        heading."""
        inputs = np.dot(self.W_CPU4_CPU1, cpu4) - np.dot(self.W_TB1_CPU1, tb1)
        return noisy_sigmoid(inputs, self.cpu1_slope,
                             self.cpu1_bias, self.noise)

    def motor_output(self, cpu1):
        """outputs a scalar where sign determines left or right turn."""
        return np.dot(self.W_CPU1_motor, cpu1)
