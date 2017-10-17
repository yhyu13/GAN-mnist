import tensorflow as tf
import itertools
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from time import sleep
def plot_learning_curve(title,ys,labels):
    #Generate a simple plot of the test and training learning curve.

    plt.figure()
    plt.title(title)
    C = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.xlabel("Steps")
    plt.ylabel("Loss")

    for i in range(len(ys)):
        y = ys[i]
        if i >= len(C):
            i -= len(C)
        color = C[i]
        plt.plot(y, color=color,
                 label=labels[i])

    plt.legend(loc="best")
    plt.plot()
    plt.show()
    return plt

def plot_image(images: np.ndarray, label: str) -> None:
    plt.figure()
    plt.title(f'Label {label}')
    for i in range(len(images)):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i])
    plt.show()

def main():

    mnist = input_data.read_data_sets('mnist',
                                    one_hot=True)

    def get_batch(train,batch_size):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
          xs, ys = mnist.train.next_batch(batch_size)
        else:
          #rows = np.random.randint(1000,size=batch_size) 
          #xs, ys = mnist.test.images[:rows,:], mnist.test.labels[:rows,:]
          xs, ys = mnist.test.images[:1000,:], mnist.test.labels[:1000,:]
        return xs, ys

    def xavier_init(dims, uniform=True):
      """Set the parameter initialization using the method described.
      This method is designed to keep the scale of the gradients roughly the same
      in all layers.
      Xavier Glorot and Yoshua Bengio (2010):
               Understanding the difficulty of training deep feedforward neural
               networks. International conference on artificial intelligence and
               statistics.
      Args:
        n_inputs: The number of input nodes into each output.
        n_outputs: The number of output nodes for each input.
        uniform: If true use a uniform distribution, otherwise use a normal.
      Returns:
        An initializer.
      """
      n_inputs,n_outputs = dims
      if uniform:
        # 6 was used in the paper.
        init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform(shape=dims,minval=-init_range, maxval=init_range)
      else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = np.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal(shape=dims,stddev=stddev)

    with tf.Session() as sess:
        # Discriminator Net
        X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

        D_W1 = tf.Variable(xavier_init([784, 128]), name='D_W1')
        D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

        D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
        D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

        theta_D = [D_W1, D_W2, D_b1, D_b2]

        # Generator Net
        Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

        G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
        G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

        G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
        G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

        theta_G = [G_W1, G_W2, G_b1, G_b2]

        def sample_z(m,n):
            return np.random.uniform(-1.,1.,size=[m,n])

        def generator(z):
            G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
            G_prob = tf.nn.sigmoid(G_log_prob)

            return G_prob

        def discriminator(x):
            D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
            D_logit = tf.matmul(D_h1, D_W2) + D_b2
            D_prob = tf.nn.sigmoid(D_logit)

            return D_prob, D_logit

        G_sample = generator(Z)
        G_image = tf.reshape(G_sample,shape=[-1,28,28])
        D_real, D_logit_real = discriminator(X)
        D_fake, D_logit_fake = discriminator(G_sample)

        D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        G_loss = -tf.reduce_mean(tf.log(D_fake))

        '''
        # Alternative losses:
        # -------------------
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_real, tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_fake, tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_fake, tf.ones_like(D_logit_fake)))
        '''
        # Only update D(X)'s parameters, so var_list = theta_D
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
        # Only update G(X)'s parameters, so var_list = theta_G
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

        sess.run(tf.global_variables_initializer())

        # batch_size
        batch_size = 100
        ys = [[],[]]
        for batch_index in itertools.count():
            train_x,_ = get_batch(True,batch_size)
            feed_dict = {X:train_x,Z:sample_z(batch_size,100)}
            d_loss,g_loss,_,_ = sess.run([D_loss,G_loss,D_solver,G_solver],feed_dict=feed_dict)
            print(
                f'Batch {batch_index}, d_loss:{d_loss:.3f}, g_loss:{g_loss:.3f}')
            ys[0].append(d_loss)
            ys[1].append(g_loss)
            if batch_index == 5000:
                break
            
        plot_learning_curve('GAN Learning Curve',ys,['D','G'])
        i = 10
        while i>0:
            images = sess.run(G_image,feed_dict={Z:sample_z(9,100)})
            plot_image(images,'G fake data')
            i -= 1

            

if __name__ == '__main__':
    main()

        
