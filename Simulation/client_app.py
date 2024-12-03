# client_app.py

"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import tensorflow as tf
from Simulation.task import define_discriminator, define_generator, define_gan, load_data

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

class GanClient(NumPyClient):
    def __init__(self, x_train, y_train, local_epochs, learning_rate, dataset, tam_ruido, classes, tam_batch, tam_img):
        self.x_train = x_train
        self.y_train = y_train
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.tam_ruido = tam_ruido
        self.tam_batch = tam_batch
        self.tam_img = tam_img
        self.dataset = dataset
        self.classes = classes
        self.generator = define_generator(noise_dim=self.tam_ruido,
                                          classes=classes,
                                          batch_size=self.tam_batch,
                                          img_size=self.tam_img)
        self.discriminator = define_discriminator(img_size=self.tam_img, 
                                                  classes=self.classes,
                                                  batch_size=self.tam_batch)
        self.model = define_gan(generator=self.generator,
                                discriminator=self.discriminator,
                                noise_dim=self.tam_ruido,
                                classes=self.classes)
        x_train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.x_train_ds = x_train_ds.batch(self.tam_batch)

    def get_parameters(self, config):
        # Combine weights into a single list
        return self.generator.get_weights() + self.discriminator.get_weights()

    def set_weights(self, weights):
        generator_num_weights = len(self.generator.get_weights())
        generator_weights = weights[:generator_num_weights]
        discriminator_weights = weights[generator_num_weights:]
        self.generator.set_weights(generator_weights)
        self.discriminator.set_weights(discriminator_weights)

    def fit(self, parameters, config):
        try:
          print("setando param")
          self.set_weights(parameters)
          embedding_layer = tf.keras.layers.Embedding(input_dim=classes, output_dim=classes)
          self.gen_losses = []
          self.disc_losses = []
          for i, data in enumerate(self.x_train_ds):
              # ---------------------
              #  Prepare Inputs
              # ---------------------

              # Generate random noise
              noise = tf.random.normal([batch_size, noise_dim])

              # Generate random labels for fake images (integer labels)
              x_fake_labels = tf.random.uniform([batch_size], minval=0, maxval=classes, dtype=tf.int32)

              # One-hot encode the fake labels
              x_fake_labels_one_hot = tf.one_hot(x_fake_labels, depth=classes)

              # One-hot encode the real labels from the dataset
              real_labels = data[1]
              real_labels_one_hot = tf.one_hot(real_labels, depth=classes)

              # Flatten the real images
              real_images = tf.reshape(data[0], [tf.shape(data[0])[0], -1])  # Shape: (batch_size, img_size * img_size)

              # ---------------------
              #  Train Discriminator
              # ---------------------

              # Set discriminator to trainable
              self.discriminator.trainable = True

              with tf.GradientTape() as disc_tape:
                  # Generate fake images
                  generated_images = self.generator([noise, x_fake_labels_one_hot], training=True)

                  # Discriminator output for real images
                  real_output = self.discriminator([real_images, real_labels_one_hot], training=True)

                  # Discriminator output for fake images
                  fake_output = self.discriminator([generated_images, x_fake_labels_one_hot], training=True)

                  # Calculate discriminator loss
                  disc_loss = discriminator_loss(real_output, fake_output)

              # Compute gradients and update discriminator weights
              gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
              discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

              # ---------------------
              #  Train Generator
              # ---------------------

              # Set discriminator to non-trainable
              self.discriminator.trainable = False

              with tf.GradientTape() as gen_tape:
                  # Generate fake images
                  generated_images = self.generator([noise, x_fake_labels_one_hot], training=True)

                  # Discriminator output for fake images
                  fake_output = self.discriminator([generated_images, x_fake_labels_one_hot], training=True)

                  # Calculate generator loss
                  gen_loss = generator_loss(fake_output)

              # Compute gradients and update generator weights
              gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
              generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

              # ---------------------
              #  Record Losses
              # ---------------------
              print('%d d=%.3f, g=%.3f' % (i + 1, disc_loss, gen_loss))
              #self.gen_losses.append(float(gen_loss.numpy()))
              #self.disc_losses.append(float(disc_loss.numpy()))
        # print(f"self.gen_losses: {self.gen_losses}")
        # print(f"self.disc_losses: {self.disc_losses}")
          self.model.get_layer(index=3).set_weights(self.discriminator.get_weights())
          self.model.get_layer(index=2).set_weights(self.generator.get_weights())
          return self.get_parameters(), len(self.x_train), {}
          # {"gen_losses": str(self.gen_losses), "disc_losses": str(self.disc_losses)}
        except Exception as e:
          print(f"Exception during fit: {e}")
          import traceback
          traceback.print_exc()
          return self.get_parameters(None), len(self.x_train), {}

    def evaluate(self, parameters, config):
        print("evaluating")
        try:
          self.set_weights(parameters)

          noise = tf.random.normal([batch_size, noise_dim])
          x_fake_labels = tf.random.uniform([batch_size], minval=0, maxval=classes, dtype=tf.int32)
          x_fake_labels_one_hot = tf.one_hot(x_fake_labels, depth=classes)

          generated_images = self.generator([noise, x_fake_labels_one_hot], training=False)

          real_labels = self.y_train[:300]
          real_labels_one_hot = tf.one_hot(real_labels, depth=classes)
          real_images = tf.reshape(self.x_train[:300], [tf.shape(self.x_train[:300])[0], -1])

          real_output = self.discriminator([real_images, real_labels_one_hot], training=False)
          fake_output = self.discriminator([generated_images, x_fake_labels_one_hot], training=False)

          loss = discriminator_loss(real_output, fake_output)

          #generate_and_save_images(self.generator, x_fake_labels_one_hot)

          np = float(loss.numpy())

          print("enviando eval")
          return np, len(self.x_train[:300]), {}
        except Exception as e:
          print(f"Exception during evaluate: {e}")
          import traceback
          traceback.print_exc()
          # Return a default loss value and proceed
          return float('inf'), len(self.x_train), {}
        

class FedVaeClient(NumPyClient):
    def __init__(self, trainloader, testloader, local_epochs, learning_rate, dataset, tam_ruido):
        self.net = Net(dataset=dataset)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.tam_ruido = tam_ruido
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        train(
            self.net,
            self.trainloader,
            epochs=self.local_epochs,
            learning_rate=self.lr,
            device=self.device,
            dataset=self.dataset
        )
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss = test(self.net, self.testloader, self.device, dataset=self.dataset)
        return float(loss), len(self.testloader), {}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    tam_batch = context.node_config["tam_batch"]

    # Read the run_config to fetch hyperparameters relevant to this run
    dataset = context.run_config["dataset"]  # Novo parâmetro
    trainloader, testloader = load_data(partition_id, num_partitions, dataset=dataset, tam_batch=tam_batch)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    tam_ruido = context.run_config["tam_ruido"]

    return FedVaeClient(trainloader, testloader, local_epochs, learning_rate, dataset, tam_ruido).to_client()


app = ClientApp(client_fn=client_fn)
